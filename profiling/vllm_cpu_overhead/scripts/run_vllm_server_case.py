#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

import httpx


class OptionalNvtxRange:
    def __init__(self, enabled: bool, message: str):
        self.enabled = enabled
        self.message = message
        self.range_id = None
        self.nvtx = None

    def __enter__(self) -> "OptionalNvtxRange":
        if not self.enabled:
            return self
        try:
            import nvtx  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            print(f"NVTX_UNAVAILABLE {exc}", flush=True)
            return self
        self.nvtx = nvtx
        self.range_id = nvtx.start_range(message=self.message)
        print(f"NVTX_RANGE_START {self.message}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.nvtx is not None and self.range_id is not None:
            self.nvtx.end_range(self.range_id)
            print(f"NVTX_RANGE_END {self.message}", flush=True)


def build_prompt(i: int) -> str:
    return (
        "You are a concise technical assistant. "
        f"Request {i}: explain in one paragraph how a GPU inference scheduler "
        "handles autoregressive decoding, KV cache blocks, and token sampling."
    )


async def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(f"{base_url}/health")
                if response.status_code == 200:
                    return
                last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            except Exception as exc:  # noqa: BLE001 - status is surfaced on timeout.
                last_error = exc
            await asyncio.sleep(1.0)
    raise TimeoutError(f"vLLM server did not become ready within {timeout_s}s: {last_error}")


async def one_completion(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    request_id: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    payload = {
        "model": model,
        "prompt": build_prompt(request_id),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    async with semaphore:
        start = time.perf_counter()
        response = await client.post(f"{base_url}/v1/completions", json=payload)
        latency_s = time.perf_counter() - start
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage", {})
    return {
        "request_id": request_id,
        "latency_s": latency_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
    }


async def run_load(
    base_url: str,
    model: str,
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    timeout_s: float,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=max(concurrency * 2, 16),
                          max_keepalive_connections=max(concurrency, 8))
    async with httpx.AsyncClient(timeout=timeout_s, limits=limits) as client:
        tasks = [
            asyncio.create_task(
                one_completion(client, base_url, model, i, max_tokens, semaphore)
            )
            for i in range(total_requests)
        ]
        return await asyncio.gather(*tasks)


def terminate_process(proc: subprocess.Popen[str], timeout_s: float = 30.0) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout_s)


def start_pyspy(args: argparse.Namespace, target_pid: int) -> subprocess.Popen[str] | None:
    if not args.pyspy_out:
        return None
    pyspy_out = Path(args.pyspy_out)
    pyspy_out.parent.mkdir(parents=True, exist_ok=True)
    pyspy_log = Path(args.pyspy_log) if args.pyspy_log else pyspy_out.with_suffix(".log")
    pyspy_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.pyspy_bin,
        "record",
        "-p",
        str(target_pid),
        "--subprocesses",
        "-o",
        str(pyspy_out),
        "-f",
        "raw",
        "-r",
        str(args.pyspy_rate),
    ]
    if args.pyspy_duration_s > 0:
        cmd.extend(["--duration", str(int(args.pyspy_duration_s))])
    if args.pyspy_native:
        cmd.append("--native")
    if args.pyspy_threads:
        cmd.append("--threads")
    print("PYSPY_CMD", json.dumps(cmd), flush=True)
    log_file = pyspy_log.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    proc._codex_log_file = log_file  # type: ignore[attr-defined]
    time.sleep(args.pyspy_startup_s)
    if proc.poll() is not None:
        log_file.close()
        raise RuntimeError(f"py-spy exited early with code {proc.returncode}; see {pyspy_log}")
    return proc


def stop_pyspy(
    proc: subprocess.Popen[str] | None,
    timeout_s: float = 30.0,
    signal_if_running: bool = True,
) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        if not signal_if_running:
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=timeout_s)
        else:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=timeout_s)
    log_file = getattr(proc, "_codex_log_file", None)
    if log_file is not None:
        log_file.close()


def pyspy_stop_timeout(args: argparse.Namespace) -> float:
    if args.pyspy_duration_s > 0:
        return args.pyspy_duration_s + 30.0
    return 30.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start vLLM OpenAI server and drive a fixed concurrency load."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="profile-model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--total-requests", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--quantization", default="compressed-tensors")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--wait-timeout-s", type=float, default=900.0)
    parser.add_argument("--request-timeout-s", type=float, default=900.0)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--disable-eager", action="store_true")
    parser.add_argument("--emit-nvtx", action="store_true")
    parser.add_argument("--nvtx-message", default="vllm_profile_window")
    parser.add_argument("--pyspy-out")
    parser.add_argument("--pyspy-log")
    parser.add_argument("--pyspy-bin", default="/root/autodl-tmp/qwen35-compress-venv/bin/py-spy")
    parser.add_argument("--pyspy-rate", type=int, default=200)
    parser.add_argument("--pyspy-native", action="store_true")
    parser.add_argument("--pyspy-threads", action="store_true")
    parser.add_argument("--pyspy-startup-s", type=float, default=1.0)
    parser.add_argument("--pyspy-duration-s", type=float, default=0.0)
    args = parser.parse_args()

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.server_log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{args.host}:{args.port}"
    env = os.environ.copy()
    env.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    env.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf-cache/hub")
    env.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf-datasets")
    env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    env.setdefault("XDG_CACHE_HOME", "/root/autodl-tmp/cache")
    env.setdefault("TORCH_EXTENSIONS_DIR", "/root/autodl-tmp/torch_extensions")
    env.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
    )

    server_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--trust-remote-code",
        "--dtype",
        args.dtype,
        "--quantization",
        args.quantization,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        "1",
        "--max-num-seqs",
        str(args.concurrency),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--language-model-only",
        "--gdn-prefill-backend",
        "triton",
        "--disable-log-stats",
        "--disable-uvicorn-access-log",
        "--uvicorn-log-level",
        "warning",
    ]
    if not args.disable_eager:
        server_cmd.append("--enforce-eager")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_path.open("w", encoding="utf-8") as log_file:
        print("SERVER_CMD", json.dumps(server_cmd), flush=True)
        proc = subprocess.Popen(
            server_cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            asyncio.run(wait_ready(base_url, args.wait_timeout_s))
            print("SERVER_READY", flush=True)

            if args.warmup_requests:
                asyncio.run(
                    run_load(
                        base_url,
                        args.served_model_name,
                        concurrency=min(args.concurrency, args.warmup_requests),
                        total_requests=args.warmup_requests,
                        max_tokens=min(args.max_tokens, 8),
                        timeout_s=args.request_timeout_s,
                    )
                )
                print("WARMUP_DONE", flush=True)

            pyspy_proc = start_pyspy(args, proc.pid)
            try:
                with OptionalNvtxRange(args.emit_nvtx, args.nvtx_message):
                    profile_start = time.perf_counter()
                    completions = asyncio.run(
                        run_load(
                            base_url,
                            args.served_model_name,
                            concurrency=args.concurrency,
                            total_requests=args.total_requests,
                            max_tokens=args.max_tokens,
                            timeout_s=args.request_timeout_s,
                        )
                    )
                    profile_elapsed_s = time.perf_counter() - profile_start
            finally:
                stop_pyspy(
                    pyspy_proc,
                    timeout_s=pyspy_stop_timeout(args),
                    signal_if_running=args.pyspy_duration_s <= 0,
                )
            print("PROFILE_DONE", flush=True)
        finally:
            terminate_process(proc)

    completion_tokens = [
        item["completion_tokens"] for item in completions if item["completion_tokens"] is not None
    ]
    prompt_tokens = [
        item["prompt_tokens"] for item in completions if item["prompt_tokens"] is not None
    ]
    latencies = [item["latency_s"] for item in completions]
    total_completion_tokens = sum(completion_tokens)

    result = {
        "started_at": started_at,
        "model": args.model,
        "served_model_name": args.served_model_name,
        "concurrency": args.concurrency,
        "total_requests": args.total_requests,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "quantization": args.quantization,
        "enforce_eager": not args.disable_eager,
        "server_cmd": server_cmd,
        "profile_elapsed_s": profile_elapsed_s,
        "total_completion_tokens": total_completion_tokens,
        "completion_tokens_per_s": (
            total_completion_tokens / profile_elapsed_s if profile_elapsed_s else None
        ),
        "avg_latency_s": mean(latencies) if latencies else None,
        "min_latency_s": min(latencies) if latencies else None,
        "max_latency_s": max(latencies) if latencies else None,
        "avg_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
        "avg_completion_tokens": mean(completion_tokens) if completion_tokens else None,
        "responses": completions,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
