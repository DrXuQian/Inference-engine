#!/usr/bin/env python3
"""
Controlled generation benchmark.

Two modes:
  serve (default): starts vllm serve → sends HTTP streaming requests → measures TTFT/TPOT
  offline:         uses vllm.LLM directly (faster but may not work on all platforms)

Usage:
    # Default (serve mode, most compatible)
    python generate_bench.py --model /path/to/model --input-len 512 --output-len 256

    # Offline mode (if vllm.LLM works on your platform)
    python generate_bench.py --model /path/to/model --input-len 512 --output-len 256 --mode offline

    # Under nsys/asys profiling (offline mode recommended)
    nsys profile ... python generate_bench.py --model /path/to/model --mode offline ...
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np


# ---------------------------------------------------------------------------
# NVTX helpers
# ---------------------------------------------------------------------------

def nvtx_range(name):
    import contextlib
    try:
        import torch
        @contextlib.contextmanager
        def _range():
            torch.cuda.nvtx.range_push(name)
            try: yield
            finally: torch.cuda.nvtx.range_pop()
        return _range()
    except Exception:
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Serve mode: vllm serve + HTTP streaming
# ---------------------------------------------------------------------------

def wait_for_server(port):
    import urllib.request
    while True:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)


def http_generate(port, prompt_ids, max_tokens, model_name):
    """Send streaming completion request, measure TTFT and per-token times."""
    import urllib.request
    payload = json.dumps({
        "model": model_name,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=payload, headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: "): continue
            if line[6:] == "[DONE]": break
            try:
                chunk = json.loads(line[6:])
                if chunk.get("choices") and chunk["choices"][0].get("text"):
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
                    n_tokens += 1
            except json.JSONDecodeError:
                continue

    total_ms = (time.perf_counter() - t0) * 1000
    ttft = ttft or total_ms
    tpot = (total_ms - ttft) / max(n_tokens - 1, 1) if n_tokens > 1 else 0
    return {"ttft_ms": ttft, "tpot_ms": tpot, "total_ms": total_ms, "n_output": n_tokens}


def run_serve_mode(args):
    port = 8199
    mml = args.max_model_len or (args.input_len + args.output_len + 64)
    env = os.environ.copy()
    env["TRITON_BACKENDS_IN_TREE"] = "1"

    server = subprocess.Popen(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", args.model, "--host", "127.0.0.1", "--port", str(port),
         "--tensor-parallel-size", str(args.tp),
         "--max-model-len", str(mml),
         "--trust-remote-code",
         "--no-enable-prefix-caching",
         "--gpu-memory-utilization", str(args.gpu_mem)],
        env=env)

    def cleanup():
        server.terminate()
        try: server.wait(timeout=10)
        except: server.kill()

    try:
        print(f"Starting vllm serve (port {port})...")
        if not wait_for_server(port):
            print("ERROR: server failed to start.", file=sys.stderr)
            try:
                with open("/tmp/generate_bench_server.log") as f:
                    log = f.read()
                    print(log[-2000:], file=sys.stderr)
            except FileNotFoundError:
                # Server process may have died before creating log
                ret = server.poll()
                print(f"Server process exited with code: {ret}", file=sys.stderr)
            cleanup(); sys.exit(1)
        print("Server ready")

        prompts = [np.random.randint(0, 10000, size=args.input_len).tolist()
                   for _ in range(args.num_prompts + args.num_warmup)]

        print(f"Warmup ({args.num_warmup})...")
        for i in range(args.num_warmup):
            http_generate(port, prompts[i], args.output_len, args.model)

        print(f"Benchmarking ({args.num_prompts} prompts)...")
        results = []
        for i in range(args.num_prompts):
            r = http_generate(port, prompts[args.num_warmup + i],
                              args.output_len, args.model)
            results.append(r)

        mid = len(results) // 2
        ttfts = sorted(r["ttft_ms"] for r in results)
        tpots = sorted(r["tpot_ms"] for r in results)
        totals = sorted(r["total_ms"] for r in results)
        n_outs = sorted(r["n_output"] for r in results)

        return {
            "ttft_median_ms": round(ttfts[mid], 3),
            "tpot_median_ms": round(tpots[mid], 3),
            "total_median_ms": round(totals[mid], 3),
            "output_tokens": n_outs[mid],
        }
    finally:
        cleanup()


# ---------------------------------------------------------------------------
# Offline mode: vllm.LLM directly
# ---------------------------------------------------------------------------

def run_offline_mode(args):
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    from vllm import LLM, SamplingParams

    mml = args.max_model_len or (args.input_len + args.output_len + 64)
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="auto",
              max_model_len=mml, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem)

    prompts = [{"prompt_token_ids": np.random.randint(0, 10000, size=args.input_len).tolist()}
               for _ in range(args.num_prompts + args.num_warmup + 5)]
    sp = SamplingParams(max_tokens=args.output_len, temperature=0, ignore_eos=True)

    with nvtx_range("warmup"):
        print(f"Warmup ({args.num_warmup})...")
        for i in range(args.num_warmup):
            llm.generate([prompts[i]], sampling_params=sp)

    with nvtx_range("bench"):
        print(f"Benchmarking ({args.num_prompts} prompts)...")
        results = []
        for i in range(args.num_prompts):
            with nvtx_range(f"request_{i}"):
                t0 = time.perf_counter()
                outputs = llm.generate([prompts[args.num_warmup + i]], sampling_params=sp)
                total_ms = (time.perf_counter() - t0) * 1000
            n_output = len(outputs[0].outputs[0].token_ids)
            results.append({"total_ms": total_ms, "n_output": n_output})

    with nvtx_range("ttft_measure"):
        print("Measuring TTFT...")
        sp_short = SamplingParams(max_tokens=1, temperature=0)
        ttft_times = []
        for i in range(min(5, args.num_prompts)):
            t0 = time.perf_counter()
            llm.generate([prompts[args.num_warmup + args.num_prompts + i]], sampling_params=sp_short)
            ttft_times.append((time.perf_counter() - t0) * 1000)

    ttft_times.sort()
    totals = sorted(r["total_ms"] for r in results)
    n_outs = sorted(r["n_output"] for r in results)
    mid = len(results) // 2
    ttft_median = ttft_times[len(ttft_times) // 2]
    tpot = (totals[mid] - ttft_median) / max(n_outs[mid] - 1, 1)

    return {
        "ttft_median_ms": round(ttft_median, 3),
        "tpot_median_ms": round(tpot, 3),
        "total_median_ms": round(totals[mid], 3),
        "output_tokens": n_outs[mid],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--num-warmup", type=int, default=3)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--mode", choices=["serve", "offline"], default="serve",
                    help="serve: HTTP (compatible); offline: vllm.LLM (fast, needs platform support)")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    print(f"Model: {args.model}")
    print(f"Config: input={args.input_len}, output={args.output_len}, "
          f"prompts={args.num_prompts}, mode={args.mode}")

    if args.mode == "offline":
        result = run_offline_mode(args)
    else:
        result = run_serve_mode(args)

    result.update({"input_len": args.input_len, "output_len": args.output_len,
                   "num_prompts": args.num_prompts, "mode": args.mode})

    print(f"\n{'='*50}")
    print(f"Results (input={args.input_len}, output={args.output_len}):")
    print(f"  Median TTFT:  {result['ttft_median_ms']:.2f} ms")
    print(f"  Median TPOT:  {result['tpot_median_ms']:.3f} ms")
    print(f"  Median total: {result['total_median_ms']:.2f} ms")
    print(f"  Output tokens: {result['output_tokens']}")
    print(f"{'='*50}")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
