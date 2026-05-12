#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import time
from pathlib import Path
from statistics import mean


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


def top_profile_rows(profile_path: Path, limit: int = 80) -> list[dict[str, object]]:
    stats = pstats.Stats(str(profile_path))
    stats.sort_stats("cumtime")
    rows: list[dict[str, object]] = []
    for func, stat in list(stats.stats.items()):
        cc, nc, tt, ct, _callers = stat
        file_name, line_no, func_name = func
        rows.append(
            {
                "primitive_calls": cc,
                "total_calls": nc,
                "self_s": round(tt, 6),
                "cumulative_s": round(ct, 6),
                "function": f"{file_name}:{line_no}:{func_name}",
            }
        )
    rows.sort(key=lambda item: item["cumulative_s"], reverse=True)
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a vLLM offline autoregressive generation window."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--total-requests", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--quantization", default="compressed-tensors")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--cprofile-out")
    parser.add_argument("--emit-nvtx", action="store_true")
    parser.add_argument("--nvtx-message", default="vllm_profile_window")
    parser.add_argument("--disable-eager", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf-cache/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/hf-datasets")
    os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/root/autodl-tmp/cache")
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/autodl-tmp/torch_extensions")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.7",
    )

    from vllm import LLM, SamplingParams

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cprofile_path = Path(args.cprofile_out) if args.cprofile_out else None
    if cprofile_path is not None:
        cprofile_path.parent.mkdir(parents=True, exist_ok=True)

    print("LOAD_START", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        quantization=args.quantization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        max_num_seqs=args.concurrency,
        max_num_batched_tokens=args.max_num_batched_tokens,
        language_model_only=True,
        gdn_prefill_backend="triton",
        enforce_eager=not args.disable_eager,
        disable_log_stats=True,
    )
    print("LOAD_DONE", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)

    sampling_params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    if args.warmup_requests:
        warmup_prompts = [build_prompt(i) for i in range(args.warmup_requests)]
        llm.generate(
            warmup_prompts,
            SamplingParams(max_tokens=min(args.max_tokens, 8), temperature=0.0),
            use_tqdm=False,
        )
        print("WARMUP_DONE", flush=True)

    prompts = [build_prompt(i) for i in range(args.total_requests)]
    profiler = cProfile.Profile() if cprofile_path is not None else None

    with OptionalNvtxRange(args.emit_nvtx, args.nvtx_message):
        start = time.perf_counter()
        if profiler is not None:
            profiler.enable()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        if profiler is not None:
            profiler.disable()
        profile_elapsed_s = time.perf_counter() - start
    print("PROFILE_DONE", flush=True)

    if profiler is not None and cprofile_path is not None:
        profiler.dump_stats(str(cprofile_path))
        print(f"CPROFILE_WRITTEN {cprofile_path}", flush=True)

    completions = []
    for idx, output in enumerate(outputs):
        completion = output.outputs[0]
        completions.append(
            {
                "request_id": idx,
                "prompt_token_count": len(output.prompt_token_ids or []),
                "completion_token_count": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            }
        )
    completion_tokens = [item["completion_token_count"] for item in completions]
    prompt_tokens = [item["prompt_token_count"] for item in completions]
    total_completion_tokens = sum(completion_tokens)
    result = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "concurrency": args.concurrency,
        "total_requests": args.total_requests,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "quantization": args.quantization,
        "enforce_eager": not args.disable_eager,
        "vllm_enable_v1_multiprocessing": os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
        "profile_elapsed_s": profile_elapsed_s,
        "total_completion_tokens": total_completion_tokens,
        "completion_tokens_per_s": (
            total_completion_tokens / profile_elapsed_s if profile_elapsed_s else None
        ),
        "avg_prompt_tokens": mean(prompt_tokens) if prompt_tokens else None,
        "avg_completion_tokens": mean(completion_tokens) if completion_tokens else None,
        "cprofile_out": str(cprofile_path) if cprofile_path else None,
        "cprofile_top_cumulative": (
            top_profile_rows(cprofile_path) if cprofile_path is not None else []
        ),
        "responses": completions,
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
