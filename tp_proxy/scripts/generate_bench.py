#!/usr/bin/env python3
"""
Controlled generation benchmark for nsys/asys profiling.
Uses vllm.LLM directly (no HTTP server) with exact input/output lengths.
NVTX markers separate warmup from bench for clean profiling.

Usage:
    python generate_bench.py --model /path/to/model --input-len 512 --output-len 256

    # Under nsys (NVIDIA)
    nsys profile -t cuda,nvtx --cuda-trace-scope=system-wide --cuda-graph-trace=node \
        -o profile python generate_bench.py --model /path/to/model ...

    # Under asys (PPU)
    asys profile -o profile -f true -t hggc,acdnn,acblas \
        python generate_bench.py --model /path/to/model ...
"""

import argparse
import json
import time

import numpy as np


def nvtx_range(name):
    """Context manager for NVTX range (no-op if unavailable)."""
    import contextlib
    try:
        import torch
        @contextlib.contextmanager
        def _range():
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        return _range()
    except Exception:
        return contextlib.nullcontext()


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
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    if args.max_model_len is None:
        args.max_model_len = args.input_len + args.output_len + 64

    import os
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")

    from vllm import LLM, SamplingParams

    print(f"Loading model: {args.model}")
    print(f"Config: input={args.input_len}, output={args.output_len}, "
          f"prompts={args.num_prompts}, warmup={args.num_warmup}, tp={args.tp}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_mem,
    )

    prompts = [{"prompt_token_ids": np.random.randint(0, 10000, size=args.input_len).tolist()}
               for _ in range(args.num_prompts + args.num_warmup + 5)]
    sp = SamplingParams(max_tokens=args.output_len, temperature=0, ignore_eos=True)

    # ---- Warmup (NVTX: "warmup") ----
    with nvtx_range("warmup"):
        print(f"Warmup ({args.num_warmup} prompts)...")
        for i in range(args.num_warmup):
            llm.generate([prompts[i]], sampling_params=sp)

    # ---- Benchmark (NVTX: "bench") ----
    with nvtx_range("bench"):
        print(f"Benchmarking ({args.num_prompts} prompts)...")
        results = []
        for i in range(args.num_prompts):
            p = prompts[args.num_warmup + i]
            with nvtx_range(f"request_{i}"):
                t0 = time.perf_counter()
                outputs = llm.generate([p], sampling_params=sp)
                t1 = time.perf_counter()
            n_output = len(outputs[0].outputs[0].token_ids)
            results.append({"total_ms": (t1 - t0) * 1000, "n_output": n_output})

    # ---- TTFT measurement (NVTX: "ttft") ----
    with nvtx_range("ttft_measure"):
        print("Measuring TTFT (1-token generation)...")
        sp_short = SamplingParams(max_tokens=1, temperature=0)
        ttft_times = []
        for i in range(min(5, args.num_prompts)):
            p = prompts[args.num_warmup + args.num_prompts + i]
            t0 = time.perf_counter()
            llm.generate([p], sampling_params=sp_short)
            t1 = time.perf_counter()
            ttft_times.append((t1 - t0) * 1000)

    # ---- Compute results ----
    totals_sorted = sorted(r["total_ms"] for r in results)
    n_sorted = sorted(r["n_output"] for r in results)
    median_total = totals_sorted[len(totals_sorted) // 2]
    median_n = n_sorted[len(n_sorted) // 2]

    ttft_times.sort()
    ttft_median = ttft_times[len(ttft_times) // 2]
    tpot = (median_total - ttft_median) / max(median_n - 1, 1)

    print(f"\n{'='*50}")
    print(f"Results (input={args.input_len}, output={args.output_len}):")
    print(f"  Median TTFT:  {ttft_median:.2f} ms")
    print(f"  Median TPOT:  {tpot:.3f} ms")
    print(f"  Median total: {median_total:.2f} ms")
    print(f"  Output tokens: {median_n}")
    print(f"{'='*50}")

    if args.output_json:
        out = {
            "input_len": args.input_len,
            "output_len": args.output_len,
            "num_prompts": args.num_prompts,
            "ttft_median_ms": round(ttft_median, 3),
            "tpot_median_ms": round(tpot, 3),
            "total_median_ms": round(median_total, 3),
            "output_tokens": median_n,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
