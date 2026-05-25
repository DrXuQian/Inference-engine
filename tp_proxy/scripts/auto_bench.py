#!/usr/bin/env python3
"""
Sweep input lengths and collect TTFT/TPOT using generate_bench.py.

Usage:
    python auto_bench.py \
        --model-dir /path/to/rank_0 \
        --input-lens 512,1024,2048,4096 \
        --output-len 256 \
        --output-json bench_results.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


def run_bench(model_dir: str, input_len: int, output_len: int,
              num_prompts: int, num_warmup: int, gpu_mem: float,
              tp: int, max_model_len: int | None) -> dict:
    """Run generate_bench.py for one input_len, return results dict."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_bench.py")
    tmp = tempfile.mktemp(suffix=".json")

    mml = max_model_len or (input_len + output_len + 64)

    cmd = [
        sys.executable, script,
        "--model", model_dir,
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--num-warmup", str(num_warmup),
        "--max-model-len", str(mml),
        "--gpu-mem", str(gpu_mem),
        "--tp", str(tp),
        "--output-json", tmp,
    ]

    env = os.environ.copy()
    env["TRITON_BACKENDS_IN_TREE"] = "1"

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"  ERROR: generate_bench failed for input_len={input_len}")
        print(f"  stderr: {result.stderr[-500:]}")
        return {"input_len": input_len, "error": result.stderr[-200:]}

    with open(tmp) as f:
        data = json.load(f)
    os.remove(tmp)
    return data


def main():
    ap = argparse.ArgumentParser(description="Sweep input lengths for TTFT/TPOT")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input-lens", required=True,
                    help="Comma-separated input lengths (e.g., 512,1024,2048)")
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--num-warmup", type=int, default=3)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--output-json", default="bench_results.json")
    args = ap.parse_args()

    input_lens = [int(x.strip()) for x in args.input_lens.split(",")]

    print(f"Model: {args.model_dir}")
    print(f"Input lengths: {input_lens}")
    print(f"Output length: {args.output_len}")
    print(f"Prompts per length: {args.num_prompts}")
    print()

    all_results = []
    for i, il in enumerate(input_lens):
        print(f"[{i+1}/{len(input_lens)}] input_len={il}...")
        data = run_bench(
            args.model_dir, il, args.output_len,
            args.num_prompts, args.num_warmup,
            args.gpu_mem, args.tp, args.max_model_len,
        )
        all_results.append(data)
        if "error" not in data:
            print(f"  TTFT={data['ttft_median_ms']:.2f}ms  TPOT={data['tpot_median_ms']:.3f}ms")
        print()

    # Summary table
    print(f"{'='*60}")
    print(f"{'input_len':>10}  {'TTFT (ms)':>10}  {'TPOT (ms)':>10}  {'total (ms)':>10}")
    print(f"{'-'*60}")
    for r in all_results:
        if "error" in r:
            print(f"{r['input_len']:>10}  {'ERROR':>10}")
        else:
            print(f"{r['input_len']:>10}  {r['ttft_median_ms']:>10.2f}  "
                  f"{r['tpot_median_ms']:>10.3f}  {r['total_median_ms']:>10.2f}")
    print(f"{'='*60}")

    # Save
    output = {
        "model_dir": args.model_dir,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "results": all_results,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
