#!/usr/bin/env python3
"""
Automated per-layer latency measurement and compensation for pruned models.

Uses differential measurement: run vllm bench latency at two different
layer counts, compute per-layer time from the delta, then extrapolate
to the full model.

  per_layer = (latency_N - latency_M) / (N - M)
  full_model = latency_N + per_layer × (original - N)

Usage:
    python profile_compensate.py \\
        --rank-dir /path/to/rank_0 \\
        --layer-counts 10 5 \\
        --original-layers 40 \\
        --batch-size 4 --input-len 128 --output-len 64

    # Single layer count (uses total/N approximation)
    python profile_compensate.py \\
        --rank-dir /path/to/rank_0 \\
        --layer-counts 10 \\
        --original-layers 40
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def prune_to(rank_dir: str, num_layers: int, tmp_root: str) -> str:
    """Create a pruned model directory with num_layers layers. Returns path."""
    output_dir = os.path.join(tmp_root, f"pruned_{num_layers}L")
    if os.path.exists(output_dir):
        return output_dir

    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "prune_layers.py"),
        "--rank-dir", rank_dir,
        "--num-layers", str(num_layers),
        "--output-dir", output_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"Pruning to {num_layers} layers failed:")
        print(result.stderr)
        sys.exit(1)
    print(f"  Pruned to {num_layers} layers -> {output_dir}")
    return output_dir


def measure_latency(model_dir: str, batch_size: int, input_len: int,
                    output_len: int, max_model_len: int,
                    gpu_mem: float, num_iters: int) -> float:
    """Run vllm bench latency and return avg latency in seconds."""
    env = os.environ.copy()
    env["TRITON_BACKENDS_IN_TREE"] = "1"

    cmd = [
        "vllm", "bench", "latency",
        "--model", model_dir,
        "--batch-size", str(batch_size),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--num-iters", str(num_iters),
        "--num-iters-warmup", "2",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--enforce-eager",
        "--trust-remote-code",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True,
                            env=env, timeout=600)

    # Parse "Avg latency: X.XXX seconds"
    for line in (result.stdout + result.stderr).split("\n"):
        if "Avg latency:" in line:
            m = re.search(r"Avg latency:\s*([\d.]+)\s*seconds", line)
            if m:
                return float(m.group(1))

    print(f"ERROR: could not parse latency from vllm bench output")
    print("STDOUT:", result.stdout[-500:])
    print("STDERR:", result.stderr[-500:])
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Differential per-layer latency measurement + compensation")
    ap.add_argument("--rank-dir", required=True,
                    help="TP-split rank directory (full layers)")
    ap.add_argument("--layer-counts", type=int, nargs="+", required=True,
                    help="Layer counts to measure (e.g., 10 5 or just 10)")
    ap.add_argument("--original-layers", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--num-iters", type=int, default=5)
    ap.add_argument("--tmp-dir", default=None,
                    help="Temp dir for pruned models (default: auto)")
    args = ap.parse_args()

    layer_counts = sorted(args.layer_counts)
    if len(layer_counts) < 1:
        print("Need at least one --layer-counts value")
        sys.exit(1)

    tmp_dir = args.tmp_dir or tempfile.mkdtemp(prefix="layer_comp_")
    print(f"Temp dir: {tmp_dir}")
    print(f"Bench config: batch={args.batch_size}, in={args.input_len}, "
          f"out={args.output_len}, iters={args.num_iters}")
    print()

    # Step 1: Prune and measure each layer count
    results = {}
    for n in layer_counts:
        print(f"--- {n} layers ---")
        pruned_dir = prune_to(args.rank_dir, n, tmp_dir)
        print(f"  Measuring latency...")
        lat = measure_latency(pruned_dir, args.batch_size, args.input_len,
                              args.output_len, args.max_model_len,
                              args.gpu_mem, args.num_iters)
        results[n] = lat
        print(f"  Avg latency: {lat*1000:.2f} ms")
        print()

    # Step 2: Compute per-layer time
    print(f"{'='*60}")
    if len(layer_counts) >= 2:
        # Differential method: use two measurements to cancel non-layer overhead
        n_hi = layer_counts[-1]
        n_lo = layer_counts[0]
        lat_hi = results[n_hi]
        lat_lo = results[n_lo]
        per_layer_s = (lat_hi - lat_lo) / (n_hi - n_lo)
        # Non-layer overhead = lat_lo - n_lo × per_layer
        overhead_s = lat_lo - n_lo * per_layer_s
        method = "differential"
        print(f"Method: differential ({n_lo}L vs {n_hi}L)")
        print(f"  {n_lo}L latency: {lat_lo*1000:.2f} ms")
        print(f"  {n_hi}L latency: {lat_hi*1000:.2f} ms")
        print(f"  Per-layer:       {per_layer_s*1000:.2f} ms")
        print(f"  Non-layer overhead: {overhead_s*1000:.2f} ms")
    else:
        # Single measurement: assume layer compute dominates
        n = layer_counts[0]
        lat = results[n]
        per_layer_s = lat / n
        overhead_s = 0
        method = "single (total/N)"
        print(f"Method: single measurement ({n}L)")
        print(f"  {n}L latency: {lat*1000:.2f} ms")
        print(f"  Per-layer (approx): {per_layer_s*1000:.2f} ms")

    # Step 3: Extrapolate to original layer count
    full_model_s = overhead_s + args.original_layers * per_layer_s
    print()
    print(f"Extrapolated {args.original_layers}-layer latency:")
    print(f"  = {overhead_s*1000:.2f} + {args.original_layers} × {per_layer_s*1000:.2f}")
    print(f"  = {full_model_s*1000:.2f} ms")
    print(f"{'='*60}")

    # Output summary JSON
    summary = {
        "method": method,
        "layer_counts": layer_counts,
        "latencies_ms": {str(n): results[n]*1000 for n in layer_counts},
        "per_layer_ms": per_layer_s * 1000,
        "overhead_ms": overhead_s * 1000,
        "original_layers": args.original_layers,
        "extrapolated_ms": full_model_s * 1000,
        "bench_config": {
            "batch_size": args.batch_size,
            "input_len": args.input_len,
            "output_len": args.output_len,
        },
    }
    summary_path = os.path.join(tmp_dir, "compensation_result.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
