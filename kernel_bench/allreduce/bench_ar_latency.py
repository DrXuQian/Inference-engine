#!/usr/bin/env python3
"""
Benchmark all-reduce latency at various message sizes.
Reproduces the table from pcie-oneshot-allreduce.md.

Uses torch.distributed (NCCL/PCCL backend) for baseline measurement.

Usage:
    # 2 GPUs
    torchrun --nproc_per_node=2 bench_ar_latency.py

    # 4 GPUs
    torchrun --nproc_per_node=4 bench_ar_latency.py

    # Custom sizes
    torchrun --nproc_per_node=2 bench_ar_latency.py --sizes 1024,4096,16384,65536

    # Save results
    torchrun --nproc_per_node=2 bench_ar_latency.py -o ar_latency.json
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def bench_allreduce(size_bytes, dtype=torch.bfloat16, warmup=50, iters=200):
    """Benchmark all-reduce latency for given message size."""
    elem_size = torch.finfo(dtype).bits // 8
    numel = size_bytes // elem_size
    if numel < 1:
        numel = 1

    buf = torch.randn(numel, dtype=dtype, device="cuda")

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(buf)
    torch.cuda.synchronize()

    # Measure
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dist.all_reduce(buf)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)  # us

    times.sort()
    # Remove top/bottom 10%
    trim = len(times) // 10
    trimmed = times[trim:-trim] if trim > 0 else times
    median = trimmed[len(trimmed) // 2]
    mean = sum(trimmed) / len(trimmed)
    p99 = times[int(len(times) * 0.99)]

    return {"median_us": round(median, 1), "mean_us": round(mean, 1), "p99_us": round(p99, 1)}


def fmt_size(b):
    if b >= 1024 * 1024:
        return f"{b // (1024*1024)} MB"
    if b >= 1024:
        return f"{b // 1024} KB"
    return f"{b} B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1024,4096,8192,16384,32768,65536,131072,262144,524288,1048576",
                    help="Comma-separated message sizes in bytes")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    if rank == 0:
        print(f"All-Reduce Latency Benchmark")
        print(f"  GPUs: {world}")
        print(f"  Backend: {dist.get_backend()}")
        print(f"  Warmup: {args.warmup}, Iters: {args.iters}")
        print()
        print(f"{'Size':>10} {'Median (µs)':>12} {'Mean (µs)':>12} {'P99 (µs)':>12}")
        print("-" * 50)

    results = []
    for size in sizes:
        r = bench_allreduce(size, warmup=args.warmup, iters=args.iters)
        results.append({"size_bytes": size, **r})
        if rank == 0:
            print(f"{fmt_size(size):>10} {r['median_us']:>12.1f} {r['mean_us']:>12.1f} {r['p99_us']:>12.1f}")

    if rank == 0 and args.output:
        output = {
            "world_size": world,
            "backend": dist.get_backend(),
            "results": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved: {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
