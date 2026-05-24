#!/usr/bin/env python3
"""
Benchmark NCCL all-reduce and all-gather latency for TP communication
compensation. Tests multiple tensor sizes matching actual model usage.

Usage:
    python nccl_bench.py
    python nccl_bench.py --hidden-size 2048 --num-layers 40

Output: per-size latency table + per-decode-step total communication time.
Use this to compensate single-card TP-split proxy results.
"""

import argparse
import os
import multiprocessing as mp

import torch
import torch.distributed as dist


def worker(rank, world_size, hidden_size, num_layers, results):
    os.environ.update({
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29503",
        "WORLD_SIZE": str(world_size),
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
    })
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = f"cuda:{rank}"
    dtype = torch.bfloat16

    def bench(fn, warmup=100, iters=500):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000)  # us
        times.sort()
        return {
            "median": times[len(times) // 2],
            "p99": times[int(len(times) * 0.99)],
            "min": times[0],
        }

    # Test sizes: decode (small) and prefill (large)
    sizes = [
        ("4KB",    4 * 1024),       # batch=1, hidden=2048, bf16
        ("16KB",   16 * 1024),      # batch=4
        ("64KB",   64 * 1024),      # batch=16
        ("256KB",  256 * 1024),     # batch=64
        ("1MB",    1 * 1024**2),    # prefill chunk
        ("8MB",    8 * 1024**2),    # full prefill (2048 × 2048 × 2)
    ]

    if rank == 0:
        print(f"{'size':>8} {'AR median':>10} {'AR p99':>10} "
              f"{'AG median':>10} {'AG p99':>10}  (us)")
        print("-" * 55)

    for label, nbytes in sizes:
        numel = nbytes // 2  # bf16
        t = torch.randn(numel, dtype=dtype, device=device)
        ar = bench(lambda: dist.all_reduce(t))

        src = torch.randn(numel, dtype=dtype, device=device)
        dst = torch.empty(numel * world_size, dtype=dtype, device=device)
        ag = bench(lambda: dist.all_gather_into_tensor(dst, src))

        if rank == 0:
            results[f"AR_{label}"] = ar["median"]
            results[f"AG_{label}"] = ag["median"]
            print(f"{label:>8} {ar['median']:10.1f} {ar['p99']:10.1f} "
                  f"{ag['median']:10.1f} {ag['p99']:10.1f}")

    if rank == 0:
        # Compute decode step total
        ar_4k = results["AR_4KB"]
        ag_4k = results["AG_4KB"]
        n_ar_per_layer = 2  # after attn + after MoE
        n_ag_per_step = 2   # lm_head all-gather
        total_ar = num_layers * n_ar_per_layer * ar_4k / 1000  # ms
        total_ag = n_ag_per_step * ag_4k / 1000
        total = total_ar + total_ag

        print(f"\n=== Per decode step (batch=1) ===")
        print(f"  AR: {num_layers}L × {n_ar_per_layer} × {ar_4k:.1f}us "
              f"= {total_ar:.3f}ms")
        print(f"  AG: {n_ag_per_step} × {ag_4k:.1f}us = {total_ag:.3f}ms")
        print(f"  Total standalone: {total:.3f}ms")
        print(f"\n  NOTE: In CUDA Graph, actual overhead is lower due to")
        print(f"  pipelining. Apply correction factor from calibration.")

        results["total_standalone_ms"] = total

    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--num-layers", type=int, default=40)
    ap.add_argument("--world-size", type=int, default=2)
    args = ap.parse_args()

    results = mp.Manager().dict()
    procs = [mp.Process(target=worker,
                        args=(r, args.world_size, args.hidden_size,
                              args.num_layers, results))
             for r in range(args.world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
