#!/usr/bin/env python3
"""
Measure the latency overhead of keeping lm_head at full vocab_size (replicated)
vs. half vocab_size (TP=2 split), and compute a compensation value to subtract
from raw benchmark results.

The lm_head projection is: [tokens, hidden_size] @ [hidden_size, vocab_size]
In TP=2 each rank would only compute half the vocab, so the delta is the
extra time spent on the redundant half.

Usage:
    python lm_head_compensation.py
    python lm_head_compensation.py --batch-size 8 --input-len 256 --output-len 128
    python lm_head_compensation.py --hidden-size 2048 --vocab-size 248320
"""

import argparse
import torch
import time


def bench_mm(M: int, K: int, N: int, dtype: torch.dtype,
             warmup: int = 50, iters: int = 200) -> float:
    """Benchmark [M, K] @ [K, N] matmul, return median latency in ms."""
    device = "cuda"
    A = torch.randn(M, K, dtype=dtype, device=device)
    B = torch.randn(K, N, dtype=dtype, device=device)

    # Warmup
    for _ in range(warmup):
        torch.mm(A, B)
    torch.cuda.synchronize()

    # Benchmark with CUDA events
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    times.sort()
    return times[len(times) // 2]  # median


def main():
    ap = argparse.ArgumentParser(
        description="Compute lm_head latency compensation for TP=2 split models")
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--vocab-size", type=int, default=248320,
                    help="Full (unsplit) vocab size")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Number of concurrent sequences (decode phase)")
    ap.add_argument("--input-len", type=int, default=128,
                    help="Prompt length per request (prefill phase)")
    ap.add_argument("--output-len", type=int, default=64,
                    help="Generated tokens per request")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = ap.parse_args()

    H = args.hidden_size
    V_full = args.vocab_size
    V_half = V_full // 2
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print(f"Config: hidden={H}, vocab_full={V_full}, vocab_half={V_half}, "
          f"dtype={args.dtype}")
    print(f"Bench:  batch_size={args.batch_size}, input_len={args.input_len}, "
          f"output_len={args.output_len}")
    print()

    # --- Prefill phase: [input_len, H] @ [H, V] ---
    # One prefill per request. In a batch, each request prefills independently
    # (or chunked), but the lm_head is applied once at the end of prefill.
    M_prefill = args.input_len
    t_prefill_full = bench_mm(M_prefill, H, V_full, dtype)
    t_prefill_half = bench_mm(M_prefill, H, V_half, dtype)
    delta_prefill = t_prefill_full - t_prefill_half

    print(f"=== Prefill lm_head: [{M_prefill}, {H}] @ [{H}, V] ===")
    print(f"  Full vocab ({V_full}):  {t_prefill_full:.3f} ms")
    print(f"  Half vocab ({V_half}):  {t_prefill_half:.3f} ms")
    print(f"  Delta per request:      {delta_prefill:.3f} ms")
    print()

    # --- Decode phase: [batch_size, H] @ [H, V] ---
    # Each decode step processes all active sequences together.
    # There are output_len decode steps total.
    M_decode = args.batch_size
    t_decode_full = bench_mm(M_decode, H, V_full, dtype)
    t_decode_half = bench_mm(M_decode, H, V_half, dtype)
    delta_decode = t_decode_full - t_decode_half

    print(f"=== Decode lm_head: [{M_decode}, {H}] @ [{H}, V] ===")
    print(f"  Full vocab ({V_full}):  {t_decode_full:.3f} ms")
    print(f"  Half vocab ({V_half}):  {t_decode_half:.3f} ms")
    print(f"  Delta per step:         {delta_decode:.3f} ms")
    print()

    # --- Total compensation ---
    # For a batch of batch_size requests:
    #   prefill: batch_size × delta_prefill  (each request prefills once)
    #   decode:  output_len × delta_decode   (output_len steps, each for whole batch)
    total_prefill_comp = args.batch_size * delta_prefill
    total_decode_comp = args.output_len * delta_decode
    total_comp = total_prefill_comp + total_decode_comp

    print(f"{'='*55}")
    print(f"Total compensation for {args.batch_size} reqs × "
          f"(in={args.input_len}, out={args.output_len}):")
    print(f"  Prefill: {args.batch_size} × {delta_prefill:.3f} = "
          f"{total_prefill_comp:.3f} ms")
    print(f"  Decode:  {args.output_len} × {delta_decode:.3f} = "
          f"{total_decode_comp:.3f} ms")
    print(f"  Total:   {total_comp:.3f} ms")
    print()
    print(f"Adjusted latency = raw_latency - {total_comp:.3f} ms")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
