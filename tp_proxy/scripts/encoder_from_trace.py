#!/usr/bin/env python3
"""
Extract per-encoder-block kernel time from nsys/asys sqlite trace.

Method:
  1. Find serving phase (after largest idle gap)
  2. Identify and subtract non-encoder kernels:
     - lm_head: by name (--lm-head-kernel) or by duration threshold (auto)
     - sampling: by name (--sampling-kernels)
  3. encoder_kernel_time = total - lm_head - sampling
  4. per_layer = encoder_kernel_time / (forward_passes × num_layers)

Usage:
    # NVIDIA (auto-detect lm_head by duration)
    python encoder_from_trace.py --sqlite trace.sqlite --num-layers 10 --output-len 64

    # PPU (explicit kernel names)
    python encoder_from_trace.py --sqlite trace.sqlite --num-layers 10 --output-len 64 \
        --lm-head-kernel gemvt_op --sampling-kernels ArgMaxOps
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict


def detect_kernel_table(cursor) -> str:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for r in cursor.fetchall():
        if "KERNEL" in r[0] and "ACTIVITY" in r[0]:
            return r[0]
    return ""


def find_serving_start(events):
    if not events:
        return 0
    t_min = events[0][0]
    bins = defaultdict(int)
    for e in events:
        bins[(e[0] - t_min) // 1_000_000_000] += 1
    max_bin = max(bins.keys()) if bins else 0
    best_start = best_len = gap_start = gap_len = 0
    in_gap = False
    for b in range(max_bin + 1):
        if bins[b] == 0:
            if not in_gap: gap_start = b; in_gap = True; gap_len = 1
            else: gap_len += 1
        else:
            if in_gap and gap_len > best_len:
                best_start = gap_start; best_len = gap_len
            in_gap = False
    return t_min + (best_start + best_len) * 1_000_000_000


def main():
    ap = argparse.ArgumentParser(description="Per-encoder-block time from trace")
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--lm-head-kernel", default=None,
                    help="LM head kernel name (e.g., gemvt_op). "
                         "If omitted, auto-detect by gemv duration threshold.")
    ap.add_argument("--sampling-kernels", default=None,
                    help="Comma-separated sampling kernel names (e.g., ArgMaxOps). "
                         "If omitted, try auto-detect.")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    N = args.num_layers
    sampling_names = set()
    if args.sampling_kernels:
        sampling_names = set(k.strip() for k in args.sampling_kernels.split(","))

    # Load events
    conn = sqlite3.connect(args.sqlite)
    cursor = conn.cursor()
    kt = detect_kernel_table(cursor)
    if not kt:
        print("ERROR: no kernel table"); sys.exit(1)
    print(f"Kernel table: {kt}")

    cursor.execute(f"""
        SELECT k.start, k."end" - k.start AS dur, s.value AS name
        FROM {kt} k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """)
    all_events = cursor.fetchall()
    conn.close()
    print(f"Total events: {len(all_events)}")

    # Filter to serving
    serve_start = find_serving_start(all_events)
    events = [(s, d, n) for s, d, n in all_events if s >= serve_start]
    print(f"Serving events: {len(events)}")

    # Aggregate
    name_durations = defaultdict(list)
    for s, d, n in events:
        name_durations[n].append(d)
    total_ns = sum(d for _, d, _ in events)

    # --- Identify lm_head ---
    lm_head_ns = 0
    lm_head_count = 0

    if args.lm_head_kernel:
        # PPU path: filter by name, then separate lm_head calls from encoder calls
        # of the same kernel using duration threshold
        all_durs = sorted(name_durations.get(args.lm_head_kernel, []))
        if all_durs:
            median = all_durs[len(all_durs) // 2]
            threshold = median * 10
            large = [d for d in all_durs if d > threshold]
            lm_head_ns = sum(large)
            lm_head_count = len(large)
            print(f"LM head kernel: {args.lm_head_kernel}")
            print(f"  Total calls: {len(all_durs)}, lm_head calls (dur>{threshold/1e3:.0f}us): {lm_head_count}")
            print(f"  LM head mean: {lm_head_ns/max(lm_head_count,1)/1e3:.0f}us")
    else:
        # NVIDIA path: auto-detect gemv kernel, use duration threshold
        gemv_candidates = []
        for name, durs in name_durations.items():
            nl = name.lower()
            if any(kw in nl for kw in ["gemv", "gemm", "matmul"]):
                gemv_candidates.append((name, sum(durs), len(durs)))
        if gemv_candidates:
            gemv_candidates.sort(key=lambda x: -x[1])
            gemv_name = gemv_candidates[0][0]
            all_durs = sorted(name_durations[gemv_name])
            median = all_durs[len(all_durs) // 2]
            threshold = median * 10
            large = [d for d in all_durs if d > threshold]
            lm_head_ns = sum(large)
            lm_head_count = len(large)
            print(f"LM head kernel (auto): {gemv_name[:60]}")
            print(f"  Total calls: {len(all_durs)}, lm_head calls (dur>{threshold/1e3:.0f}us): {lm_head_count}")
            print(f"  LM head mean: {lm_head_ns/max(lm_head_count,1)/1e3:.0f}us")

    # --- Identify sampling ---
    sampling_ns = 0
    sampling_count = 0

    if sampling_names:
        for name in sampling_names:
            if name in name_durations:
                durs = name_durations[name]
                sampling_ns += sum(durs)
                sampling_count += len(durs)
                print(f"Sampling kernel: {name}, calls: {len(durs)}, "
                      f"total: {sum(durs)/1e6:.2f}ms")
    else:
        # Auto-detect: look for topk/argmax/sample
        for name, durs in name_durations.items():
            nl = name.lower()
            if any(kw in nl for kw in ["topk_topp", "argmax", "_sample_"]):
                if name not in (args.lm_head_kernel or ""):
                    sampling_ns += sum(durs)
                    sampling_count += len(durs)
                    print(f"Sampling kernel (auto): {name[:60]}, calls: {len(durs)}")

    # --- Compute encoder ---
    encoder_ns = total_ns - lm_head_ns - sampling_ns

    # Estimate forward passes from lm_head count or sampling count
    n_fwd = lm_head_count or sampling_count
    if n_fwd == 0:
        # Fallback: estimate from kernel counts
        print("WARNING: could not detect forward pass count from lm_head/sampling")
        n_fwd = 1

    n_requests = max(round(n_fwd / args.output_len), 1)
    n_decode = n_requests * args.output_len
    n_prefill = n_requests

    # Separate prefill vs decode encoder time
    # Prefill forward passes are heavier (process input_len tokens)
    # Estimate: prefill contributes disproportionately to encoder_ns
    # Simple split: attribute (n_prefill × weight) to prefill, rest to decode
    # Weight ≈ ratio of prefill to decode kernel duration for same operation
    # Heuristic: prefill is ~5x heavier per forward pass than decode
    prefill_weight = 5
    total_weighted = n_decode + n_prefill * prefill_weight
    decode_encoder_ns = encoder_ns * n_decode / total_weighted
    prefill_encoder_ns = encoder_ns * n_prefill * prefill_weight / total_weighted

    decode_per_step = decode_encoder_ns / max(n_decode, 1)
    decode_per_layer = decode_per_step / N
    prefill_per_layer = prefill_encoder_ns / max(n_prefill, 1) / N

    print(f"\n{'='*60}")
    print(f"Encoder analysis ({N} layers)")
    print(f"{'='*60}")
    print(f"  Total kernel time:    {total_ns/1e6:.2f} ms")
    print(f"  Encoder total:        {encoder_ns/1e6:.2f} ms ({encoder_ns/total_ns*100:.1f}%)")
    print(f"  LM head total:        {lm_head_ns/1e6:.2f} ms ({lm_head_count} calls)")
    print(f"  Sampling total:       {sampling_ns/1e6:.2f} ms ({sampling_count} calls)")
    print(f"  Forward passes:       {n_fwd} ({n_requests} req × {args.output_len} out + {n_prefill} prefill)")
    print(f"")
    print(f"  DECODE:")
    print(f"    Encoder total:      {decode_encoder_ns/1e6:.2f} ms")
    print(f"    Per step:           {decode_per_step/1e6:.4f} ms")
    print(f"    Per encoder block:  {decode_per_layer/1e6:.5f} ms")
    print(f"")
    print(f"  PREFILL:")
    print(f"    Encoder total:      {prefill_encoder_ns/1e6:.2f} ms")
    print(f"    Per encoder block:  {prefill_per_layer/1e6:.4f} ms")
    print(f"")
    print(f"  LM HEAD per call:     {lm_head_ns/max(lm_head_count,1)/1e3:.1f} us")
    print(f"{'='*60}")

    if args.output_json:
        output = {
            "num_layers": N,
            "output_len": args.output_len,
            "n_requests": n_requests,
            "n_decode_steps": n_decode,
            "total_kernel_ms": round(total_ns / 1e6, 2),
            "encoder_total_ms": round(encoder_ns / 1e6, 2),
            "lm_head_total_ms": round(lm_head_ns / 1e6, 2),
            "sampling_total_ms": round(sampling_ns / 1e6, 2),
            "decode_per_step_ms": round(decode_per_step / 1e6, 4),
            "decode_per_layer_ms": round(decode_per_layer / 1e6, 5),
            "prefill_per_layer_ms": round(prefill_per_layer / 1e6, 4),
            "lm_head_per_call_us": round(lm_head_ns / max(lm_head_count, 1) / 1e3, 1),
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
