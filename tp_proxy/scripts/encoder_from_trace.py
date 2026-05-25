#!/usr/bin/env python3
"""
Extract per-encoder-block kernel time from nsys/asys sqlite trace.

Method:
  1. Find serving phase (after largest idle gap)
  2. Sum ALL kernel durations in serving → total_kernel_time
  3. Identify lm_head calls: gemv/gemm with abnormally large duration
     (lm_head output = vocab_size, much larger than encoder projections)
  4. encoder_kernel_time = total - lm_head - other_non_encoder
  5. Separate prefill vs decode encoder by duration ranking
  6. per_layer = decode_encoder_sum / (decode_steps × num_layers)

No hardcoded kernel names required. Auto-detects gemv kernel and
uses duration threshold to separate lm_head from encoder gemv.

Usage:
    python encoder_from_trace.py --sqlite trace.sqlite --num-layers 10 --output-len 64
    python encoder_from_trace.py --sqlite trace.sqlite --num-layers 10 --output-len 64 \
        --gemv-kernel gemvt_op  # PPU
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


def find_gemv_kernel(name_durations: dict) -> str | None:
    """Find the gemv/gemm kernel name (highest total duration typically)."""
    candidates = []
    for name, durs in name_durations.items():
        nl = name.lower()
        if any(kw in nl for kw in ["gemv", "gemm", "matmul", "cutlass", "marlin"]):
            candidates.append((name, sum(durs)))
    if not candidates:
        # Fallback: just pick the kernel with highest total time
        candidates = [(n, sum(d)) for n, d in name_durations.items()]
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0] if candidates else None


def main():
    ap = argparse.ArgumentParser(description="Per-encoder-block time from trace")
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--gemv-kernel", default=None,
                    help="Gemv kernel name (auto-detect if omitted)")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    N = args.num_layers

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
    all_events = cursor.fetchall()  # (start, dur, name)
    conn.close()
    print(f"Total events: {len(all_events)}")

    # Filter to serving
    serve_start = find_serving_start(all_events)
    events = [(s, d, n) for s, d, n in all_events if s >= serve_start]
    print(f"Serving events: {len(events)}")

    # Aggregate by kernel name
    name_durations = defaultdict(list)
    for s, d, n in events:
        name_durations[n].append(d)

    total_kernel_ns = sum(d for _, d, _ in events)

    # Find gemv kernel
    gemv_name = args.gemv_kernel
    if not gemv_name:
        gemv_name = find_gemv_kernel(name_durations)
    if not gemv_name:
        print("ERROR: could not detect gemv kernel")
        sys.exit(1)
    print(f"Gemv kernel: {gemv_name}")

    # Split gemv into lm_head vs encoder by duration threshold
    gemv_durs = sorted(name_durations[gemv_name])
    n_gemv = len(gemv_durs)

    if n_gemv == 0:
        print("ERROR: no gemv calls found")
        sys.exit(1)

    # Find threshold: bimodal split
    # lm_head gemv is ~50-100x larger than encoder gemv
    # Use 10x median as threshold
    median_dur = gemv_durs[n_gemv // 2]
    threshold = median_dur * 10

    small_gemv = [d for d in gemv_durs if d <= threshold]
    large_gemv = [d for d in gemv_durs if d > threshold]

    lm_head_total_ns = sum(large_gemv)
    encoder_gemv_total_ns = sum(small_gemv)

    print(f"Gemv calls: {n_gemv}")
    print(f"  Encoder gemv (dur <= {threshold/1e3:.0f}us): {len(small_gemv)}, "
          f"total={encoder_gemv_total_ns/1e6:.2f}ms")
    print(f"  LM head gemv (dur > {threshold/1e3:.0f}us): {len(large_gemv)}, "
          f"total={lm_head_total_ns/1e6:.2f}ms, mean={sum(large_gemv)/max(len(large_gemv),1)/1e3:.0f}us")

    # Non-gemv kernels: all go into encoder (norms, activations, MoE routing, etc.)
    non_gemv_ns = total_kernel_ns - encoder_gemv_total_ns - lm_head_total_ns
    encoder_total_ns = encoder_gemv_total_ns + non_gemv_ns

    # Estimate number of forward passes
    # lm_head calls = n_decode_steps (prefill might use a different kernel)
    n_lm_head = len(large_gemv)
    # Total forward passes ≈ n_lm_head + some prefill passes
    # For simplicity: n_lm_head ≈ total decode steps
    # n_requests = n_lm_head / output_len
    n_requests = max(round(n_lm_head / args.output_len), 1)
    n_decode = n_requests * args.output_len
    n_prefill = n_requests

    # Split encoder into prefill vs decode
    # Prefill encoder kernels have larger durations (process input_len tokens)
    # Gather all non-lm_head kernel durations, sort, top portion = prefill
    all_encoder_durs = sorted(
        [d for d in small_gemv] +
        [d for s, d, n in events if n != gemv_name],
        reverse=True
    )

    # Prefill fraction: prefill does ~input_len/1 more work per kernel than decode
    # So prefill kernels are the largest ones
    # Expected: n_prefill forward passes out of (n_prefill + n_decode) total
    # Each prefill kernel is ~input_len times longer than decode kernel
    # Fraction of total time from prefill ≈ input_len / (input_len + output_len)
    # We don't know input_len here, so estimate from duration distribution

    # Simple approach: attribute lm_head time to decode only
    # decode_encoder = encoder_total - prefill_encoder
    # We estimate prefill_encoder from the tail of the duration distribution

    # For now: just report total and decode estimate
    # decode_encoder ≈ encoder_total × (output_len / (output_len + 1))
    # (rough: prefill is 1 out of output_len+1 forward passes, but heavier)
    # Better: use lm_head count as decode count
    decode_encoder_ns = encoder_total_ns * n_decode / (n_decode + n_prefill * 5)
    # The "×5" accounts for prefill being ~5x heavier per step

    decode_per_step = decode_encoder_ns / max(n_decode, 1)
    decode_per_layer = decode_per_step / N

    print(f"\n{'='*60}")
    print(f"Encoder analysis ({N} layers)")
    print(f"{'='*60}")
    print(f"  Total kernel time:    {total_kernel_ns/1e6:.2f} ms")
    print(f"  Encoder kernel time:  {encoder_total_ns/1e6:.2f} ms")
    print(f"  LM head time:         {lm_head_total_ns/1e6:.2f} ms")
    print(f"  Requests (est):       {n_requests}")
    print(f"  Decode steps:         {n_decode}")
    print(f"")
    print(f"  DECODE (estimated):")
    print(f"    Per step:           {decode_per_step/1e6:.4f} ms")
    print(f"    Per encoder block:  {decode_per_layer/1e6:.5f} ms")
    print(f"")
    print(f"  LM HEAD:")
    print(f"    Per call:           {sum(large_gemv)/max(len(large_gemv),1)/1e3:.1f} us")
    print(f"    Per step:           {lm_head_total_ns/max(n_decode+n_prefill,1)/1e6:.4f} ms")
    print(f"{'='*60}")

    if args.output_json:
        output = {
            "num_layers": N,
            "output_len": args.output_len,
            "n_requests": n_requests,
            "n_decode_steps": n_decode,
            "total_kernel_ms": round(total_kernel_ns / 1e6, 2),
            "encoder_total_ms": round(encoder_total_ns / 1e6, 2),
            "lm_head_total_ms": round(lm_head_total_ns / 1e6, 2),
            "decode_per_step_ms": round(decode_per_step / 1e6, 4),
            "decode_per_layer_ms": round(decode_per_layer / 1e6, 5),
            "lm_head_per_call_us": round(sum(large_gemv) / max(len(large_gemv), 1) / 1e3, 1),
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
