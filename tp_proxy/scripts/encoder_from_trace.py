#!/usr/bin/env python3
"""
Extract per-encoder-block time from nsys/asys trace.

Method:
  1. Find serving phase (after largest idle gap)
  2. Find request bursts (separated by >500us gaps)
  3. Within each request burst, classify kernels by instance count:
     - count divisible by N (layers) → encoder kernel
     - otherwise → non-encoder (lm_head, sampling, scheduling)
  4. encoder_span = last_encoder_kernel.end - first_encoder_kernel.start
  5. Each request burst = 1 prefill + output_len decode steps
  6. per_layer_per_step = encoder_span / (output_len + 1) / N

Works with both nsys sqlite (CUPTI tables) and asys sqlite (HGPTI tables).
No hardcoded kernel names — uses instance count pattern detection.

Usage:
    python encoder_from_trace.py --sqlite profile.sqlite --num-layers 10 --output-len 64
"""

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict


def detect_kernel_table(cursor) -> str:
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    for t in tables:
        if "KERNEL" in t and "ACTIVITY" in t:
            return t
    return ""


def load_events(sqlite_path: str, platform: str) -> list[dict]:
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    kt = detect_kernel_table(cursor)
    if not kt:
        print("ERROR: no kernel table found")
        sys.exit(1)
    print(f"Kernel table: {kt}")

    cursor.execute(f"""
        SELECT k.start, k."end" - k.start AS dur, k."end", s.value AS name
        FROM {kt} k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """)
    events = [{"start": r[0], "dur": r[1], "end": r[2], "name": r[3]}
              for r in cursor.fetchall()]
    conn.close()
    return events


def find_serving_phase(events: list[dict]) -> int:
    """Return start timestamp of serving phase (after largest idle gap)."""
    if not events:
        return 0
    t_min = events[0]["start"]
    bins = defaultdict(int)
    for e in events:
        bins[(e["start"] - t_min) // 1_000_000_000] += 1

    max_bin = max(bins.keys()) if bins else 0
    best_start = best_len = 0
    gap_start = gap_len = 0
    in_gap = False
    for b in range(max_bin + 1):
        if bins[b] == 0:
            if not in_gap:
                gap_start = b; in_gap = True; gap_len = 1
            else:
                gap_len += 1
        else:
            if in_gap and gap_len > best_len:
                best_start = gap_start; best_len = gap_len
            in_gap = False

    return t_min + (best_start + best_len) * 1_000_000_000


def find_decode_bursts(events: list[dict], serve_start: int,
                       gap_us: float = 500) -> list[list[dict]]:
    """Find kernel bursts (decode steps) in serving phase."""
    serve = [e for e in events if e["start"] >= serve_start]
    if not serve:
        return []

    gap_ns = int(gap_us * 1000)
    bursts = []
    current = [serve[0]]
    for e in serve[1:]:
        if e["start"] - current[-1]["end"] > gap_ns:
            bursts.append(current)
            current = [e]
        else:
            current.append(e)
    if current:
        bursts.append(current)
    return bursts


def split_prefill_decode(burst: list[dict], output_len: int) -> tuple[list, list]:
    """Split a request burst into prefill kernels and decode kernels.

    Prefill is the first forward pass (much heavier, processes all input tokens).
    Decode is the remaining output_len forward passes (one token each).

    Heuristic: find the largest internal gap — that's the prefill/decode boundary.
    Prefill runs in eager mode, followed by CUDA Graph decode steps with tiny gaps.
    """
    if len(burst) < 2:
        return burst, []

    # Find all internal gaps
    gaps = []
    for i in range(1, len(burst)):
        gap = burst[i]["start"] - burst[i - 1]["end"]
        gaps.append((i, gap))

    # The prefill/decode boundary is typically the largest gap in the first portion
    # (prefill is eager → large gap → first CUDA Graph decode)
    # Look in the first 30% of the burst for the boundary
    search_range = max(len(gaps) // 3, 10)
    best_idx = 0
    best_gap = 0
    for i, gap in gaps[:search_range]:
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    if best_gap > 50_000:  # > 50us gap = likely prefill/decode boundary
        prefill = burst[:best_idx]
        decode = burst[best_idx:]
    else:
        # No clear boundary — estimate by kernel count
        # Prefill has more kernels (processes input_len tokens)
        # Decode steps are uniform (1 token each), so decode ≈ rest
        est_decode_kernels = len(burst) * output_len / (output_len + 1)
        split_point = len(burst) - int(est_decode_kernels)
        prefill = burst[:split_point]
        decode = burst[split_point:]

    return prefill, decode


def classify_burst(burst: list[dict], num_layers: int) -> dict:
    """Classify kernels in a burst into encoder vs non-encoder by instance count."""
    name_counts = Counter(e["name"] for e in burst)

    encoder_names = set()
    for name, count in name_counts.items():
        if count >= num_layers:
            ratio = count / num_layers
            if abs(ratio - round(ratio)) < 0.3:
                encoder_names.add(name)

    encoder_events = [e for e in burst if e["name"] in encoder_names]

    if not encoder_events:
        return {"encoder_span_us": 0, "total_wall_us": 0,
                "n_encoder_kernels": 0, "n_encoder_types": 0,
                "n_total_kernels": len(burst)}

    enc_start = encoder_events[0]["start"]
    enc_end = encoder_events[-1]["end"]
    encoder_span = (enc_end - enc_start) / 1e3

    burst_start = burst[0]["start"]
    burst_end = burst[-1]["end"]
    total_wall = (burst_end - burst_start) / 1e3

    return {
        "encoder_span_us": encoder_span,
        "total_wall_us": total_wall,
        "n_encoder_kernels": len(encoder_events),
        "n_encoder_types": len(encoder_names),
        "n_total_kernels": len(burst),
    }


def main():
    ap = argparse.ArgumentParser(description="Per-encoder-block time from trace")
    ap.add_argument("--sqlite", required=True, help="nsys/asys sqlite trace")
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True,
                    help="Output tokens per request (to compute per-step time)")
    ap.add_argument("--platform", choices=["nvidia", "ppu"], default="nvidia")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    N = args.num_layers
    n_steps = args.output_len + 1  # 1 prefill + output_len decode
    print(f"Loading trace: {args.sqlite}")
    events = load_events(args.sqlite, args.platform)
    print(f"Total events: {len(events)}")

    # Find serving phase
    serve_start = find_serving_phase(events)
    serve_events = [e for e in events if e["start"] >= serve_start]
    print(f"Serving events: {len(serve_events)}")

    # Find decode bursts
    bursts = find_decode_bursts(events, serve_start)
    burst_sizes = [len(b) for b in bursts]
    print(f"Bursts found: {len(bursts)}")
    if not bursts:
        print("ERROR: no bursts found")
        sys.exit(1)

    # Filter to decode bursts: take the largest cluster of similar-sized bursts
    # (decode steps have consistent kernel count, much larger than misc bursts)
    max_size = max(burst_sizes)
    # Decode bursts are the large ones (>50% of max size)
    decode_bursts = [b for b in bursts if len(b) > max_size * 0.5]
    if decode_bursts:
        typical = len(decode_bursts[0])
    else:
        typical = max_size
    print(f"Decode bursts (size ~{typical}): {len(decode_bursts)}")

    if not decode_bursts:
        print("ERROR: no typical decode bursts found")
        sys.exit(1)

    # Classify each burst: split into prefill + decode, then classify decode
    decode_results = []
    prefill_results = []

    for burst in decode_bursts:
        prefill_kernels, decode_kernels = split_prefill_decode(burst, args.output_len)

        if decode_kernels:
            r = classify_burst(decode_kernels, N)
            if r["encoder_span_us"] > 0:
                decode_results.append(r)

        if prefill_kernels:
            r = classify_burst(prefill_kernels, N)
            if r["encoder_span_us"] > 0:
                prefill_results.append(r)

    if not decode_results:
        print("ERROR: no encoder kernels detected in decode phase")
        sys.exit(1)

    # Decode: encoder_span covers output_len decode steps
    decode_spans = sorted(r["encoder_span_us"] for r in decode_results)
    mid = len(decode_spans) // 2
    decode_span = decode_spans[mid]
    decode_per_step = decode_span / args.output_len
    decode_per_layer = decode_per_step / N

    # Prefill
    prefill_span = 0
    prefill_per_layer = 0
    if prefill_results:
        prefill_spans = sorted(r["encoder_span_us"] for r in prefill_results)
        prefill_span = prefill_spans[len(prefill_spans) // 2]
        prefill_per_layer = prefill_span / N

    print(f"\n{'='*60}")
    print(f"Encoder block analysis ({N} layers)")
    print(f"{'='*60}")
    print(f"  Requests analyzed: {len(decode_results)}")
    print(f"")
    print(f"  DECODE ({args.output_len} steps):")
    print(f"    Encoder span:      {decode_span/1000:.3f} ms")
    print(f"    Per step:          {decode_per_step/1000:.4f} ms")
    print(f"    Per encoder block: {decode_per_layer/1000:.5f} ms")
    print(f"")
    if prefill_results:
        print(f"  PREFILL (1 step):")
        print(f"    Encoder span:      {prefill_span/1000:.3f} ms")
        print(f"    Per encoder block: {prefill_per_layer/1000:.4f} ms")
    print(f"{'='*60}")

    if args.output_json:
        output = {
            "num_layers": N,
            "output_len": args.output_len,
            "decode_encoder_span_ms": round(decode_span / 1000, 3),
            "decode_per_step_ms": round(decode_per_step / 1000, 4),
            "decode_per_layer_ms": round(decode_per_layer / 1000, 5),
            "prefill_encoder_span_ms": round(prefill_span / 1000, 3),
            "prefill_per_layer_ms": round(prefill_per_layer / 1000, 4),
            "n_requests": len(decode_results),
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
