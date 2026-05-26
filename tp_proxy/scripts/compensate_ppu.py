#!/usr/bin/env python3
"""
One-script compensation for PPU platform.

Combines:
  a) Communication (from comm.json, pccl_tools standalone)
  b) Tail from trace (lm_head + sampling per step)
  c) Encoder scaling (subtract tail, scale by layer ratio, add back)

Usage:
    python compensate_ppu.py \
        --bench-results bench.json \
        --model-dir /path/to/model \
        --asys-sqlite trace.sqlite

    python compensate_ppu.py \
        --bench-results bench.json \
        --model-dir /path/to/model \
        --asys-sqlite trace.sqlite \
        --comm-json comm.json \
        --pruned-layers 4 --original-layers 24 --tp-size 2
"""

import argparse
import json
import os
import sqlite3
import sys


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_model_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    return {
        "hidden_size": tc["hidden_size"],
        "vocab_size": tc["vocab_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
    }


def load_meta(model_dir: str) -> dict | None:
    for d in [model_dir, os.path.dirname(model_dir)]:
        p = os.path.join(d, "split_meta.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


# ---------------------------------------------------------------------------
# a) Tail from trace (lm_head + sampling per step)
# ---------------------------------------------------------------------------

def measure_tail_from_trace(sqlite_path: str,
                            lm_head_kernel: str) -> dict | None:
    """Extract tail time (lm_head + sampling) per decode step from trace.

    Walk backward from end of trace to find the last lm_head kernel
    (substring match). Sum kernel durations from there to end = tail.
    """
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Find kernel table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    kt = ""
    for r in cursor.fetchall():
        if "KERNEL" in r[0].upper() and "ACTIVITY" in r[0].upper():
            kt = r[0]; break
    if not kt:
        conn.close()
        return None

    cursor.execute(f"""
        SELECT k.start, k."end" - k.start AS dur, k."end", s.value AS name
        FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """)
    all_events = [(r[0], r[1], r[2], r[3]) for r in cursor.fetchall()]
    conn.close()

    if not all_events:
        return None

    # Walk backward from end to find last lm_head kernel (substring match)
    tail_start_idx = None
    for i in range(len(all_events) - 1, -1, -1):
        if lm_head_kernel in all_events[i][3]:
            tail_start_idx = i
            break

    if tail_start_idx is None:
        print("  WARNING: lm_head kernel not found by substring match")
        print(f"  Looking for: '{lm_head_kernel}'")
        print(f"  Last 10 kernels:")
        for ev in all_events[-10:]:
            print(f"    {ev[3][:80]}  dur={ev[1]/1e3:.1f}us")
        return None

    # Split tail into lm_head (the gemvt kernel) and sampling (everything after)
    lm_head_ns = all_events[tail_start_idx][1]
    sampling_kernels = all_events[tail_start_idx + 1:]
    sampling_ns = sum(d for _, d, _, _ in sampling_kernels)
    tail_kernel_ns = lm_head_ns + sampling_ns

    print(f"  Last lm_head kernel: '{all_events[tail_start_idx][3][:60]}'")
    print(f"  lm_head time:  {lm_head_ns / 1e6:.4f} ms")
    print(f"  sampling time: {sampling_ns / 1e6:.4f} ms ({len(sampling_kernels)} kernels)")
    print(f"  tail total:    {tail_kernel_ns / 1e6:.4f} ms")

    return {
        "lm_head_ms": round(lm_head_ns / 1e6, 4),
        "sampling_ms": round(sampling_ns / 1e6, 4),
        "tail_per_step_ms": round(tail_kernel_ns / 1e6, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PPU compensation")
    ap.add_argument("--bench-results", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-len", type=int, default=None)
    ap.add_argument("--comm-json", default=None,
                    help="comm.json from comm_bench.sh (AR/AG latency)")
    ap.add_argument("--asys-sqlite", default=None,
                    help="asys trace sqlite for tail extraction")
    ap.add_argument("--pruned-layers", type=int, default=None)
    ap.add_argument("--original-layers", type=int, default=None)
    ap.add_argument("--tp-size", type=int, default=None)
    ap.add_argument("--lm-head-kernel", default="gemvt_op")
    ap.add_argument("--output-json", default="compensated_ppu.json")
    args = ap.parse_args()

    # Load bench results
    with open(args.bench_results) as f:
        bench = json.load(f)

    # Load config + meta
    cfg = load_model_config(args.model_dir)
    meta = load_meta(args.model_dir)
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]

    tp_size = args.tp_size or (meta and meta.get("tp_size")) or 1
    pruned = args.pruned_layers or (meta and meta.get("pruned_layers")) or cfg["num_hidden_layers"]
    original = args.original_layers or (meta and meta.get("original_layers")) or pruned
    layer_scale = original / pruned if pruned > 0 else 1

    results_list = bench.get("results", [bench])

    # Auto-read output_len
    output_len = args.output_len
    if output_len is None:
        output_len = bench.get("output_len") or next(
            (r.get("output_len") or r.get("output_tokens") for r in results_list if "error" not in r), 64)

    print(f"Model: hidden={hidden}, vocab={vocab}")
    print(f"Layers: {pruned} pruned / {original} original (scale={layer_scale:.2f}x), TP={tp_size}")
    print(f"Output len: {output_len}")
    print()

    # === a) Communication ===
    if args.comm_json:
        with open(args.comm_json) as f:
            comm = json.load(f)
        decode_comm = comm.get("decode_total_per_step_ms", comm.get("total_per_step_ms", 0))
        prefill_comm = comm.get("prefill_total_per_step_ms", decode_comm)
        comm["decode_comm_ms"] = decode_comm
        comm["prefill_comm_ms"] = prefill_comm
        print(f"=== a) Communication (from {args.comm_json}) ===")
        print(f"  Decode:  {decode_comm:.3f} ms/step")
        print(f"  Prefill: {prefill_comm:.3f} ms/step")
    elif tp_size > 1:
        print("=== a) Communication: no --comm-json, using 0 ===")
        print("    Run comm_scenarios first to measure AR/AG latency")
        comm = {"decode_comm_ms": 0, "prefill_comm_ms": 0, "method": "none"}
    else:
        print("=== a) Communication: TP=1, skip ===")
        comm = {"decode_comm_ms": 0, "prefill_comm_ms": 0, "method": "none"}
    print()

    # === b) Tail from trace ===
    tail = None
    if args.asys_sqlite:
        print("=== b) Tail from trace (lm_head + sampling) ===")
        tail = measure_tail_from_trace(args.asys_sqlite, args.lm_head_kernel)
        if tail:
            print(f"  → tail_per_step: {tail['tail_per_step_ms']:.4f} ms")
    if not tail:
        print("=== b) Tail: no trace or kernel not found, using 0 ===")
        tail = {"tail_per_step_ms": 0}
    print()

    # === c) LM head compensation (TP>1: scale down by 1/tp) ===
    tail_ms = tail["tail_per_step_ms"]
    lm_head_ms = tail.get("lm_head_ms", tail_ms)
    sampling_ms = tail.get("sampling_ms", 0)
    decode_comm = comm["decode_comm_ms"]
    prefill_comm = comm["prefill_comm_ms"]

    if tp_size > 1:
        lm_head_comp = lm_head_ms / tp_size
        print(f"=== c) LM head compensation (TP={tp_size}) ===")
        print(f"  lm_head (proxy, full vocab): {lm_head_ms:.4f} ms")
        print(f"  lm_head (TP={tp_size}, vocab/{tp_size}):  {lm_head_comp:.4f} ms")
        print(f"  delta: -{lm_head_ms - lm_head_comp:.4f} ms")
    else:
        lm_head_comp = lm_head_ms
        print("=== c) LM head: TP=1, no compensation ===")
    print()

    tail_comp = lm_head_comp + sampling_ms

    # === d) Compensated Results ===
    print("=== d) Compensated Results ===")
    print()
    print("Step 1: Extract encoder time")
    print(f"  tail (proxy) = lm_head({lm_head_ms:.4f}) + sampling({sampling_ms:.4f}) = {tail_ms:.4f} ms")
    print(f"  decode_encoder = raw_TPOT - tail")
    print(f"  prefill_encoder = raw_TTFT - tail")
    print()
    print("Step 2: Scale encoder by layer ratio")
    print(f"  layer_scale = {original}/{pruned} = {layer_scale:.2f}x")
    print()
    print("Step 3: Add back tail (lm_head compensated) + comm")
    print(f"  tail_comp = lm_head/{tp_size}({lm_head_comp:.4f}) + sampling({sampling_ms:.4f}) = {tail_comp:.4f} ms")
    print(f"  decode_comm = {decode_comm:.3f} ms, prefill_comm = {prefill_comm:.3f} ms")
    print()
    print("Formula:")
    print(f"  comp_TPOT = decode_encoder × {layer_scale:.2f} + {tail_comp:.4f} + {decode_comm:.3f}")
    print(f"  comp_TTFT = prefill_encoder × {layer_scale:.2f} + {tail_comp:.4f} + {prefill_comm:.3f}")
    print()
    print(f"{'input':>8} {'raw_ttft':>10} {'comp_ttft':>10} "
          f"{'raw_tpot':>10} {'comp_tpot':>10} {'comp_total':>10}")
    print("-" * 65)

    compensated = []
    for r in results_list:
        if "error" in r:
            compensated.append(r); continue

        il = r["input_len"]
        raw_ttft = r["ttft_median_ms"]
        raw_tpot = r["tpot_median_ms"]
        output_tokens = r.get("output_tokens", output_len)

        decode_encoder = raw_tpot - tail_ms
        prefill_encoder = raw_ttft - tail_ms

        comp_tpot = decode_encoder * layer_scale + tail_comp + decode_comm
        comp_ttft = prefill_encoder * layer_scale + tail_comp + prefill_comm

        comp_total = comp_ttft + (output_tokens - 1) * comp_tpot

        print(f"{il:>8} {raw_ttft:>10.2f} {comp_ttft:>10.2f} "
              f"{raw_tpot:>10.3f} {comp_tpot:>10.3f} {comp_total:>10.1f}")

        compensated.append({
            **r,
            "comp_ttft_ms": round(comp_ttft, 3),
            "comp_tpot_ms": round(comp_tpot, 3),
            "comp_total_ms": round(comp_total, 3),
        })

    # Save
    output = {
        "platform": "ppu",
        "tp_size": tp_size,
        "pruned_layers": pruned,
        "original_layers": original,
        "layer_scale": layer_scale,
        "communication": comm,
        "tail": tail,
        "results": compensated,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
