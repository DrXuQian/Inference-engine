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
    layer_types = tc.get("layer_types", [])
    n_layers = tc["num_hidden_layers"]
    n_full = sum(1 for lt in layer_types[:n_layers] if lt == "full_attention") if layer_types else n_layers
    return {
        "hidden_size": tc["hidden_size"],
        "vocab_size": tc["vocab_size"],
        "num_hidden_layers": n_layers,
        "num_key_value_heads": tc.get("num_key_value_heads", 2),
        "head_dim": tc.get("head_dim", 256),
        "n_full_attn_layers": n_full,
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
    ap.add_argument("--batch-size", type=int, default=1,
                    help="Batch size. lm_head scales by 1.1^batch, sampling scales by batch.")
    ap.add_argument("--actual-seq-len", type=int, default=None,
                    help="Actual decode seq_len (e.g. 100K for agent hit with 80%% prefix cache). "
                         "If set and > bench input_len, compensates extra KV cache read time.")
    ap.add_argument("--peak-bw", type=float, default=680,
                    help="Peak memory bandwidth GB/s (default: 680)")
    ap.add_argument("--kv-bw-util", type=float, default=0.8,
                    help="KV cache bandwidth utilization ratio (default: 0.8)")
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

    # === b) Tail from trace (lm_head + sampling) ===
    tail = None
    if args.asys_sqlite:
        print("=== b) Tail from trace (lm_head + sampling) ===")
        tail = measure_tail_from_trace(args.asys_sqlite, args.lm_head_kernel)
        if tail:
            print(f"  lm_head (batch=1): {tail['lm_head_ms']:.4f} ms")
            print(f"  sampling (batch=1): {tail['sampling_ms']:.4f} ms")
    if not tail:
        print("=== b) Tail: no trace or kernel not found, using 0 ===")
        tail = {"tail_per_step_ms": 0, "lm_head_ms": 0, "sampling_ms": 0}

    lm_head_ms = tail.get("lm_head_ms", tail["tail_per_step_ms"])
    sampling_ms = tail.get("sampling_ms", 0)
    batch = args.batch_size
    decode_comm = comm["decode_comm_ms"]
    prefill_comm = comm["prefill_comm_ms"]

    # Batch scaling: sampling × batch, lm_head read directly from trace (no scaling)
    if batch > 1:
        lm_head_scaled = lm_head_ms  # already batch-N value from trace
        sampling_scaled = sampling_ms * batch
        print(f"  batch={batch}: lm_head = {lm_head_scaled:.4f} ms (from trace)")
        print(f"  batch={batch}: sampling × {batch} = {sampling_scaled:.4f} ms")
    else:
        lm_head_scaled = lm_head_ms
        sampling_scaled = sampling_ms

    # Apply TP scaling: lm_head / tp
    if tp_size > 1:
        lm_head_final = lm_head_scaled / tp_size
        print(f"  TP={tp_size}: lm_head / {tp_size} = {lm_head_final:.4f} ms")
    else:
        lm_head_final = lm_head_scaled

    # tail to subtract from raw TPOT (batch-scaled, since bench ran at batch=N)
    tail_subtract = lm_head_scaled + sampling_scaled
    # tail to add back (TP-compensated)
    tail_add = lm_head_final + sampling_scaled
    print(f"  tail_subtract (from raw): {tail_subtract:.4f} ms "
          f"(lm_head×1.1^{batch}={lm_head_scaled:.4f} + sampling×{batch}={sampling_scaled:.4f})")
    print(f"  tail_add (compensated):   {tail_add:.4f} ms "
          f"(lm_head/{tp_size}={lm_head_final:.4f} + sampling={sampling_scaled:.4f})")
    print()

    # === d) KV cache compensation (for prefix cache hit scenarios) ===
    kv_extra_tpot_ms = 0
    if args.actual_seq_len:
        n_kv = cfg["num_key_value_heads"]
        head_dim = cfg["head_dim"]
        n_full = cfg["n_full_attn_layers"]
        # Use original layer count for full attn
        n_full_orig = round(n_full * original / cfg["num_hidden_layers"]) if cfg["num_hidden_layers"] > 0 else n_full
        # KV heads per rank
        kv_heads_per_rank = n_kv if tp_size > n_kv else n_kv // tp_size

        print(f"=== d) KV cache compensation ===")
        # Will be computed per input_len in the loop below
        print(f"  actual_seq_len: {args.actual_seq_len}")
        print(f"  full_attn_layers: {n_full_orig}, kv_heads/rank: {kv_heads_per_rank}, head_dim: {head_dim}")
        print(f"  peak_bw: {args.peak_bw} GB/s, kv_bw_util: {args.kv_bw_util}")
    print()

    # === e) Compensated Results ===
    print("=== e) Compensated Results ===" if args.actual_seq_len else "=== d) Compensated Results ===")
    print()
    print("Step 1: Extract encoder (subtract batch-scaled tail)")
    print(f"  tail_subtract = {tail_subtract:.4f} ms")
    print(f"  encoder = raw_TPOT - {tail_subtract:.4f}")
    print()
    print("Step 2: Scale encoder by layer ratio")
    print(f"  layer_scale = {original}/{pruned} = {layer_scale:.2f}x")
    print()
    print("Step 3: Add back TP-compensated tail + comm")
    print(f"  tail_add = {tail_add:.4f} ms")
    print(f"  decode_comm = {decode_comm:.3f} ms, prefill_comm = {prefill_comm:.3f} ms")
    if args.actual_seq_len:
        print("Step 4: KV cache compensation (actual_seq > bench input)")
    print()
    print("Formula:")
    print(f"  comp_TPOT = (raw_TPOT - {tail_subtract:.4f}) × {layer_scale:.2f} + {tail_add:.4f} + {decode_comm:.3f}" +
          (" + kv_extra" if args.actual_seq_len else ""))
    print(f"  comp_TTFT = (raw_TTFT - {tail_subtract:.4f}) × {layer_scale:.2f} + {tail_add:.4f} + {prefill_comm:.3f}")
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

        decode_encoder = max(raw_tpot - tail_subtract, 0)
        prefill_encoder = max(raw_ttft - tail_subtract, 0)

        comp_tpot = decode_encoder * layer_scale + tail_add + decode_comm
        comp_ttft = prefill_encoder * layer_scale + tail_add + prefill_comm

        # KV cache compensation: bench tested with input_len=il, but actual
        # decode reads KV for actual_seq_len tokens. Add extra KV read time.
        kv_extra = 0
        if args.actual_seq_len and args.actual_seq_len > il:
            delta_seq = args.actual_seq_len - il
            n_kv = cfg["num_key_value_heads"]
            head_dim_v = cfg["head_dim"]
            n_full_orig = round(cfg["n_full_attn_layers"] * original / cfg["num_hidden_layers"]) \
                if cfg["num_hidden_layers"] > 0 else cfg["n_full_attn_layers"]
            kv_heads_per_rank = n_kv if tp_size > n_kv else n_kv // tp_size
            # Extra bytes = n_full_layers × 2(K+V) × kv_heads × head_dim × delta_seq × 2(bf16)
            extra_bytes = n_full_orig * 2 * kv_heads_per_rank * head_dim_v * delta_seq * 2
            effective_bw = args.peak_bw * 1e9 * args.kv_bw_util
            kv_extra = extra_bytes / effective_bw * 1000  # ms
            comp_tpot += kv_extra

        comp_total = comp_ttft + (output_tokens - 1) * comp_tpot

        print(f"{il:>8} {raw_ttft:>10.2f} {comp_ttft:>10.2f} "
              f"{raw_tpot:>10.3f} {comp_tpot:>10.3f} {comp_total:>10.1f}" +
              (f" (kv_extra={kv_extra:.3f}ms)" if kv_extra > 0 else ""))

        entry = {
            **r,
            "comp_ttft_ms": round(comp_ttft, 3),
            "comp_tpot_ms": round(comp_tpot, 3),
            "comp_total_ms": round(comp_total, 3),
        }
        if kv_extra > 0:
            entry["kv_extra_tpot_ms"] = round(kv_extra, 3)
        compensated.append(entry)

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
