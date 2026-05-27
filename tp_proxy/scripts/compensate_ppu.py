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
    """Extract per-step decode breakdown from trace using graphNodeId.

    Uses graphNodeId > 0 to identify CUDA Graph (decode) kernels.
    Within decode kernels, finds two consecutive lm_head kernels (largest),
    measures encoder = kernels between them (excluding prev lm_head + sampling).

    Returns: lm_head_ms, sampling_ms, encoder_ms, overhead_ms
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

    # Check if graphNodeId column exists
    cursor.execute(f'PRAGMA table_info("{kt}")')
    columns = [c[1] for c in cursor.fetchall()]
    has_graph_id = "graphNodeId" in columns

    # Query: include graphNodeId if available
    if has_graph_id:
        cursor.execute(f"""
            SELECT k.start, k."end" - k.start AS dur, k."end",
                   s.value AS name, k.graphNodeId
            FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start
        """)
        all_events = [(r[0], r[1], r[2], r[3], r[4] or 0) for r in cursor.fetchall()]
    else:
        cursor.execute(f"""
            SELECT k.start, k."end" - k.start AS dur, k."end", s.value AS name
            FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start
        """)
        all_events = [(r[0], r[1], r[2], r[3], 0) for r in cursor.fetchall()]
    conn.close()

    if not all_events:
        return None

    # Filter decode-only kernels (graphNodeId > 0)
    if has_graph_id:
        decode_events = [(s, d, e, n, g) for s, d, e, n, g in all_events if g > 0]
        n_graph = len(decode_events)
        n_nongraph = len(all_events) - n_graph
        print(f"  graphNodeId: {n_graph} decode, {n_nongraph} non-decode (prefill/setup)")
    else:
        decode_events = all_events
        print(f"  WARNING: no graphNodeId column, using all {len(all_events)} kernels")

    if not decode_events:
        print("  WARNING: no decode kernels found")
        return None

    # Find all lm_head kernels in decode
    lm_heads = [(i, decode_events[i][1]) for i in range(len(decode_events))
                if lm_head_kernel in decode_events[i][3]]

    if not lm_heads:
        print(f"  WARNING: lm_head kernel '{lm_head_kernel}' not found in decode kernels")
        print(f"  Last 10 decode kernels:")
        for ev in decode_events[-10:]:
            print(f"    {ev[3][:80]}  dur={ev[1]/1e3:.1f}us")
        return None

    print(f"  Found {len(lm_heads)} lm_head kernels in decode")

    # Cluster lm_head by duration: split into groups where durations are similar.
    # In multi-batch, batch=1 lm_head (small) and batch=N lm_head (large) coexist.
    # Use simple threshold: sort by duration, find largest gap → two clusters.
    durations = sorted(set(h[1] for h in lm_heads))
    if len(durations) >= 2:
        # Find largest relative gap between consecutive sorted durations
        max_gap_ratio = 0
        split_val = durations[-1]
        for i in range(1, len(durations)):
            ratio = durations[i] / durations[i - 1] if durations[i - 1] > 0 else 1
            if ratio > max_gap_ratio:
                max_gap_ratio = ratio
                split_val = (durations[i - 1] + durations[i]) / 2
        # Take the cluster with largest durations (≥ split_val)
        large_cluster = [(idx, dur) for idx, dur in lm_heads if dur >= split_val]
        small_cluster = [(idx, dur) for idx, dur in lm_heads if dur < split_val]
        if large_cluster:
            print(f"  Clusters: large={len(large_cluster)} (≥{split_val/1e3:.1f}us), "
                  f"small={len(small_cluster)} (<{split_val/1e3:.1f}us)")
            lm_heads = large_cluster
        # If gap ratio < 1.5, all lm_heads are similar size → single cluster
        if max_gap_ratio < 1.5:
            print(f"  No significant gap (ratio={max_gap_ratio:.2f}), single cluster")
            lm_heads = [(idx, dur) for idx, dur in lm_heads]  # keep all

    if len(lm_heads) < 2:
        idx = lm_heads[0][0]
        lm_head_ns = decode_events[idx][1]
        samp_ns = sum(d for _, d, _, _, _ in decode_events[idx + 1:])
        print(f"  lm_head: {lm_head_ns / 1e6:.4f} ms")
        print(f"  sampling: {samp_ns / 1e6:.4f} ms")
        print(f"  WARNING: only 1 lm_head in cluster, cannot compute encoder")
        return {
            "lm_head_ms": round(lm_head_ns / 1e6, 4),
            "sampling_ms": round(samp_ns / 1e6, 4),
            "encoder_ms": 0, "overhead_ms": 0,
            "tail_per_step_ms": round((lm_head_ns + samp_ns) / 1e6, 4),
        }

    # Pick last two adjacent lm_heads from the large cluster
    lm_heads.sort(key=lambda x: x[0])  # sort by position
    prev_idx, prev_dur = lm_heads[-2]
    curr_idx, curr_dur = lm_heads[-1]
    print(f"  Selected adjacent pair: dur={prev_dur/1e3:.1f}us (pos {prev_idx}), "
          f"{curr_dur/1e3:.1f}us (pos {curr_idx})")

    # lm_head time = current (last) lm_head duration
    lm_head_ns = curr_dur

    # sampling = decode kernels after curr_lm_head (until next lm_head or end)
    # Find next lm_head after curr_idx (if any), otherwise use end
    next_lm = len(decode_events)
    for i in range(curr_idx + 1, len(decode_events)):
        if lm_head_kernel in decode_events[i][3]:
            next_lm = i
            break
    sampling_ns = sum(d for _, d, _, _, _ in decode_events[curr_idx + 1:next_lm])

    # encoder = decode kernels between prev_lm_head and curr_lm_head,
    # excluding prev_lm_head itself and prev step's sampling
    # Structure: [prev_lm_head] [prev_sampling...] [encoder...] [curr_lm_head]
    between = decode_events[prev_idx + 1:curr_idx]

    # prev step's sampling = kernels right after prev_lm_head until
    # next encoder-like kernel. Use same logic: find next lm_head-like
    # pattern. Simpler: total_between - prev_lm - sampling ≈ encoder
    # Best: encoder = total_between - (sampling from prev step)
    # prev sampling ≈ same as curr sampling
    total_between_ns = sum(d for _, d, _, _, _ in between)
    encoder_ns = max(total_between_ns - sampling_ns, 0)  # subtract prev step's sampling

    # Wall clock for one step
    step_wall_ns = decode_events[curr_idx][0] - decode_events[prev_idx][0]
    all_kernel_ns = sum(d for _, d, _, _, _ in decode_events[prev_idx:curr_idx + 1])
    overhead_ns = max(step_wall_ns - all_kernel_ns, 0)

    print(f"  encoder (between 2 lm_heads - sampling): {encoder_ns / 1e6:.4f} ms")
    print(f"  lm_head:  {lm_head_ns / 1e6:.4f} ms")
    print(f"  sampling: {sampling_ns / 1e6:.4f} ms")
    print(f"  step wall: {step_wall_ns / 1e6:.4f} ms")
    print(f"  overhead:  {overhead_ns / 1e6:.4f} ms (not scaled)")

    return {
        "lm_head_ms": round(lm_head_ns / 1e6, 4),
        "sampling_ms": round(sampling_ns / 1e6, 4),
        "encoder_ms": round(encoder_ns / 1e6, 4),
        "overhead_ms": round(overhead_ns / 1e6, 4),
        "tail_per_step_ms": round((lm_head_ns + sampling_ns) / 1e6, 4),
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
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--sampling-trace", default=None,
                    help="Batch=1 trace sqlite for sampling time (used when --batch-size>=2)")
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
            print(f"  lm_head: {tail['lm_head_ms']:.4f} ms")
            print(f"  sampling: {tail['sampling_ms']:.4f} ms")
    if not tail:
        print("=== b) Tail: no trace or kernel not found, using 0 ===")
        tail = {"tail_per_step_ms": 0, "lm_head_ms": 0, "sampling_ms": 0}

    lm_head_ms = tail.get("lm_head_ms", 0)
    sampling_ms = tail.get("sampling_ms", 0)
    encoder_ms = tail.get("encoder_ms", 0)
    overhead_ms = tail.get("overhead_ms", 0)
    batch = args.batch_size
    decode_comm = comm["decode_comm_ms"]
    prefill_comm = comm["prefill_comm_ms"]

    # For batch>=2: lm_head from batch=N trace, sampling from batch=1 trace
    if batch >= 2 and args.sampling_trace:
        print(f"  Reading sampling from batch=1 trace: {args.sampling_trace}")
        b1_tail = measure_tail_from_trace(args.sampling_trace, "gemvt_op")
        if b1_tail:
            sampling_ms = b1_tail["sampling_ms"]
            print(f"  sampling (batch=1): {sampling_ms:.4f} ms")

    # TP scaling: lm_head / tp
    if tp_size > 1:
        lm_head_final = lm_head_ms / tp_size
        print(f"  TP={tp_size}: lm_head / {tp_size} = {lm_head_final:.4f} ms")
    else:
        lm_head_final = lm_head_ms

    # COMP_MODE env: "trace" (precise, default) or "tpot" (old method)
    comp_mode = os.environ.get("COMP_MODE", "trace").lower()
    use_trace_encoder = encoder_ms > 0 and comp_mode == "trace"
    if use_trace_encoder:
        print(f"\n  Mode: trace-based (COMP_MODE=trace)")
        print(f"  encoder (from trace): {encoder_ms:.4f} ms/step ({pruned} layers)")
        print(f"  overhead (not scaled): {overhead_ms:.4f} ms")
    else:
        if comp_mode == "tpot" and encoder_ms > 0:
            print(f"\n  Mode: TPOT-based (COMP_MODE=tpot, forced)")
        else:
            print(f"\n  Mode: TPOT-based (no encoder from trace)")
    print()
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

    # === Compensated Results ===
    print("=== Compensated Results ===")
    print()
    if use_trace_encoder:
        print("Formula (trace-based, precise):")
        print(f"  comp_TPOT = encoder({encoder_ms:.4f}) × {layer_scale:.2f} "
              f"+ lm_head({lm_head_final:.4f}) + sampling({sampling_ms:.4f}) "
              f"+ overhead({overhead_ms:.4f}) + comm({decode_comm:.3f})")
    else:
        tail_subtract = lm_head_ms + sampling_ms
        tail_add = lm_head_final + sampling_ms
        print("Formula (TPOT-based, fallback):")
        print(f"  comp_TPOT = (raw_TPOT - {tail_subtract:.4f}) × {layer_scale:.2f} "
              f"+ {tail_add:.4f} + {decode_comm:.3f}")
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

        if use_trace_encoder:
            # Precise: encoder from trace, overhead not scaled
            comp_tpot = encoder_ms * layer_scale + lm_head_final + sampling_ms + overhead_ms + decode_comm
            # For TTFT: use raw_TTFT - (encoder + lm_head + sampling + overhead) as prefill non-encoder,
            # then scale encoder part. Simpler: scale proportionally.
            prefill_encoder = max(raw_ttft - lm_head_ms - sampling_ms - overhead_ms, 0)
            comp_ttft = prefill_encoder * layer_scale + lm_head_final + sampling_ms + overhead_ms + prefill_comm
        else:
            tail_subtract = lm_head_ms + sampling_ms
            tail_add = lm_head_final + sampling_ms
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
