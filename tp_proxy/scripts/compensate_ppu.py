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
    """Extract per-step decode breakdown using graphNodeId.

    Decode step structure:
      [CUDA Graph: encoder, graphNodeId>0] [gap: lm_head+sampling, graphNodeId=0]

    Uses graph kernel count consistency to filter correct batch size steps.
    """
    from collections import Counter

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    kt = ""
    for r in cursor.fetchall():
        if "KERNEL" in r[0].upper() and "ACTIVITY" in r[0].upper():
            kt = r[0]; break
    if not kt:
        conn.close(); return None

    cursor.execute(f'PRAGMA table_info("{kt}")')
    columns = [c[1] for c in cursor.fetchall()]
    if "graphNodeId" not in columns:
        # Fallback: no graphNodeId, use last lm_head
        cursor.execute(f"""
            SELECT k.start, k."end" - k.start AS dur, k."end", s.value AS name
            FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start
        """)
        all_events = [(r[0], r[1], r[2], r[3]) for r in cursor.fetchall()]
        conn.close()
        if not all_events:
            return None
        for i in range(len(all_events) - 1, -1, -1):
            if lm_head_kernel in all_events[i][3]:
                lm_ns = all_events[i][1]
                samp_ns = sum(d for _, d, _, _ in all_events[i+1:])
                print(f"  WARNING: no graphNodeId, fallback to last lm_head")
                return {"lm_head_ms": round(lm_ns/1e6, 4),
                        "sampling_ms": round(samp_ns/1e6, 4),
                        "encoder_ms": 0, "overhead_ms": 0,
                        "tail_per_step_ms": round((lm_ns+samp_ns)/1e6, 4)}
        return None

    cursor.execute(f'''
        SELECT k.start, k."end" - k.start AS dur, k."end",
               s.value AS name, k.graphNodeId
        FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    ''')
    all_events = [(r[0], r[1], r[2], r[3], r[4] or 0) for r in cursor.fetchall()]
    conn.close()

    n_graph = sum(1 for *_, g in all_events if g > 0)
    print(f"  Total: {len(all_events)} kernels ({n_graph} graph, {len(all_events)-n_graph} non-graph)")

    # Split into alternating graph/gap segments
    segments = []
    cur_type = "graph" if all_events[0][4] > 0 else "gap"
    cur_evts = [all_events[0]]
    for ev in all_events[1:]:
        t = "graph" if ev[4] > 0 else "gap"
        if t == cur_type:
            cur_evts.append(ev)
        else:
            segments.append((cur_type, cur_evts))
            cur_type = t
            cur_evts = [ev]
    segments.append((cur_type, cur_evts))

    # Pair: each decode step = (graph_seg, gap_seg)
    steps = []
    for i in range(len(segments) - 1):
        if segments[i][0] == "graph" and segments[i+1][0] == "gap":
            steps.append((segments[i][1], segments[i+1][1]))

    if not steps:
        print("  WARNING: no (graph, gap) pairs found")
        return None
    print(f"  Decode steps (graph→gap pairs): {len(steps)}")

    # Filter to consistent graph kernel count (skip warmup/capture)
    graph_counts = [len(s[0]) for s in steps]
    count_freq = Counter(graph_counts)
    most_common_count = count_freq.most_common(1)[0][0]
    consistent_steps = [(g, gap) for g, gap in steps if len(g) == most_common_count]
    print(f"  Graph kernel count mode: {most_common_count} "
          f"({len(consistent_steps)}/{len(steps)} steps)")

    if not consistent_steps:
        return None

    # Among consistent steps, find ones with matching lm_head kernel
    matched_steps = []
    for g, gap in consistent_steps:
        for ev in gap:
            if lm_head_kernel in ev[3]:
                matched_steps.append((g, gap, ev[1]))  # (graph, gap, lm_head_dur)
                break
    if matched_steps:
        print(f"  Steps with '{lm_head_kernel}' in gap: {len(matched_steps)}")
        # Pick steps with largest lm_head duration (= largest actual batch size)
        # Cluster: take top 25% by lm_head duration
        lm_durs = sorted([d for _, _, d in matched_steps])
        threshold = lm_durs[len(lm_durs) * 3 // 4] if len(lm_durs) > 4 else lm_durs[0]
        top_steps = [(g, gap) for g, gap, d in matched_steps if d >= threshold]
        print(f"  lm_head dur range: {lm_durs[0]/1e3:.1f} - {lm_durs[-1]/1e3:.1f} us, "
              f"threshold(p75): {threshold/1e3:.1f} us, selected: {len(top_steps)} steps")
        consistent_steps = top_steps
    else:
        print(f"  WARNING: '{lm_head_kernel}' not found in any gap")

    # Use last step from the top-batch cluster
    graph_evts, gap_evts = consistent_steps[-1]

    # Encoder = wall clock of CUDA Graph region (not kernel sum, kernels may overlap)
    encoder_ns = graph_evts[-1][2] - graph_evts[0][0]

    # Gap = wall clock between two CUDA Graph regions
    # lm_head = kernel matching lm_head_kernel name
    # sampling = gap wall - lm_head
    gap_wall_ns = gap_evts[-1][2] - gap_evts[0][0] if len(gap_evts) > 1 else (gap_evts[0][1] if gap_evts else 0)
    gap_total_ns = gap_wall_ns
    lm_head_ns = 0
    for ev in gap_evts:
        if lm_head_kernel in ev[3]:
            lm_head_ns = ev[1]
            break
    sampling_ns = gap_total_ns - lm_head_ns

    if lm_head_ns == 0:
        print(f"  WARNING: '{lm_head_kernel}' not found in gap. Gap kernels:")
        for ev in gap_evts[:5]:
            print(f"    {ev[3][:70]}  dur={ev[1]/1e3:.1f}us")
        # Fallback: largest kernel = lm_head
        if gap_evts:
            largest = max(gap_evts, key=lambda e: e[1])
            lm_head_ns = largest[1]
            sampling_ns = gap_total_ns - lm_head_ns

    # Step wall = from first graph kernel to last gap kernel (full step)
    step_wall_ns = gap_evts[-1][2] - graph_evts[0][0] if gap_evts else encoder_wall_ns
    step_kernel_ns = encoder_ns + lm_head_ns + sampling_ns
    overhead_ns = max(step_wall_ns - step_kernel_ns, 0)

    # TPOT = step wall clock (one decode step)
    tpot_ms = step_wall_ns / 1e6

    # Prefill (TTFT): find the last large gap segment before decode starts.
    # Prefill = non-graph kernels, typically the largest gap segment.
    # Look for the gap segment right before the first consistent graph segment.
    first_consistent_graph = consistent_steps[0][0]
    first_graph_start = first_consistent_graph[0][0]  # start time of first graph kernel

    # All non-graph kernels before first consistent decode graph = prefill
    prefill_ns = 0
    prefill_count = 0
    for s, d, e, n, g in all_events:
        if s >= first_graph_start:
            break
        if g == 0:
            prefill_ns += d
            prefill_count += 1
    # Prefill wall clock
    prefill_wall_ns = first_graph_start - all_events[0][0]
    ttft_ms = prefill_wall_ns / 1e6

    # Count decode steps for averaging
    n_decode_steps = len(consistent_steps)

    print(f"  === Decode (per step) ===")
    print(f"  encoder (CUDA Graph): {encoder_ns/1e6:.4f} ms ({len(graph_evts)} kernels)")
    print(f"  lm_head:  {lm_head_ns/1e6:.4f} ms")
    print(f"  sampling: {sampling_ns/1e6:.4f} ms")
    print(f"  overhead: {overhead_ns/1e6:.4f} ms")
    print(f"  TPOT (step wall): {tpot_ms:.4f} ms")
    print(f"  === Prefill ===")
    print(f"  prefill kernels: {prefill_count}, kernel time: {prefill_ns/1e6:.2f} ms")
    print(f"  TTFT (wall): {ttft_ms:.2f} ms")
    print(f"  === Summary ===")
    print(f"  decode steps: {n_decode_steps}")

    return {
        "lm_head_ms": round(lm_head_ns / 1e6, 4),
        "sampling_ms": round(sampling_ns / 1e6, 4),
        "encoder_ms": round(encoder_ns / 1e6, 4),
        "overhead_ms": round(overhead_ns / 1e6, 4),
        "tpot_ms": round(tpot_ms, 4),
        "ttft_ms": round(ttft_ms, 2),
        "prefill_kernel_ms": round(prefill_ns / 1e6, 2),
        "n_decode_steps": n_decode_steps,
        "tail_per_step_ms": round((lm_head_ns + sampling_ns) / 1e6, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PPU compensation")
    ap.add_argument("--bench-results", default=None,
                    help="bench.json (optional if --asys-sqlite provided, reads TTFT/TPOT from trace)")
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

    # Load bench results (optional — can read from trace instead)
    bench = None
    if args.bench_results and os.path.exists(args.bench_results):
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

    if bench:
        results_list = bench.get("results", [bench])
    else:
        results_list = []

    # Auto-read output_len
    output_len = args.output_len
    if output_len is None and bench:
        output_len = bench.get("output_len") or next(
            (r.get("output_len") or r.get("output_tokens") for r in results_list if "error" not in r), 64)
    if output_len is None:
        output_len = 64

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

    # If no bench results, create one entry from trace data
    if not results_list and use_trace_encoder:
        trace_ttft = tail.get("ttft_ms", 0)
        trace_tpot = tail.get("tpot_ms", 0)
        print(f"  (No bench.json — using trace: TTFT={trace_ttft:.2f}ms, TPOT={trace_tpot:.4f}ms)")
        results_list = [{
            "input_len": 0,  # unknown from trace
            "ttft_median_ms": trace_ttft,
            "tpot_median_ms": trace_tpot,
            "output_tokens": output_len,
        }]

    compensated = []
    for r in results_list:
        if "error" in r:
            compensated.append(r); continue

        il = r.get("input_len", 0)
        raw_ttft = r.get("ttft_median_ms", tail.get("ttft_ms", 0))
        raw_tpot = r.get("tpot_median_ms", tail.get("tpot_ms", 0))
        output_tokens = r.get("output_tokens", output_len)

        if use_trace_encoder:
            # Precise: encoder from trace, overhead not scaled
            comp_tpot = encoder_ms * layer_scale + lm_head_final + sampling_ms + overhead_ms + decode_comm
            # TTFT: prefill kernel time from trace, scale encoder portion
            trace_ttft = tail.get("ttft_ms", raw_ttft)
            prefill_encoder = max(trace_ttft - lm_head_ms - sampling_ms - overhead_ms, 0)
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
