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
# Prefill from NVTX marker
# ---------------------------------------------------------------------------

def extract_prefill_from_nvtx(sqlite_path: str, kernel_evts: list) -> dict | None:
    """Extract prefill time from NVTX/HGTX 'prefill' marker.

    Finds the 'prefill' marker range, then sums kernel durations
    that fall within that time range.

    Returns {kernel_ms, wall_ms, n_kernels} or None.
    """
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    # Try HGTX_EVENTS (PPU) or NVTX (nsys)
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]

    nvtx_table = None
    for t in tables:
        if 'HGTX' in t.upper() and 'EVENT' in t.upper():
            nvtx_table = t; break
    if not nvtx_table:
        for t in tables:
            if 'NVTX' in t.upper() and ('PUSH' in t.upper() or 'EVENT' in t.upper()):
                nvtx_table = t; break
    if not nvtx_table:
        conn.close()
        return None

    # Find 'prefill' marker — look for text containing 'prefill'
    try:
        c.execute(f'SELECT * FROM "{nvtx_table}" LIMIT 1')
        cols = [desc[0] for desc in c.description]
    except Exception:
        conn.close()
        return None

    # Determine column names (varies by platform)
    start_col = next((c for c in cols if c.lower() in ('start', 'timestamp', 'starttime')), None)
    end_col = next((c for c in cols if c.lower() in ('end', 'endtime', 'endtimestamp')), None)
    text_col = next((c for c in cols if c.lower() in ('text', 'message', 'name', 'value')), None)

    if not start_col or not text_col:
        conn.close()
        return None

    if end_col:
        c.execute(f'SELECT "{start_col}", "{end_col}", "{text_col}" FROM "{nvtx_table}" '
                  f'WHERE "{text_col}" LIKE "%prefill%"')
    else:
        # Some formats store duration instead of end
        dur_col = next((c for c in cols if c.lower() in ('duration', 'dur')), None)
        if dur_col:
            c.execute(f'SELECT "{start_col}", "{start_col}" + "{dur_col}", "{text_col}" '
                      f'FROM "{nvtx_table}" WHERE "{text_col}" LIKE "%prefill%"')
        else:
            conn.close()
            return None

    rows = c.fetchall()
    conn.close()

    if not rows:
        return None

    # Use the LAST prefill marker (skip warmup ones if any)
    pf_start, pf_end, pf_text = rows[-1]
    if pf_end <= pf_start:
        return None

    # Count kernels within the prefill time range
    kernel_dur_ns = 0
    n_kernels = 0
    for evt_start, evt_dur, evt_end, evt_name, evt_gid in kernel_evts:
        if evt_start >= pf_start and evt_end <= pf_end:
            kernel_dur_ns += evt_dur
            n_kernels += 1

    wall_ns = pf_end - pf_start

    return {
        "kernel_ms": kernel_dur_ns / 1e6,
        "wall_ms": wall_ns / 1e6,
        "n_kernels": n_kernels,
    }


# ---------------------------------------------------------------------------
# a) Tail from trace (lm_head + sampling per step)
# ---------------------------------------------------------------------------

def measure_tail_from_trace(sqlite_path: str,
                            lm_head_kernel: str) -> dict | None:
    """One decode step = CUDA Graph (encoder) + gap to next CUDA Graph (tail).

    encoder = graph_end - graph_start  (wall clock)
    tail    = graph_end → next gap_end (wall clock, = lm_head + sampling)
    tpot    = encoder + tail
    """
    from collections import Counter

    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    kt = [r[0] for r in c.fetchall() if 'KERNEL' in r[0].upper() and 'ACTIVITY' in r[0].upper()]
    if not kt:
        conn.close(); return None
    kt = kt[0]

    # Check graphNodeId
    c.execute(f'PRAGMA table_info("{kt}")')
    if "graphNodeId" not in [col[1] for col in c.fetchall()]:
        conn.close()
        print("  WARNING: no graphNodeId column")
        return None

    c.execute(f'''
        SELECT k.start, k."end" - k.start AS dur, k."end",
               s.value AS name, k.graphNodeId
        FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    ''')
    evts = [(r[0], r[1], r[2], r[3], r[4] or 0) for r in c.fetchall()]
    conn.close()

    n_g = sum(1 for *_, g in evts if g > 0)
    print(f"  kernels: {len(evts)} ({n_g} graph, {len(evts)-n_g} non-graph)")

    # Split into graph/gap segments
    segs = []
    ct = "graph" if evts[0][4] > 0 else "gap"
    ce = [evts[0]]
    for e in evts[1:]:
        t = "graph" if e[4] > 0 else "gap"
        if t == ct:
            ce.append(e)
        else:
            segs.append((ct, ce))
            ct, ce = t, [e]
    segs.append((ct, ce))

    # Each decode step: graph_seg → gap_seg
    # encoder = graph wall, tail = graph_end to gap_end
    steps = []
    for i in range(len(segs) - 1):
        if segs[i][0] == "graph" and segs[i+1][0] == "gap":
            g = segs[i][1]
            gap = segs[i+1][1]
            g_start, g_end = g[0][0], g[-1][2]
            gap_end = gap[-1][2]
            lm_dur = 0
            for ev in gap:
                if lm_head_kernel in ev[3]:
                    lm_dur = ev[1]; break
            steps.append({
                "encoder_wall": g_end - g_start,
                "tail_wall": gap_end - g_end,
                "step_wall": gap_end - g_start,
                "lm_head_dur": lm_dur,
                "graph_kernels": len(g),
                "_graph_evts": g,
                "_gap_evts": gap,
            })

    if not steps:
        print("  WARNING: no decode steps found")
        return None
    print(f"  decode steps: {len(steps)}")

    # Filter: consistent graph kernel count
    counts = Counter(s["graph_kernels"] for s in steps)
    mode = counts.most_common(1)[0][0]
    filtered = [s for s in steps if s["graph_kernels"] == mode]
    print(f"  graph kernel mode: {mode} ({len(filtered)}/{len(steps)} steps)")

    # Filter: has matching lm_head, pick top 25% by duration (= largest batch)
    matched = [s for s in filtered if s["lm_head_dur"] > 0]
    if matched:
        matched.sort(key=lambda s: s["lm_head_dur"])
        thr = matched[len(matched) * 3 // 4]["lm_head_dur"] if len(matched) > 4 else matched[0]["lm_head_dur"]
        top = [s for s in matched if s["lm_head_dur"] >= thr]
        print(f"  lm_head: {matched[0]['lm_head_dur']/1e3:.1f}-{matched[-1]['lm_head_dur']/1e3:.1f}us, "
              f"p75={thr/1e3:.1f}us, selected={len(top)}")
        selected = top
    else:
        print(f"  WARNING: '{lm_head_kernel}' not found in gaps")
        selected = filtered

    s = selected[len(selected) // 2]
    encoder_ms = s["encoder_wall"] / 1e6
    tail_ms = s["tail_wall"] / 1e6
    lm_head_ms = s["lm_head_dur"] / 1e6
    sampling_ms_wall = max(tail_ms - lm_head_ms, 0)
    tpot_ms = s["step_wall"] / 1e6

    # Prefill: extract from NVTX "prefill" marker if available,
    # otherwise fall back to gap-before-first-decode heuristic.
    ttft_ms = 0
    ttft_wall_ms = 0
    prefill_from_nvtx = extract_prefill_from_nvtx(sqlite_path, evts)
    if prefill_from_nvtx:
        ttft_ms = prefill_from_nvtx["kernel_ms"]
        ttft_wall_ms = prefill_from_nvtx["wall_ms"]
        print(f"\n  Prefill (from NVTX 'prefill' marker, {prefill_from_nvtx['n_kernels']} kernels):")
        print(f"    kernel time: {ttft_ms:.2f} ms")
        print(f"    wall time:   {ttft_wall_ms:.2f} ms")
    else:
        # Fallback: find the last gap before first mode-count graph
        first_mode_seg_idx = None
        for si, (stype, sevts) in enumerate(segs):
            if stype == "graph" and len(sevts) == mode:
                first_mode_seg_idx = si
                break

        if first_mode_seg_idx and first_mode_seg_idx > 0:
            prev_seg = segs[first_mode_seg_idx - 1]
            if prev_seg[0] == "gap":
                prefill_kernel_ns = sum(d for _, d, _, _, _ in prev_seg[1])
                prefill_wall_ns = prev_seg[1][-1][2] - prev_seg[1][0][0]
                ttft_ms = prefill_kernel_ns / 1e6
                ttft_wall_ms = prefill_wall_ns / 1e6
                print(f"\n  Prefill (fallback: gap before decode, {len(prev_seg[1])} kernels):")
                print(f"    kernel time: {ttft_ms:.2f} ms")
                print(f"    wall time:   {ttft_wall_ms:.2f} ms")
            else:
                print(f"\n  WARNING: no gap before first decode graph")
        else:
            print(f"\n  WARNING: no mode-count graph found for prefill")

    # Kernel classification for the SELECTED decode step (same as s above)
    import re as _re
    gemm_re = _re.compile(r"deep_gemm|GemmKernel|gemm_ktype|cutlass|cublas|cublasLt|marlin|moe_wna16|moe.*gemm|xmma|batched_gemvt", _re.IGNORECASE)
    fa_re = _re.compile(r"flash_fwd|flash_bwd|fmha|FlashAttn", _re.IGNORECASE)

    graph_kernels = s["_graph_evts"]
    gap_kernels = s["_gap_evts"]

    # Classify encoder (graph) kernels
    enc_gemm_ns = enc_fa_ns = enc_other_ns = 0
    for evt in graph_kernels:
        name, dur = evt[3], evt[1]
        if fa_re.search(name):
            enc_fa_ns += dur
        elif gemm_re.search(name):
            enc_gemm_ns += dur
        else:
            enc_other_ns += dur

    # Classify gap (tail) kernels — use lm_head_kernel from CLI (not hardcoded regex)
    gap_lmhead_ns = gap_other_ns = 0
    for evt in gap_kernels:
        name, dur = evt[3], evt[1]
        if lm_head_kernel in name:
            gap_lmhead_ns += dur
        else:
            gap_other_ns += dur

    enc_total = enc_gemm_ns + enc_fa_ns + enc_other_ns
    kernel_breakdown = {
        # Encoder (graph): these get ×layer_scale
        "enc_gemm_ms": round(enc_gemm_ns / 1e6, 4),   # INT4-able ×0.25
        "enc_fa_ms": round(enc_fa_ns / 1e6, 4),        # KV cache, BW-bound
        "enc_other_ms": round(enc_other_ns / 1e6, 4),   # compute-bound
        "enc_gemm_frac": round(enc_gemm_ns / enc_total, 4) if enc_total > 0 else 0,
        # Tail (gap): NO layer_scale, lm_head /tp
        "tail_lmhead_ms": round(gap_lmhead_ns / 1e6, 4),  # BF16, /tp
        "tail_other_ms": round(gap_other_ns / 1e6, 4),     # sampling etc
    }

    # Use kernel duration for sampling (not wall clock) to avoid
    # inflated values from CPU scheduling, post-inference cleanup, etc.
    sampling_ms_kernel = kernel_breakdown["tail_other_ms"]
    if sampling_ms_wall > 10 * max(sampling_ms_kernel, 0.01):
        print(f"\n  WARNING: tail wall={sampling_ms_wall:.2f}ms >> kernel={sampling_ms_kernel:.4f}ms"
              f" (wall includes CPU overhead / cleanup, using kernel time)")
    sampling_ms = sampling_ms_kernel

    gf = kernel_breakdown["enc_gemm_frac"]
    print(f"\n  encoder (graph wall): {encoder_ms:.4f} ms")
    print(f"  tail wall clock:      {tail_ms:.4f} ms")
    print(f"  tail kernel time:     {(sampling_ms + lm_head_ms):.4f} ms")
    print(f"  TPOT (step wall):     {tpot_ms:.4f} ms")
    print(f"  TTFT (prefill):       {ttft_ms:.2f} ms")
    print(f"  Encoder kernel breakdown (×layer_scale):")
    print(f"    gemm (INT4-able):     {kernel_breakdown['enc_gemm_ms']:.4f} ms ({gf*100:.0f}%)")
    print(f"    flash_attn:           {kernel_breakdown['enc_fa_ms']:.4f} ms")
    print(f"    other:                {kernel_breakdown['enc_other_ms']:.4f} ms")
    print(f"  Tail breakdown (no layer_scale):")
    print(f"    lm_head (BF16):       {kernel_breakdown['tail_lmhead_ms']:.4f} ms")
    print(f"    sampling (kernel):    {sampling_ms:.4f} ms")

    return {
        "encoder_ms": round(encoder_ms, 4),
        "lm_head_ms": round(lm_head_ms, 4),
        "sampling_ms": round(sampling_ms, 4),
        "tpot_ms": round(tpot_ms, 4),
        "ttft_kernel_ms": round(ttft_ms, 2),
        "ttft_wall_ms": round(ttft_wall_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "overhead_ms": 0,
        "tail_per_step_ms": round(tail_ms, 4),
        "kernel_breakdown": kernel_breakdown,
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
        # Prefer kernel time from trace (if available), fallback to wall clock
        decode_comm = comm.get("decode_total_kernel_ms",
                               comm.get("decode_total_per_step_ms",
                                        comm.get("total_per_step_ms", 0)))
        prefill_comm = comm.get("prefill_total_kernel_ms",
                                comm.get("prefill_total_per_step_ms", decode_comm))
        comm["decode_comm_ms"] = decode_comm
        comm["prefill_comm_ms"] = prefill_comm
        source = "kernel" if "decode_total_kernel_ms" in comm else "wall"
        print(f"=== a) Communication (from {args.comm_json}, {source}) ===")
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
        output_tokens = r.get("output_tokens", output_len)

        if use_trace_encoder:
            # All from trace — ignore bench.json values
            raw_tpot = tail.get("tpot_ms", 0)
            raw_ttft = tail.get("ttft_ms", 0)

            # comp_TPOT = encoder × scale + lm_head/tp + sampling + comm
            comp_tpot = encoder_ms * layer_scale + lm_head_final + sampling_ms + decode_comm
            # comp_TTFT = prefill_encoder × scale + lm_head/tp + sampling + comm
            prefill_encoder = max(raw_ttft - lm_head_ms - sampling_ms, 0)
            comp_ttft = prefill_encoder * layer_scale + lm_head_final + sampling_ms + prefill_comm
        else:
            raw_ttft = r.get("ttft_median_ms", 0)
            raw_tpot = r.get("tpot_median_ms", 0)
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
