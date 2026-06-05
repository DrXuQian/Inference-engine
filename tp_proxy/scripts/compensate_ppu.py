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
# NVTX helpers
# ---------------------------------------------------------------------------

def _find_nvtx_table(sqlite_path: str):
    """Return (table, start_col, end_col, text_col) or None."""
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()
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
        conn.close(); return None

    c.execute(f'SELECT * FROM "{nvtx_table}" LIMIT 1')
    cols = [d[0] for d in c.description]
    start_col = next((c for c in cols if c.lower() in ('start', 'timestamp', 'starttime')), None)
    end_col = next((c for c in cols if c.lower() in ('end', 'endtime', 'endtimestamp')), None)
    text_col = next((c for c in cols if c.lower() in ('text', 'message', 'name', 'value')), None)
    conn.close()

    if not start_col or not end_col or not text_col:
        return None
    return nvtx_table, start_col, end_col, text_col


def _load_kernel_events(sqlite_path: str) -> list | None:
    """Load all kernel events sorted by start time.

    Returns list of (start, dur, end, name, graphNodeId).
    """
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in c.fetchall()]
    kt = [t for t in tables if 'KERNEL' in t.upper() and 'ACTIVITY' in t.upper()]
    if not kt:
        conn.close(); return None
    kt = kt[0]

    c.execute(f'PRAGMA table_info("{kt}")')
    cols = [col[1] for col in c.fetchall()]
    has_gid = "graphNodeId" in cols

    if has_gid:
        c.execute(f'''
            SELECT k.start, k."end" - k.start AS dur, k."end",
                   s.value AS name, k.graphNodeId
            FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start
        ''')
        evts = [(r[0], r[1], r[2], r[3], r[4] or 0) for r in c.fetchall()]
    else:
        c.execute(f'''
            SELECT k.start, k."end" - k.start AS dur, k."end",
                   s.value AS name
            FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
            ORDER BY k.start
        ''')
        evts = [(r[0], r[1], r[2], r[3], 0) for r in c.fetchall()]
    conn.close()
    return evts


# ---------------------------------------------------------------------------
# a) Decode step from NVTX marker boundary (preferred)
# ---------------------------------------------------------------------------

def extract_decode_from_nvtx(sqlite_path: str,
                             lm_head_kernel: str) -> dict | None:
    """Extract decode step timing using NVTX decode_0 marker boundary.

    Finds the last complete decode step within the 'decode_0' NVTX range:
    - The last CUDA Graph (encoder) ends at the last lm_head kernel
    - Tail = last lm_head + sampling kernels from lm_head to NVTX end

    The NVTX boundary prevents post-inference cleanup from contaminating
    the measurement.
    """
    import re as _re

    info = _find_nvtx_table(sqlite_path)
    if not info:
        return None
    nvtx_table, start_col, end_col, text_col = info

    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    # Find decode_0 outer marker
    c.execute(f'SELECT "{start_col}", "{end_col}" FROM "{nvtx_table}" '
              f'WHERE "{text_col}" = \'decode_0\'')
    row = c.fetchone()
    if not row:
        conn.close()
        return None
    nvtx_start, nvtx_end = row

    # Find prefill marker for TTFT
    c.execute(f'SELECT "{start_col}", "{end_col}" FROM "{nvtx_table}" '
              f'WHERE "{text_col}" LIKE \'%prefill%\'')
    prefill_rows = c.fetchall()
    conn.close()

    all_evts = _load_kernel_events(sqlite_path)
    if not all_evts:
        return None

    print(f"  Total kernels: {len(all_evts)}")

    # Prefill kernel time (from standalone prefill NVTX marker)
    ttft_ms = 0.0
    ttft_wall_ms = 0.0
    if prefill_rows:
        pf_start, pf_end = prefill_rows[-1]
        pf_kernels = [e for e in all_evts if e[0] >= pf_start and e[2] <= pf_end]
        ttft_ms = sum(e[1] for e in pf_kernels) / 1e6
        ttft_wall_ms = (pf_end - pf_start) / 1e6
        print(f"  Prefill (NVTX, {len(pf_kernels)} kernels): "
              f"kernel={ttft_ms:.2f}ms, wall={ttft_wall_ms:.2f}ms")

    # Kernels within decode_0 NVTX range
    dec_evts = [e for e in all_evts if e[0] >= nvtx_start and e[2] <= nvtx_end]
    print(f"  decode_0 NVTX: {len(dec_evts)} kernels, "
          f"wall={(nvtx_end - nvtx_start) / 1e6:.2f}ms")

    # Find the last CUDA graph via graphNodeId > 0
    n_graph = sum(1 for e in dec_evts if e[4] > 0)
    n_nongraph = len(dec_evts) - n_graph
    print(f"  graph kernels: {n_graph}, non-graph: {n_nongraph}")

    # Find the last kernel with graphNodeId > 0 = end of last CUDA graph
    last_graph_idx = None
    for i in range(len(dec_evts) - 1, -1, -1):
        if dec_evts[i][4] > 0:
            last_graph_idx = i
            break

    if last_graph_idx is None:
        print("  WARNING: no graphNodeId > 0 kernels in decode_0")
        return None

    # Find start of the last CUDA graph: walk backwards from last_graph_idx
    # until we hit a non-graph kernel (graphNodeId == 0)
    first_graph_idx = last_graph_idx
    for i in range(last_graph_idx - 1, -1, -1):
        if dec_evts[i][4] > 0:
            first_graph_idx = i
        else:
            break

    enc_evts = dec_evts[first_graph_idx:last_graph_idx + 1]
    tail_evts = dec_evts[last_graph_idx + 1:]
    print(f"  Last CUDA graph: kernels [{first_graph_idx}..{last_graph_idx}] "
          f"({len(enc_evts)} kernels)")
    print(f"  Tail (after graph): {len(tail_evts)} kernels")

    # Split tail into lm_head + sampling
    lm_dur = 0
    samp_evts = []
    for e in tail_evts:
        if lm_head_kernel in e[3]:
            lm_dur += e[1]
        else:
            samp_evts.append(e)

    encoder_ns = sum(e[1] for e in enc_evts)
    sampling_ns = sum(e[1] for e in samp_evts)

    # Step wall: from encoder start to last tail kernel end
    if tail_evts:
        step_wall = tail_evts[-1][2] - enc_evts[0][0]
    else:
        step_wall = enc_evts[-1][2] - enc_evts[0][0]

    # Kernel breakdown
    gemm_re = _re.compile(
        r"deep_gemm|GemmKernel|gemm_ktype|cutlass|cublas|cublasLt|"
        r"marlin|moe_wna16|moe.*gemm|xmma|batched_gemvt", _re.IGNORECASE)
    fa_re = _re.compile(
        r"flash_fwd|flash_bwd|fmha|FlashAttn", _re.IGNORECASE)

    enc_gemm = enc_fa = enc_other = 0
    for e in enc_evts:
        name, dur = e[3], e[1]
        if fa_re.search(name):
            enc_fa += dur
        elif gemm_re.search(name):
            enc_gemm += dur
        else:
            enc_other += dur

    enc_total = enc_gemm + enc_fa + enc_other
    kernel_breakdown = {
        "enc_gemm_ms": round(enc_gemm / 1e6, 4),
        "enc_fa_ms": round(enc_fa / 1e6, 4),
        "enc_other_ms": round(enc_other / 1e6, 4),
        "enc_gemm_frac": round(enc_gemm / enc_total, 4) if enc_total > 0 else 0,
        "tail_lmhead_ms": round(lm_dur / 1e6, 4),
        "tail_other_ms": round(sampling_ns / 1e6, 4),
    }

    encoder_ms_v = encoder_ns / 1e6
    lm_head_ms_v = lm_dur / 1e6
    sampling_ms_v = sampling_ns / 1e6
    tpot_ms_v = step_wall / 1e6

    gf = kernel_breakdown["enc_gemm_frac"]
    print(f"\n  Last decode step (NVTX + graphNodeId):")
    print(f"    encoder (CUDA graph): {encoder_ms_v:.4f} ms ({len(enc_evts)} kernels)")
    print(f"    lm_head:              {lm_head_ms_v:.4f} ms")
    print(f"    sampling:             {sampling_ms_v:.4f} ms ({len(samp_evts)} kernels)")
    print(f"    step wall:            {tpot_ms_v:.4f} ms")
    print(f"  Encoder breakdown:")
    print(f"    gemm (INT4-able):     {kernel_breakdown['enc_gemm_ms']:.4f} ms ({gf*100:.0f}%)")
    print(f"    flash_attn:           {kernel_breakdown['enc_fa_ms']:.4f} ms")
    print(f"    other (norm etc):     {kernel_breakdown['enc_other_ms']:.4f} ms")
    print(f"  Tail breakdown:")
    print(f"    lm_head (BF16):       {kernel_breakdown['tail_lmhead_ms']:.4f} ms")
    print(f"    sampling:             {kernel_breakdown['tail_other_ms']:.4f} ms")
    if ttft_ms > 0:
        print(f"  TTFT (prefill):         {ttft_ms:.2f} ms")

    return {
        "encoder_ms": round(encoder_ms_v, 4),
        "lm_head_ms": round(lm_head_ms_v, 4),
        "sampling_ms": round(sampling_ms_v, 4),
        "tpot_ms": round(tpot_ms_v, 4),
        "ttft_kernel_ms": round(ttft_ms, 2),
        "ttft_wall_ms": round(ttft_wall_ms, 2),
        "ttft_ms": round(ttft_ms, 2),
        "overhead_ms": 0,
        "tail_per_step_ms": round(lm_head_ms_v + sampling_ms_v, 4),
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
        tail = extract_decode_from_nvtx(args.asys_sqlite, args.lm_head_kernel)
        if not tail:
            print("ERROR: NVTX decode extraction failed. "
                  "Ensure trace has 'decode_0' NVTX marker and lm_head kernels.")
            sys.exit(1)
        print(f"  lm_head: {tail['lm_head_ms']:.4f} ms, "
              f"sampling: {tail['sampling_ms']:.4f} ms")
    else:
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
        b1_tail = extract_decode_from_nvtx(args.sampling_trace, "gemvt_op")
        if not b1_tail:
            print("ERROR: NVTX decode extraction failed for sampling trace.")
            sys.exit(1)
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
