#!/usr/bin/env python3
"""
One-script compensation for PPU platform.

Combines:
  a) LM head (torch.mm benchmark, TP>1 only)
  b) Communication (pccl_tools standalone or asys trace)
  c) Encoder block (asys trace, if layers pruned)

Usage:
    # Full compensation with asys trace
    python compensate_ppu.py \
        --bench-results bench.json \
        --model-dir /path/to/model \
        --asys-sqlite trace.sqlite \
        --output-len 64

    # Without asys trace (pccl_tools for comm, no encoder comp)
    python compensate_ppu.py \
        --bench-results bench.json \
        --model-dir /path/to/model \
        --output-len 64

    # Override auto-detected params
    python compensate_ppu.py \
        --bench-results bench.json \
        --model-dir /path/to/model \
        --asys-sqlite trace.sqlite \
        --output-len 64 \
        --pruned-layers 10 --original-layers 40 --tp-size 2 \
        --lm-head-kernel gemvt_op --sampling-kernels ArgMaxOps
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_model_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    num_heads = tc.get("num_attention_heads", 0)
    head_dim = tc.get("head_dim", 0)
    if not head_dim and num_heads:
        head_dim = tc["hidden_size"] // num_heads
    # Attention cycle: e.g. [lin, lin, lin, full] → 3/4 are linear
    attn_pattern = tc.get("interleave_attn_pattern", None)
    if attn_pattern:
        n_lin = sum(1 for x in attn_pattern if x != "full")
        lin_ratio = n_lin / len(attn_pattern)
    else:
        lin_ratio = 0.0  # no linear attention layers
    return {
        "hidden_size": tc["hidden_size"],
        "vocab_size": tc["vocab_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": num_heads,
        "head_dim": head_dim,
        "lin_attn_ratio": lin_ratio,
    }


def load_meta(model_dir: str) -> dict | None:
    for d in [model_dir, os.path.dirname(model_dir)]:
        p = os.path.join(d, "split_meta.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


# ---------------------------------------------------------------------------
# a) LM head
# ---------------------------------------------------------------------------

def measure_lm_head(hidden: int, vocab: int, tp_size: int,
                    input_lens: list[int],
                    num_heads: int = 0, head_dim: int = 0,
                    lin_attn_ratio: float = 0.0) -> dict:
    """Measure lm_head delta and estimate linear attention delta by scaling.

    lm_head gemvt:   [1, hidden] × [hidden, vocab]
    lin attn gemvt:  [1, hidden] × [hidden, num_heads * head_dim]
    scale = num_heads * head_dim / vocab
    """
    if tp_size <= 1:
        return {"decode_delta_ms": 0,
                "prefill_deltas_ms": {str(il): 0 for il in input_lens},
                "lin_attn_decode_delta_per_layer_ms": 0,
                "lin_attn_prefill_deltas_per_layer_ms": {str(il): 0 for il in input_lens}}

    import torch
    device, dtype = "cuda", torch.bfloat16
    vocab_half = vocab // tp_size

    def bench_mm(M, K, N, warmup=50, iters=200):
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        for _ in range(warmup): torch.mm(A, B)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); torch.mm(A, B); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]

    decode_delta = bench_mm(1, hidden, vocab) - bench_mm(1, hidden, vocab_half)
    prefill_deltas = {}
    for il in input_lens:
        prefill_deltas[il] = bench_mm(il, hidden, vocab) - bench_mm(il, hidden, vocab_half)

    # Linear attention: scale down from lm_head delta
    # lm_head output dim = vocab, lin attn output dim = num_heads * head_dim
    lin_attn_dim = num_heads * head_dim
    scale = lin_attn_dim / vocab if vocab > 0 and lin_attn_dim > 0 else 0
    lin_decode_delta = decode_delta * scale
    lin_prefill_deltas = {il: d * scale for il, d in prefill_deltas.items()}

    return {"decode_delta_ms": round(decode_delta, 4),
            "prefill_deltas_ms": {str(k): round(v, 4) for k, v in prefill_deltas.items()},
            "lin_attn_decode_delta_per_layer_ms": round(lin_decode_delta, 6),
            "lin_attn_prefill_deltas_per_layer_ms": {str(k): round(v, 6) for k, v in lin_prefill_deltas.items()},
            "lin_attn_scale": round(scale, 6)}


# ---------------------------------------------------------------------------
# b) Communication
# ---------------------------------------------------------------------------

def measure_comm_pccl(hidden: int, vocab: int, num_layers: int,
                      tp_size: int, ar_tool: str, ag_tool: str) -> dict:
    ar_size = hidden * 2
    ag_size = (vocab // tp_size) * 2

    def run(tool, size, iters=200, warmup=50):
        cmd = [tool, "-b", str(size), "-e", str(size), "-f", "2",
               "-d", "bf16", "-o", "sum", "-n", str(iters), "-w", str(warmup),
               "-g", str(tp_size), "-c", "0", "-a", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in r.stdout.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="): continue
            parts = line.split()
            if len(parts) >= 8:
                try: return float(parts[5])
                except: pass
        return 0.0

    ar_us = run(ar_tool, ar_size)
    ag_us = run(ag_tool, ag_size)
    n_ar = num_layers * 2
    n_ag = 2
    return {"method": "pccl_standalone",
            "ar_us": round(ar_us, 1), "ag_us": round(ag_us, 1),
            "n_ar": n_ar, "n_ag": n_ag,
            "total_per_step_ms": round((n_ar * ar_us + n_ag * ag_us) / 1000, 3)}


def measure_comm_trace(sqlite_path: str, num_layers: int) -> dict:
    """Extract comm kernel time from asys sqlite."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Detect table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    kt = ""
    for r in cursor.fetchall():
        if "KERNEL" in r[0] and "ACTIVITY" in r[0]:
            kt = r[0]; break
    if not kt:
        return {"method": "trace", "total_per_step_ms": 0, "error": "no kernel table"}

    cursor.execute(f"""
        SELECT k.start, k."end" - k.start AS dur, s.value AS name
        FROM {kt} k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """)
    events = cursor.fetchall()
    conn.close()

    if not events:
        return {"method": "trace", "total_per_step_ms": 0}

    # Find serving phase
    t_min = events[0][0]
    bins = defaultdict(int)
    for s, d, n in events:
        bins[(s - t_min) // 1_000_000_000] += 1
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
    serve_start = t_min + (best_start + best_len) * 1_000_000_000

    comm_kw = ["allreduce", "all_reduce", "cross_device_reduce",
               "allgather", "all_gather", "pccl", "nccl"]
    comm_ns = 0; comm_count = 0; serve_count = 0
    for s, d, n in events:
        if s < serve_start: continue
        serve_count += 1
        if any(kw in n.lower() for kw in comm_kw):
            comm_ns += d; comm_count += 1

    # Estimate steps
    serve_events = [(s, s + d) for s, d, n in events if s >= serve_start]
    serve_events.sort()
    n_bursts = 1
    for i in range(1, len(serve_events)):
        if serve_events[i][0] - serve_events[i - 1][1] > 500_000:
            n_bursts += 1

    per_step = (comm_ns / n_bursts) / 1e6 if n_bursts > 0 else 0
    return {"method": "trace", "comm_count": comm_count,
            "total_per_step_ms": round(per_step, 3)}


# ---------------------------------------------------------------------------
# c) Encoder block from trace
# ---------------------------------------------------------------------------

def measure_tail_from_trace(sqlite_path: str,
                            lm_head_kernel: str) -> dict | None:
    """Extract tail time (lm_head + sampling) per decode step from trace.

    Simple approach:
      1. From the END of the trace, walk backward to find the last lm_head kernel
      2. Sum kernel durations from that kernel to the end = tail_per_step
      3. This tail is subtracted from raw TPOT/TTFT, then encoder is scaled up

    The lm_head kernel is matched by SUBSTRING (e.g. "gemvt_op" matches
    "void gemvt_op<...>").
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
        # Debug: print last 10 kernel names to help diagnose
        print("  WARNING: lm_head kernel not found by substring match")
        print(f"  Looking for: '{lm_head_kernel}'")
        print(f"  Last 10 kernels:")
        for ev in all_events[-10:]:
            print(f"    {ev[3][:80]}  dur={ev[1]/1e3:.1f}us")
        return None

    # Tail = sum of kernel durations from lm_head to end
    tail_kernels = all_events[tail_start_idx:]
    tail_kernel_ns = sum(d for _, d, _, _ in tail_kernels)
    # Also measure wall clock span
    tail_wall_ns = tail_kernels[-1][2] - tail_kernels[0][0]

    print(f"  Last lm_head kernel: '{all_events[tail_start_idx][3][:60]}'")
    print(f"  Tail kernels: {len(tail_kernels)}")
    print(f"  Tail kernel time: {tail_kernel_ns / 1e6:.4f} ms")
    print(f"  Tail wall time:   {tail_wall_ns / 1e6:.4f} ms")

    return {
        "tail_per_step_ms": round(tail_kernel_ns / 1e6, 4),
        "tail_wall_ms": round(tail_wall_ns / 1e6, 4),
        "tail_kernel_count": len(tail_kernels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PPU compensation (all-in-one)")
    ap.add_argument("--bench-results", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-len", type=int, default=None,
                    help="Override output_len (auto-read from bench_results if omitted)")
    ap.add_argument("--comm-json", default=None,
                    help="comm.json from comm_bench.sh (AR/AG latency)")
    ap.add_argument("--asys-sqlite", default=None,
                    help="asys trace sqlite (for encoder block compensation)")
    ap.add_argument("--pruned-layers", type=int, default=None)
    ap.add_argument("--original-layers", type=int, default=None)
    ap.add_argument("--tp-size", type=int, default=None)
    ap.add_argument("--lm-head-kernel", default="gemvt_op")
    ap.add_argument("--sampling-kernels", default="ArgMaxOps,DeviceRadixSortHistogramKernel,DeviceRadixSortExclusiveSumKernel,DeviceRadixSortOnesweepKernel,cunn_SoftMaxForward,DeviceScanInitKernel,DeviceScanKernel")
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

    results_list = bench.get("results", [bench])
    input_lens = [r["input_len"] for r in results_list if "error" not in r]

    # Auto-read output_len from bench results
    output_len = args.output_len
    if output_len is None:
        output_len = bench.get("output_len") or next(
            (r.get("output_len") or r.get("output_tokens") for r in results_list if "error" not in r), 64)
    print(f"Output len: {output_len}")

    sampling_names = set(k.strip() for k in args.sampling_kernels.split(","))

    num_heads = cfg["num_attention_heads"]
    head_dim_val = cfg["head_dim"]
    lin_ratio = cfg["lin_attn_ratio"]

    print(f"Model: hidden={hidden}, vocab={vocab}, heads={num_heads}, "
          f"head_dim={head_dim_val}, lin_ratio={lin_ratio:.2f}")
    print(f"Layers: {pruned} pruned / {original} original, TP={tp_size}")
    print()

    # === a) LM head + linear attention (torch.mm, scaled) ===
    if tp_size > 1:
        print("=== a) LM head + linear attention (torch.mm) ===")
        lm = measure_lm_head(hidden, vocab, tp_size, input_lens,
                             num_heads, head_dim_val, lin_ratio)
        print(f"  LM head decode delta: {lm['decode_delta_ms']:.3f} ms")
        for il, d in lm["prefill_deltas_ms"].items():
            print(f"  LM head prefill delta (input={il}): {d:.3f} ms")
        print(f"  Linear attn scale: {lm['lin_attn_scale']:.6f} "
              f"({num_heads}×{head_dim_val}/{vocab})")
        print(f"  Linear attn decode delta/layer: {lm['lin_attn_decode_delta_per_layer_ms']:.6f} ms")
        n_lin_layers = round(original * lin_ratio)
        print(f"  Linear attn layers: {n_lin_layers}/{original}")
        print(f"  Total lin attn decode delta: "
              f"{n_lin_layers * lm['lin_attn_decode_delta_per_layer_ms']:.3f} ms")
    else:
        print("=== a) LM head: TP=1, skip ===")
        lm = {"decode_delta_ms": 0, "prefill_deltas_ms": {},
              "lin_attn_decode_delta_per_layer_ms": 0,
              "lin_attn_prefill_deltas_per_layer_ms": {}}
    print()

    # === b) Communication (read from comm.json) ===
    if args.comm_json:
        with open(args.comm_json) as f:
            comm = json.load(f)
        print(f"=== b) Communication (from {args.comm_json}) ===")
        print(f"  Total/step: {comm['total_per_step_ms']:.3f} ms ({comm.get('method', '?')})")
    elif tp_size > 1:
        print("=== b) Communication: no --comm-json, using 0 ===")
        print("    Run comm_scenarios first to measure AR/AG latency")
        comm = {"total_per_step_ms": 0, "method": "none"}
    else:
        print("=== b) Communication: TP=1, skip ===")
        comm = {"total_per_step_ms": 0, "method": "none"}
    print()

    # === c) Tail from trace (lm_head + sampling per step) ===
    tail = None
    if args.asys_sqlite:
        print("=== c) Tail from trace (lm_head + sampling) ===")
        tail = measure_tail_from_trace(args.asys_sqlite, args.lm_head_kernel)
        if tail:
            print(f"  → tail_per_step: {tail['tail_per_step_ms']:.4f} ms")
    if not tail:
        print("=== c) Tail: no trace or kernel not found, using 0 ===")
        tail = {"tail_per_step_ms": 0}
    print()

    # === d) Apply ===
    print("=== Compensated Results ===")
    print(f"{'input':>8} {'raw_ttft':>10} {'comp_ttft':>10} "
          f"{'raw_tpot':>10} {'comp_tpot':>10} {'comp_total':>10}")
    print("-" * 65)

    results = bench.get("results", [bench])
    compensated = []
    for r in results:
        if "error" in r:
            compensated.append(r); continue

        il = r["input_len"]
        raw_ttft = r["ttft_median_ms"]
        raw_tpot = r["tpot_median_ms"]
        output_tokens = r.get("output_tokens", output_len)

        # Approach: subtract tail (lm_head + sampling), scale encoder, add back
        # raw_TPOT = encoder_per_step + tail + overhead
        # raw_TTFT = prefill_encoder + tail + overhead
        tail_ms = tail["tail_per_step_ms"]
        layer_scale = original / pruned if pruned > 0 else 1

        decode_encoder = raw_tpot - tail_ms
        prefill_encoder = raw_ttft - tail_ms

        comp_tpot = decode_encoder * layer_scale + tail_ms + comm["total_per_step_ms"]
        comp_ttft = prefill_encoder * layer_scale + tail_ms

        # lm_head TP delta (replicated lm_head)
        comp_tpot -= lm["decode_delta_ms"]
        pf_delta = lm["prefill_deltas_ms"].get(str(il), lm["decode_delta_ms"])
        comp_ttft -= pf_delta

        # Linear attention TP delta (scaled from lm_head)
        n_lin_layers = round(original * lin_ratio)
        lin_decode = lm.get("lin_attn_decode_delta_per_layer_ms", 0)
        lin_prefill = lm.get("lin_attn_prefill_deltas_per_layer_ms", {}).get(
            str(il), lin_decode)
        comp_tpot -= n_lin_layers * lin_decode
        comp_ttft -= n_lin_layers * lin_prefill

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
        "lm_head": lm,
        "communication": comm,
        "tail": tail,
        "layer_scale": original / pruned if pruned > 0 else 1,
        "results": compensated,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
