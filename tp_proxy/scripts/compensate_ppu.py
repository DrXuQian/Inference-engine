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
# a) LM head
# ---------------------------------------------------------------------------

def measure_lm_head(hidden: int, vocab: int, tp_size: int,
                    input_lens: list[int]) -> dict:
    if tp_size <= 1:
        return {"decode_delta_ms": 0,
                "prefill_deltas_ms": {str(il): 0 for il in input_lens}}

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

    return {"decode_delta_ms": round(decode_delta, 4),
            "prefill_deltas_ms": {str(k): round(v, 4) for k, v in prefill_deltas.items()}}


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

def measure_encoder_trace(sqlite_path: str, num_layers: int, output_len: int,
                          lm_head_kernel: str | None,
                          sampling_kernels: set[str]) -> dict | None:
    """Extract encoder block time from trace.

    Timeline structure (from end, working backward):
      ... [prefill (eager, sparse)] [gap] [decode × output_len (CUDA Graph, dense)] [gap] ...

    Decode = dense kernel region (inter-kernel gap < 5us)
    Prefill = sparse kernel region (inter-kernel gap > 10us)
    Request boundary = large gap (> 500us)
    """
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    kt = ""
    for r in cursor.fetchall():
        if "KERNEL" in r[0] and "ACTIVITY" in r[0]:
            kt = r[0]; break
    if not kt:
        return None

    cursor.execute(f"""
        SELECT k.start, k."end" - k.start AS dur, k."end", s.value AS name
        FROM {kt} k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """)
    all_events = [(r[0], r[1], r[2], r[3]) for r in cursor.fetchall()]
    conn.close()

    # Find serving phase (after largest idle gap)
    t_min = all_events[0][0]
    bins = defaultdict(int)
    for s, d, e, n in all_events:
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

    events = [(s, d, e, n) for s, d, e, n in all_events if s >= serve_start]
    if not events:
        return None

    # --- Split into request blocks (separated by large gaps > 500us) ---
    request_blocks = []
    current = [events[0]]
    for ev in events[1:]:
        gap = ev[0] - current[-1][2]  # start - prev_end
        if gap > 500_000:  # 500us = new request block
            request_blocks.append(current)
            current = [ev]
        else:
            current.append(ev)
    if current:
        request_blocks.append(current)

    # --- Within each request block, split prefill vs decode by kernel density ---
    # Prefill (eager): large gaps between kernels (>10us median gap)
    # Decode (CUDA Graph): tiny gaps (<5us median gap)
    #
    # Structure: [prefill kernels (sparse)] [transition gap] [decode kernels (dense)]
    # Find the transition point: scan from start, when gap pattern changes from sparse to dense

    decode_encoder_ns = 0
    prefill_encoder_ns = 0
    total_lm_head_ns = 0
    total_sampling_ns = 0
    n_decode_steps = 0
    n_prefill_steps = 0
    lm_head_count = 0

    for block in request_blocks:
        if len(block) < 10:
            continue

        # Compute inter-kernel gaps
        gaps = []
        for i in range(1, len(block)):
            gaps.append(block[i][0] - block[i - 1][2])

        # Find transition: sliding window median gap
        # Dense (decode) has median gap < 5us, sparse (prefill) has > 10us
        window = min(50, len(gaps) // 4)
        if window < 5:
            # Too few kernels, treat all as decode
            split_idx = 0
        else:
            split_idx = 0
            for i in range(len(gaps) - window):
                window_gaps = sorted(gaps[i:i + window])
                median_gap = window_gaps[len(window_gaps) // 2]
                if median_gap < 5000:  # < 5us median = decode region starts
                    split_idx = i
                    break

        prefill_part = block[:split_idx] if split_idx > 0 else []
        decode_part = block[split_idx:] if split_idx < len(block) else block

        # --- For each part, identify non-encoder (lm_head + sampling) ---
        def classify_part(part):
            enc_ns = 0; lm_ns = 0; samp_ns = 0; lm_cnt = 0

            # Find sampling positions
            samp_indices = set()
            for i, (s, d, e, n) in enumerate(part):
                if n in sampling_kernels:
                    samp_indices.add(i)

            # For each sampling, walk backward to find lm_head
            lm_indices = set()
            non_enc_indices = set()
            for si in sorted(samp_indices):
                for j in range(si - 1, max(si - 50, -1), -1):
                    if lm_head_kernel and part[j][3] == lm_head_kernel:
                        lm_indices.add(j)
                        break
                    elif not lm_head_kernel and any(
                            kw in part[j][3].lower() for kw in ["gemv", "gemm"]):
                        lm_indices.add(j)
                        break

            # Mark lm_head → sampling range as non-encoder
            for si in sorted(samp_indices):
                lm_idx = None
                for j in range(si - 1, max(si - 50, -1), -1):
                    if j in lm_indices:
                        lm_idx = j; break
                start_idx = lm_idx if lm_idx is not None else si
                for k in range(start_idx, si + 1):
                    non_enc_indices.add(k)

            for i, (s, d, e, n) in enumerate(part):
                if i in lm_indices:
                    lm_ns += d; lm_cnt += 1
                elif i in non_enc_indices:
                    samp_ns += d
                else:
                    enc_ns += d

            n_fwd = len(samp_indices) or (lm_cnt or 0)
            return enc_ns, lm_ns, samp_ns, lm_cnt, n_fwd

        # Classify decode
        if decode_part:
            d_enc, d_lm, d_samp, d_lm_cnt, d_n_fwd = classify_part(decode_part)
            decode_encoder_ns += d_enc
            total_lm_head_ns += d_lm
            total_sampling_ns += d_samp
            lm_head_count += d_lm_cnt
            n_decode_steps += d_n_fwd

        # Classify prefill
        if prefill_part:
            p_enc, p_lm, p_samp, p_lm_cnt, p_n_fwd = classify_part(prefill_part)
            prefill_encoder_ns += p_enc
            total_lm_head_ns += p_lm
            total_sampling_ns += p_samp
            lm_head_count += p_lm_cnt
            n_prefill_steps += p_n_fwd or 1

    # Compute per-layer
    n_requests = len([b for b in request_blocks if len(b) >= 10])
    if n_decode_steps == 0:
        n_decode_steps = n_requests * output_len
    if n_prefill_steps == 0:
        n_prefill_steps = n_requests

    decode_per_layer = decode_encoder_ns / max(n_decode_steps * num_layers, 1)
    prefill_per_layer = prefill_encoder_ns / max(n_prefill_steps * num_layers, 1)

    total_encoder_ns = decode_encoder_ns + prefill_encoder_ns

    return {
        "decode_per_layer_ms": round(decode_per_layer / 1e6, 5),
        "prefill_per_layer_ms": round(prefill_per_layer / 1e6, 4),
        "decode_encoder_total_ms": round(decode_encoder_ns / 1e6, 2),
        "prefill_encoder_total_ms": round(prefill_encoder_ns / 1e6, 2),
        "encoder_total_ms": round(total_encoder_ns / 1e6, 2),
        "lm_head_total_ms": round(total_lm_head_ns / 1e6, 2),
        "lm_head_per_call_us": round(total_lm_head_ns / max(lm_head_count, 1) / 1e3, 1),
        "sampling_total_ms": round(total_sampling_ns / 1e6, 2),
        "n_requests": n_requests,
        "n_decode_steps": n_decode_steps,
        "n_prefill_steps": n_prefill_steps,
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

    print(f"Model: hidden={hidden}, vocab={vocab}")
    print(f"Layers: {pruned} pruned / {original} original, TP={tp_size}")
    print()

    # === a) LM head ===
    if tp_size > 1:
        print("=== a) LM head (torch.mm) ===")
        lm = measure_lm_head(hidden, vocab, tp_size, input_lens)
        print(f"  Decode delta: {lm['decode_delta_ms']:.3f} ms")
        for il, d in lm["prefill_deltas_ms"].items():
            print(f"  Prefill delta (input={il}): {d:.3f} ms")
    else:
        print("=== a) LM head: TP=1, skip ===")
        lm = {"decode_delta_ms": 0, "prefill_deltas_ms": {}}
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

    # === c) Encoder block ===
    enc = None
    if pruned < original and args.asys_sqlite:
        print("=== c) Encoder block (from trace) ===")
        enc = measure_encoder_trace(args.asys_sqlite, pruned, output_len,
                                    args.lm_head_kernel, sampling_names)
        if enc:
            removed = original - pruned
            print(f"  Decode per_layer: {enc['decode_per_layer_ms']:.5f} ms")
            print(f"  Prefill per_layer: {enc['prefill_per_layer_ms']:.4f} ms")
            print(f"  Compensation ({removed} layers):")
            print(f"    TPOT: +{removed * enc['decode_per_layer_ms']:.3f} ms")
            print(f"    TTFT: +{removed * enc['prefill_per_layer_ms']:.3f} ms")
            print(f"  LM head per call: {enc['lm_head_per_call_us']:.1f} us")
    elif pruned < original:
        print("=== c) Encoder block: no asys trace, skip ===")
        print("    (provide --asys-sqlite for encoder block compensation)")
    else:
        print("=== c) Encoder block: no pruning, skip ===")
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

        comp_tpot = raw_tpot - lm["decode_delta_ms"] + comm["total_per_step_ms"]
        pf_delta = lm["prefill_deltas_ms"].get(str(il), lm["decode_delta_ms"])
        comp_ttft = raw_ttft - pf_delta

        if enc and pruned < original:
            removed = original - pruned
            comp_tpot += removed * enc["decode_per_layer_ms"]
            comp_ttft += removed * enc["prefill_per_layer_ms"]

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
        "encoder_block": enc,
        "results": compensated,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
