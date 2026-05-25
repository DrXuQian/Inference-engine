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
        SELECT k.start, k."end" - k.start AS dur, s.value AS name
        FROM {kt} k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """)
    all_events = cursor.fetchall()
    conn.close()

    # Find serving phase
    t_min = all_events[0][0]
    bins = defaultdict(int)
    for s, d, n in all_events:
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

    events = [(s, d, n) for s, d, n in all_events if s >= serve_start]
    name_durations = defaultdict(list)
    for s, d, n in events:
        name_durations[n].append(d)

    total_ns = sum(d for _, d, _ in events)

    # LM head
    lm_head_ns = 0; lm_head_count = 0
    if lm_head_kernel and lm_head_kernel in name_durations:
        all_durs = sorted(name_durations[lm_head_kernel])
        median = all_durs[len(all_durs) // 2]
        threshold = median * 10
        large = [d for d in all_durs if d > threshold]
        lm_head_ns = sum(large)
        lm_head_count = len(large)
    else:
        # Auto-detect
        for name, durs in name_durations.items():
            if any(kw in name.lower() for kw in ["gemv", "gemm", "matmul"]):
                all_durs = sorted(durs)
                median = all_durs[len(all_durs) // 2]
                threshold = median * 10
                large = [d for d in all_durs if d > threshold]
                if len(large) > 0:
                    lm_head_ns = sum(large)
                    lm_head_count = len(large)
                    break

    # Sampling
    sampling_ns = 0
    for name in sampling_kernels:
        if name in name_durations:
            sampling_ns += sum(name_durations[name])

    # Encoder
    encoder_ns = total_ns - lm_head_ns - sampling_ns
    n_fwd = lm_head_count or 1
    n_requests = max(round(n_fwd / output_len), 1)
    n_decode = n_requests * output_len

    # Prefill/decode split (heuristic: prefill 5x heavier)
    prefill_weight = 5
    total_w = n_decode + n_requests * prefill_weight
    decode_enc = encoder_ns * n_decode / total_w
    prefill_enc = encoder_ns * n_requests * prefill_weight / total_w

    decode_per_layer = decode_enc / max(n_decode, 1) / num_layers
    prefill_per_layer = prefill_enc / max(n_requests, 1) / num_layers

    return {
        "decode_per_layer_ms": round(decode_per_layer / 1e6, 5),
        "prefill_per_layer_ms": round(prefill_per_layer / 1e6, 4),
        "encoder_total_ms": round(encoder_ns / 1e6, 2),
        "lm_head_total_ms": round(lm_head_ns / 1e6, 2),
        "lm_head_per_call_us": round(lm_head_ns / max(lm_head_count, 1) / 1e3, 1),
        "n_requests": n_requests,
        "n_decode": n_decode,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PPU compensation (all-in-one)")
    ap.add_argument("--bench-results", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--asys-sqlite", default=None,
                    help="asys trace sqlite (for encoder block + optional comm)")
    ap.add_argument("--pruned-layers", type=int, default=None)
    ap.add_argument("--original-layers", type=int, default=None)
    ap.add_argument("--tp-size", type=int, default=None)
    ap.add_argument("--pccl-ar", default="/usr/local/PPU_SDK/pccl_tools/all_reduce_perf")
    ap.add_argument("--pccl-ag", default="/usr/local/PPU_SDK/pccl_tools/all_gather_perf")
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

    input_lens = [r["input_len"] for r in bench.get("results", [bench]) if "error" not in r]

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

    # === b) Communication ===
    if tp_size > 1:
        print("=== b) Communication ===")
        if args.asys_sqlite:
            print(f"  From trace: {args.asys_sqlite}")
            comm = measure_comm_trace(args.asys_sqlite, original, tp_size)
        else:
            print(f"  From pccl_tools")
            comm = measure_comm_pccl(hidden, vocab, original, tp_size,
                                     args.pccl_ar, args.pccl_ag)
        print(f"  Total/step: {comm['total_per_step_ms']:.3f} ms ({comm['method']})")
    else:
        print("=== b) Communication: TP=1, skip ===")
        comm = {"total_per_step_ms": 0, "method": "none"}
    print()

    # === c) Encoder block ===
    enc = None
    if pruned < original and args.asys_sqlite:
        print("=== c) Encoder block (from trace) ===")
        enc = measure_encoder_trace(args.asys_sqlite, pruned, args.output_len,
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
        output_tokens = r.get("output_tokens", args.output_len)

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
