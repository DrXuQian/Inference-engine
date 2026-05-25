#!/usr/bin/env python3
"""
Compensate single-card TP proxy results for PPU platform.

Same compensation logic as compensate.py but uses:
  - pccl_tools for communication benchmark
  - asys + sqlite for trace analysis (instead of nsys)

Usage:
    python compensate_ppu.py \
        --bench-results bench_results.json \
        --model-dir /sim/eec/shared/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
        --pruned-layers 10 --original-layers 40 \
        --tp-size 2

    # With custom pccl paths
    python compensate_ppu.py \
        --bench-results bench_results.json \
        --model-dir /path/to/model \
        --pruned-layers 10 --original-layers 40 \
        --tp-size 2 \
        --pccl-ar /usr/local/PPU_SDK/pccl_tools/all_reduce_perf \
        --pccl-ag /usr/local/PPU_SDK/pccl_tools/all_gather_perf

    # With asys trace for real comm time (instead of standalone pccl)
    python compensate_ppu.py \
        --bench-results bench_results.json \
        --model-dir /path/to/model \
        --pruned-layers 40 --original-layers 40 \
        --tp-size 2 \
        --asys-sqlite /path/to/result.sqlite
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys


# ---------------------------------------------------------------------------
# Config extraction
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


# ---------------------------------------------------------------------------
# a) LM head compensation (same as NVIDIA — torch.mm)
# ---------------------------------------------------------------------------

def measure_lm_head(hidden: int, vocab: int, tp_size: int,
                    input_lens: list[int]) -> dict:
    import torch
    device = "cuda"
    dtype = torch.bfloat16
    vocab_half = vocab // tp_size

    def bench_mm(M, K, N, warmup=50, iters=200):
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        for _ in range(warmup):
            torch.mm(A, B)
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

    full_dec = bench_mm(1, hidden, vocab)
    half_dec = bench_mm(1, hidden, vocab_half)
    decode_delta = full_dec - half_dec

    prefill_deltas = {}
    for il in input_lens:
        full_pre = bench_mm(il, hidden, vocab)
        half_pre = bench_mm(il, hidden, vocab_half)
        prefill_deltas[il] = full_pre - half_pre

    return {
        "decode_delta_ms": round(decode_delta, 4),
        "prefill_deltas_ms": {str(k): round(v, 4) for k, v in prefill_deltas.items()},
    }


# ---------------------------------------------------------------------------
# b) Communication: pccl_tools or asys sqlite
# ---------------------------------------------------------------------------

def measure_comm_pccl(hidden: int, vocab: int, num_layers: int,
                      tp_size: int, ar_tool: str, ag_tool: str) -> dict:
    """Measure communication using pccl_tools standalone bench."""
    ar_size = hidden * 2  # bf16 decode: [1, hidden]
    ag_size = (vocab // tp_size) * 2  # bf16: [1, vocab/tp]

    def run_pccl(tool, size, iters=200, warmup=50):
        cmd = [tool, "-b", str(size), "-e", str(size), "-f", "2",
               "-d", "bf16", "-o", "sum", "-n", str(iters), "-w", str(warmup),
               "-g", str(tp_size), "-c", "0", "-a", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # nccl-tests output: column 5 = out-of-place time (us)
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    return float(parts[5])
                except (ValueError, IndexError):
                    pass
        return 0.0

    ar_us = run_pccl(ar_tool, ar_size)
    ag_us = run_pccl(ag_tool, ag_size)

    n_ar = num_layers * 2
    n_ag = 2
    total_ms = (n_ar * ar_us + n_ag * ag_us) / 1000

    return {
        "method": "pccl_standalone",
        "ar_per_call_us": round(ar_us, 1),
        "ag_per_call_us": round(ag_us, 1),
        "n_ar_per_step": n_ar,
        "n_ag_per_step": n_ag,
        "total_per_step_ms": round(total_ms, 3),
        "note": "Standalone bench. If vLLM uses custom P2P reduce, actual may be lower.",
    }


def measure_comm_asys(sqlite_path: str, num_layers: int) -> dict:
    """Extract real communication kernel time from asys sqlite trace.

    Looks for:
      - All-reduce kernels (PCCL, NCCL, or custom reduce)
      - All-gather kernels
    in the serving phase (after largest idle gap).
    """
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Get all kernel names + times
    cursor.execute("""
        SELECT k.start, k."end" - k.start AS duration, k.deviceId, s.value AS name
        FROM HGPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """)
    events = cursor.fetchall()
    conn.close()

    if not events:
        print("  WARNING: no kernel events in asys trace")
        return {"method": "asys_trace", "total_per_step_ms": 0, "error": "no events"}

    t_min = events[0][0]

    # Find serving phase: largest idle gap (1s bins)
    from collections import defaultdict
    bins = defaultdict(int)
    for start, dur, dev, name in events:
        b = (start - t_min) // 1_000_000_000
        bins[b] += 1

    max_bin = max(bins.keys()) if bins else 0
    best_gap_start = best_gap_len = 0
    gap_start = gap_len = 0
    in_gap = False
    for b in range(max_bin + 1):
        if bins[b] == 0:
            if not in_gap:
                gap_start = b
                in_gap = True
                gap_len = 1
            else:
                gap_len += 1
        else:
            if in_gap and gap_len > best_gap_len:
                best_gap_start = gap_start
                best_gap_len = gap_len
            in_gap = False

    serve_start_ns = t_min + (best_gap_start + best_gap_len) * 1_000_000_000

    # Filter serving events, identify communication kernels
    comm_keywords = ["allreduce", "all_reduce", "cross_device_reduce",
                     "allgather", "all_gather", "pccl", "nccl"]

    comm_total_ns = 0
    comm_count = 0
    total_serve_events = 0
    # Only count from one device (device 0) to avoid double-counting
    device0_id = None

    for start, dur, dev, name in events:
        if start < serve_start_ns:
            continue
        if device0_id is None:
            device0_id = dev
        if dev != device0_id:
            continue
        total_serve_events += 1
        nl = name.lower()
        if any(kw in nl for kw in comm_keywords):
            comm_total_ns += dur
            comm_count += 1

    # Estimate forward passes from total events
    # Heuristic: count unique "burst" boundaries (gaps > 100us)
    serve_events = [(s, d, dev, n) for s, d, dev, n in events
                    if s >= serve_start_ns and dev == device0_id]
    serve_events.sort(key=lambda x: x[0])

    n_bursts = 1
    for i in range(1, len(serve_events)):
        gap = serve_events[i][0] - (serve_events[i-1][0] + serve_events[i-1][1])
        if gap > 500_000:  # 500us gap = new step
            n_bursts += 1

    comm_per_step_ms = (comm_total_ns / n_bursts) / 1e6 if n_bursts > 0 else 0
    comm_per_call_us = (comm_total_ns / comm_count) / 1e3 if comm_count > 0 else 0

    return {
        "method": "asys_trace",
        "comm_kernel_count": comm_count,
        "comm_total_ms": round(comm_total_ns / 1e6, 2),
        "serve_events": total_serve_events,
        "n_bursts": n_bursts,
        "comm_per_call_us": round(comm_per_call_us, 1),
        "comm_per_step_ms": round(comm_per_step_ms, 3),
        "total_per_step_ms": round(comm_per_step_ms, 3),
    }


# ---------------------------------------------------------------------------
# c) Encoder block compensation (differential — same as NVIDIA)
# ---------------------------------------------------------------------------

def _get_attn_cycle(model_dir: str) -> int:
    """Get attention type cycle length from config."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    interval = tc.get("full_attention_interval")
    if interval:
        return interval
    layer_types = tc.get("layer_types", [])
    if not layer_types:
        return 1
    for cycle in range(1, len(layer_types) + 1):
        pattern = layer_types[:cycle]
        if all(layer_types[i] == pattern[i % cycle] for i in range(len(layer_types))):
            return cycle
    return 1


def measure_encoder_block(model_dir: str, pruned_layers: int,
                          input_len: int, output_len: int,
                          gpu_mem: float) -> dict:
    """Differential method aligned to attention cycle."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prune_script = os.path.join(script_dir, "prune_layers.py")
    bench_script = os.path.join(script_dir, "generate_bench.py")

    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp(prefix="encoder_comp_")

    cycle = _get_attn_cycle(model_dir)
    n_hi = (pruned_layers // cycle) * cycle
    if n_hi >= 2 * cycle:
        n_lo = max(n_hi // 2, cycle)
        n_lo = (n_lo // cycle) * cycle
        if n_lo == n_hi:
            n_lo = n_hi - cycle
    elif n_hi > 1:
        n_lo = max(n_hi // 2, 1)
    else:
        return None
    print(f"  Attention cycle: {cycle} (using {n_lo}L and {n_hi}L for differential)")

    results = {}
    for n in [n_lo, n_hi]:
        pruned_dir = os.path.join(tmp_dir, f"pruned_{n}L")
        subprocess.run([sys.executable, prune_script,
                        "--rank-dir", model_dir,
                        "--num-layers", str(n),
                        "--output-dir", pruned_dir],
                       capture_output=True, check=True, timeout=300)

        tmp_json = os.path.join(tmp_dir, f"bench_{n}L.json")
        env = os.environ.copy()
        env["TRITON_BACKENDS_IN_TREE"] = "1"
        subprocess.run([sys.executable, bench_script,
                        "--model", pruned_dir,
                        "--input-len", str(input_len),
                        "--output-len", str(output_len),
                        "--num-prompts", "5", "--num-warmup", "2",
                        "--gpu-mem", str(gpu_mem),
                        "--output-json", tmp_json],
                       env=env, capture_output=True, timeout=600)

        with open(tmp_json) as f:
            results[n] = json.load(f)
        shutil.rmtree(pruned_dir, ignore_errors=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    tpot_hi = results[n_hi]["tpot_median_ms"]
    tpot_lo = results[n_lo]["tpot_median_ms"]
    ttft_hi = results[n_hi]["ttft_median_ms"]
    ttft_lo = results[n_lo]["ttft_median_ms"]

    return {
        "n_lo": n_lo, "n_hi": n_hi,
        "tpot_lo": tpot_lo, "tpot_hi": tpot_hi,
        "ttft_lo": ttft_lo, "ttft_hi": ttft_hi,
        "per_layer_tpot_ms": round((tpot_hi - tpot_lo) / (n_hi - n_lo), 4),
        "per_layer_ttft_ms": round((ttft_hi - ttft_lo) / (n_hi - n_lo), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compensate TP proxy results (PPU)")
    ap.add_argument("--bench-results", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pruned-layers", type=int, required=True)
    ap.add_argument("--original-layers", type=int, required=True)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--pccl-ar", default="/usr/local/PPU_SDK/pccl_tools/all_reduce_perf")
    ap.add_argument("--pccl-ag", default="/usr/local/PPU_SDK/pccl_tools/all_gather_perf")
    ap.add_argument("--asys-sqlite", default=None,
                    help="Path to asys exported sqlite for real comm time")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--skip-encoder", action="store_true")
    ap.add_argument("--output-json", default="compensated_results_ppu.json")
    args = ap.parse_args()

    with open(args.bench_results) as f:
        bench = json.load(f)

    cfg = load_model_config(args.model_dir)
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    input_lens = [r["input_len"] for r in bench["results"] if "error" not in r]

    print(f"Model: hidden={hidden}, vocab={vocab}, "
          f"layers={args.original_layers} (pruned={args.pruned_layers})")
    print(f"Platform: PPU, TP={args.tp_size}")
    print()

    # a) LM head (only for TP > 1)
    if args.tp_size > 1:
        print("=== a) LM head compensation ===")
        lm = measure_lm_head(hidden, vocab, args.tp_size, input_lens)
        print(f"  Decode delta: {lm['decode_delta_ms']:.3f} ms")
        for il, d in lm["prefill_deltas_ms"].items():
            print(f"  Prefill delta (input={il}): {d:.3f} ms")
        print()
    else:
        print("=== a) LM head: TP=1, no compensation needed ===\n")
        lm = {"decode_delta_ms": 0, "prefill_deltas_ms": {}}

    # b) Communication (only for TP > 1)
    if args.tp_size > 1:
        print("=== b) Communication compensation ===")
        if args.asys_sqlite:
            print(f"  Using asys trace: {args.asys_sqlite}")
            comm = measure_comm_asys(args.asys_sqlite, args.original_layers)
            print(f"  Method: {comm['method']}")
            print(f"  Comm kernels: {comm.get('comm_kernel_count', 0)}")
            print(f"  Per-call: {comm.get('comm_per_call_us', 0):.1f} us")
            print(f"  Per-step: {comm['total_per_step_ms']:.3f} ms")
        else:
            print(f"  Using pccl_tools standalone")
            comm = measure_comm_pccl(hidden, vocab, args.original_layers,
                                     args.tp_size, args.pccl_ar, args.pccl_ag)
            print(f"  AR: {comm['ar_per_call_us']:.1f} us/call × {comm['n_ar_per_step']}")
            print(f"  AG: {comm['ag_per_call_us']:.1f} us/call × {comm['n_ag_per_step']}")
            print(f"  Total/step: {comm['total_per_step_ms']:.3f} ms")
            print(f"  NOTE: {comm.get('note', '')}")
        print()
    else:
        print("=== b) Communication: TP=1, no compensation needed ===\n")
        comm = {"total_per_step_ms": 0}

    # c) Encoder block
    enc = None
    if not args.skip_encoder and args.pruned_layers < args.original_layers:
        print("=== c) Encoder block compensation (differential) ===")
        enc = measure_encoder_block(
            args.model_dir, args.pruned_layers,
            input_lens[0], bench["output_len"], args.gpu_mem,
        )
        removed = args.original_layers - args.pruned_layers
        print(f"  {enc['n_lo']}L→{enc['n_hi']}L: "
              f"per_layer_tpot={enc['per_layer_tpot_ms']:.4f}ms, "
              f"per_layer_ttft={enc['per_layer_ttft_ms']:.4f}ms")
        print(f"  TPOT comp: {removed} × {enc['per_layer_tpot_ms']:.4f} = "
              f"{removed * enc['per_layer_tpot_ms']:.3f} ms")
        print()

    # d) Apply
    print("=== Compensated Results ===")
    print(f"{'input':>8} {'raw_ttft':>10} {'comp_ttft':>10} "
          f"{'raw_tpot':>10} {'comp_tpot':>10} {'raw_total':>10} {'comp_total':>10}")
    print("-" * 75)

    compensated = []
    for r in bench["results"]:
        if "error" in r:
            compensated.append(r)
            continue

        il = r["input_len"]
        raw_ttft = r["ttft_median_ms"]
        raw_tpot = r["tpot_median_ms"]
        raw_total = r["total_median_ms"]
        output_len = r.get("output_tokens", bench.get("output_len", 64))

        comp_tpot = raw_tpot - lm["decode_delta_ms"] + comm["total_per_step_ms"]
        pf_delta = lm["prefill_deltas_ms"].get(str(il), lm["decode_delta_ms"])
        comp_ttft = raw_ttft - pf_delta

        if enc and args.pruned_layers < args.original_layers:
            removed = args.original_layers - args.pruned_layers
            comp_tpot += removed * enc["per_layer_tpot_ms"]
            comp_ttft += removed * enc["per_layer_ttft_ms"]

        comp_total = comp_ttft + (output_len - 1) * comp_tpot

        print(f"{il:>8} {raw_ttft:>10.2f} {comp_ttft:>10.2f} "
              f"{raw_tpot:>10.3f} {comp_tpot:>10.3f} {raw_total:>10.1f} {comp_total:>10.1f}")

        compensated.append({
            **r,
            "comp_ttft_ms": round(comp_ttft, 3),
            "comp_tpot_ms": round(comp_tpot, 3),
            "comp_total_ms": round(comp_total, 3),
        })

    output = {
        "platform": "ppu",
        "model_dir": args.model_dir,
        "tp_size": args.tp_size,
        "pruned_layers": args.pruned_layers,
        "original_layers": args.original_layers,
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
