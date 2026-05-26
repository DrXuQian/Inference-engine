#!/usr/bin/env python3
"""
Measure all-reduce and all-gather communication overhead.

NVIDIA: uses nccl_bench.py (torch.distributed)
PPU: uses pccl_tools (standalone)

Usage:
    # NVIDIA
    python compensate_comm.py --model-dir /path/to/model --tp-size 2 --platform nvidia

    # PPU
    python compensate_comm.py --model-dir /path/to/model --tp-size 2 --platform ppu \
        --pccl-ar /usr/local/PPU_SDK/pccl_tools/all_reduce_perf \
        --pccl-ag /usr/local/PPU_SDK/pccl_tools/all_gather_perf

    # From asys/nsys trace (most accurate)
    python compensate_comm.py --model-dir /path/to/model --tp-size 2 \
        --trace-sqlite result.sqlite --platform ppu
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from collections import defaultdict


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    return {"hidden_size": tc["hidden_size"], "vocab_size": tc["vocab_size"],
            "num_hidden_layers": tc["num_hidden_layers"]}


def measure_nvidia(hidden: int, vocab: int, num_layers: int, tp_size: int) -> dict:
    """NCCL bench via subprocess."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nccl_bench.py")
    env = os.environ.copy()
    env["TRITON_BACKENDS_IN_TREE"] = "1"
    result = subprocess.run(
        [sys.executable, script, "--hidden-size", str(hidden),
         "--num-layers", str(num_layers), "--world-size", str(tp_size)],
        env=env, capture_output=True, text=True, timeout=120)

    ar_us = ag_us = 0.0
    for line in result.stdout.split("\n"):
        if "4KB" in line:
            nums = []
            for x in line.split():
                try: nums.append(float(x))
                except ValueError: pass
            if len(nums) >= 2:
                ar_us, ag_us = nums[0], nums[1]
                break

    n_ar = num_layers * 2
    n_ag = 2
    return {"method": "nccl_standalone",
            "ar_us": round(ar_us, 1), "ag_us": round(ag_us, 1),
            "n_ar": n_ar, "n_ag": n_ag,
            "total_per_step_ms": round((n_ar * ar_us + n_ag * ag_us) / 1000, 3)}


def measure_pccl(hidden: int, vocab: int, num_layers: int, tp_size: int,
                 ar_tool: str, ag_tool: str) -> dict:
    """PCCL standalone bench."""
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


def measure_trace(sqlite_path: str, platform: str, num_layers: int) -> dict:
    """Extract real comm kernel time from nsys/asys sqlite."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    kt = "HGPTI_ACTIVITY_KIND_KERNEL" if platform == "ppu" else "CUPTI_ACTIVITY_KIND_KERNEL"
    # Auto-detect table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    if kt not in tables:
        for t in tables:
            if "KERNEL" in t and "ACTIVITY" in t:
                kt = t; break

    cursor.execute(f"""
        SELECT k.start, k."end" - k.start AS dur, s.value AS name
        FROM {kt} k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """)
    events = cursor.fetchall()
    conn.close()

    if not events:
        return {"method": "trace", "error": "no events", "total_per_step_ms": 0}

    t_min = events[0][0]

    # Find serving window (after largest idle gap)
    bins = defaultdict(int)
    for start, dur, name in events:
        bins[(start - t_min) // 1_000_000_000] += 1
    max_bin = max(bins.keys()) if bins else 0
    best_gap_start = best_gap_len = gap_start = gap_len = 0
    in_gap = False
    for b in range(max_bin + 1):
        if bins[b] == 0:
            if not in_gap: gap_start = b; in_gap = True; gap_len = 1
            else: gap_len += 1
        else:
            if in_gap and gap_len > best_gap_len:
                best_gap_start = gap_start; best_gap_len = gap_len
            in_gap = False
    serve_start = t_min + (best_gap_start + best_gap_len) * 1_000_000_000

    # Count comm kernels in serving phase
    comm_kw = ["allreduce", "all_reduce", "cross_device_reduce",
               "allgather", "all_gather", "pccl", "nccl"]
    comm_ns = 0; comm_count = 0; serve_count = 0
    for start, dur, name in events:
        if start < serve_start: continue
        serve_count += 1
        if any(kw in name.lower() for kw in comm_kw):
            comm_ns += dur; comm_count += 1

    # Estimate steps from burst count
    serve_events = [(s, d) for s, d, n in events if s >= serve_start]
    serve_events.sort()
    n_bursts = 1
    for i in range(1, len(serve_events)):
        if serve_events[i][0] - (serve_events[i-1][0] + serve_events[i-1][1]) > 500_000:
            n_bursts += 1

    per_step = (comm_ns / n_bursts) / 1e6 if n_bursts > 0 else 0

    return {"method": "trace", "comm_count": comm_count,
            "comm_total_ms": round(comm_ns / 1e6, 2),
            "serve_steps": n_bursts,
            "total_per_step_ms": round(per_step, 3)}


def main():
    ap = argparse.ArgumentParser(description="Communication overhead measurement")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--platform", choices=["nvidia", "ppu"], default="nvidia")
    ap.add_argument("--pccl-ar", default="/usr/local/PPU_SDK/pccl_tools/all_reduce_perf")
    ap.add_argument("--pccl-ag", default="/usr/local/PPU_SDK/pccl_tools/all_gather_perf")
    ap.add_argument("--trace-sqlite", default=None, help="nsys/asys sqlite for real comm time")
    ap.add_argument("--output-json", default="comm_comp.json")
    args = ap.parse_args()

    if args.tp_size <= 1:
        print("TP=1, no communication. Saving zero.")
        result = {"method": "none", "total_per_step_ms": 0}
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        return

    cfg = load_config(args.model_dir)
    print(f"Model: hidden={cfg['hidden_size']}, vocab={cfg['vocab_size']}, "
          f"layers={cfg['num_hidden_layers']}, TP={args.tp_size}")

    if args.trace_sqlite:
        print(f"\nUsing trace: {args.trace_sqlite}")
        result = measure_trace(args.trace_sqlite, args.platform, cfg["num_hidden_layers"])
    elif args.platform == "nvidia":
        print("\nUsing NCCL standalone bench...")
        result = measure_nvidia(cfg["hidden_size"], cfg["vocab_size"],
                                cfg["num_hidden_layers"], args.tp_size)
    else:
        print("\nUsing PCCL standalone bench...")
        result = measure_pccl(cfg["hidden_size"], cfg["vocab_size"],
                              cfg["num_hidden_layers"], args.tp_size,
                              args.pccl_ar, args.pccl_ag)

    print(f"\nResult: {result['total_per_step_ms']:.3f} ms/decode_step "
          f"(method: {result['method']})")

    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
