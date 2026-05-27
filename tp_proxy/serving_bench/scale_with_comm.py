#!/usr/bin/env python3
"""
Scale serving benchmark results from source platform to target platform,
using measured comm.json for both platforms.

Approach:
  TTFT = prefill_compute + prefill_comm
    prefill_compute scales with FLOPS ratio
    prefill_comm is replaced by target comm.json's prefill_total_per_step_ms

  TPOT = decode_compute + decode_comm
    decode_compute = TPOT_src - src_comm → scale with BW ratio → + tgt_comm
    decode_comm is replaced by target comm.json's decode_total_per_step_ms

Usage:
    # Generate comm.json on source platform (NVIDIA with nccl-tests):
    bash comm_bench_nccl.sh /path/to/model 2 13400 src_comm.json

    # Generate comm.json on target platform (PPU with pccl_tools):
    AR_TOOL=/usr/local/PPU_SDK/pccl_tools/all_reduce_perf \
    AG_TOOL=/usr/local/PPU_SDK/pccl_tools/all_gather_perf \
    bash comm_bench_nccl.sh /path/to/model 2 13400 tgt_comm.json

    # Scale results:
    python scale_with_comm.py \
        --input-csv results/input_sweep/summary.csv \
        --scenario-dir results/scenarios/ \
        --src-comm src_comm.json --tgt-comm tgt_comm.json \
        --src-flops 312 --src-bw 2039 \
        --tgt-flops 100 --tgt-bw 680 \
        --tp-size 2 \
        -o scaled_report.png
"""

import argparse
import csv
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_comm(path: str) -> dict:
    with open(path) as f:
        c = json.load(f)
    return {
        "decode_ms": c.get("decode_total_per_step_ms", c.get("total_per_step_ms", 0)),
        "prefill_ms": c.get("prefill_total_per_step_ms",
                            c.get("decode_total_per_step_ms",
                                  c.get("total_per_step_ms", 0))),
    }


def fmt_ms(v: float) -> str:
    if v < 1:
        return f"{v * 1000:.1f}us"
    if v < 1000:
        return f"{v:.2f}ms"
    if v < 60000:
        return f"{v / 1000:.2f}s"
    if v < 3600000:
        return f"{v / 60000:.2f}min"
    return f"{v / 3600000:.2f}h"


def scale_ttft(ttft_src: float, src_flops: float, tgt_flops: float,
               src_comm_prefill: float, tgt_comm_prefill: float) -> float:
    """TTFT = compute + comm.
    compute scales with FLOPS, comm replaced by target measurement.
    """
    compute = max(ttft_src - src_comm_prefill, 0)
    compute_tgt = compute * (src_flops / tgt_flops) if tgt_flops > 0 else compute
    return compute_tgt + tgt_comm_prefill


def scale_tpot(tpot_src: float, src_bw: float, tgt_bw: float,
               src_comm_decode: float, tgt_comm_decode: float) -> float:
    """TPOT = memory_time + comm.
    memory_time scales with BW, comm replaced by target measurement.
    """
    mem_time = max(tpot_src - src_comm_decode, 0)
    mem_tgt = mem_time * (src_bw / tgt_bw) if tgt_bw > 0 else mem_time
    return mem_tgt + tgt_comm_decode


def parse_scenario_log(path: str) -> dict:
    metrics = {}
    with open(path) as f:
        text = f.read()
    for line in text.split("\n"):
        ll = line.lower().strip()
        if "median" in ll and "ttft" in ll:
            try:
                metrics["ttft"] = float(line.split(":")[-1].strip().replace("ms", "").strip())
            except ValueError:
                pass
        if "median" in ll and ("tpot" in ll or "inter-token" in ll):
            try:
                metrics["tpot"] = float(line.split(":")[-1].strip().replace("ms", "").strip())
            except ValueError:
                pass
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Scale results using measured comm.json")

    ap.add_argument("--src-flops", type=float, required=True,
                    help="Source FP16 TFLOPS per GPU")
    ap.add_argument("--src-bw", type=float, required=True,
                    help="Source DDR bandwidth GB/s per GPU")
    ap.add_argument("--tgt-flops", type=float, required=True,
                    help="Target FP16 TFLOPS per GPU")
    ap.add_argument("--tgt-bw", type=float, required=True,
                    help="Target DDR bandwidth GB/s per GPU")

    ap.add_argument("--src-comm", required=True,
                    help="Source platform comm.json (from comm_bench_nccl.sh)")
    ap.add_argument("--tgt-comm", required=True,
                    help="Target platform comm.json (from comm_bench_nccl.sh)")

    ap.add_argument("--input-csv", default=None,
                    help="Input sweep summary.csv")
    ap.add_argument("--scenario-dir", default=None,
                    help="Scenario logs directory")
    ap.add_argument("--concurrency-dir", default=None,
                    help="Concurrency logs directory")
    ap.add_argument("--tp-size", type=int, default=2)

    ap.add_argument("-o", "--output", default="scaled_comm.png")
    ap.add_argument("--src-label", default="Source")
    ap.add_argument("--tgt-label", default="Target")
    args = ap.parse_args()

    src_comm = load_comm(args.src_comm)
    tgt_comm = load_comm(args.tgt_comm)

    print(f"Source: {args.src_flops} TFLOPS, {args.src_bw} GB/s")
    print(f"  comm decode: {src_comm['decode_ms']:.3f} ms, prefill: {src_comm['prefill_ms']:.3f} ms")
    print(f"Target: {args.tgt_flops} TFLOPS, {args.tgt_bw} GB/s")
    print(f"  comm decode: {tgt_comm['decode_ms']:.3f} ms, prefill: {tgt_comm['prefill_ms']:.3f} ms")
    print(f"TP={args.tp_size}")
    print()

    # ========================================
    # Input sweep
    # ========================================
    if args.input_csv:
        print("=== Input Sweep Scaling ===")
        print(f"{'input':>8} {'src_ttft':>10} {'tgt_ttft':>10} "
              f"{'src_tpot':>10} {'tgt_tpot':>10} {'src_tps':>8} {'tgt_tps':>8}")
        print("-" * 70)

        input_lens = []
        src_ttfts = []; tgt_ttfts = []
        src_tpots = []; tgt_tpots = []

        with open(args.input_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                il = int(row["input_len"])
                ttft = float(row["ttft_median_ms"]) if row.get("ttft_median_ms") else None
                tpot = float(row["tpot_median_ms"]) if row.get("tpot_median_ms") else None
                if ttft is None or tpot is None:
                    continue

                t_ttft = scale_ttft(ttft, args.src_flops, args.tgt_flops,
                                    src_comm["prefill_ms"], tgt_comm["prefill_ms"])
                t_tpot = scale_tpot(tpot, args.src_bw, args.tgt_bw,
                                    src_comm["decode_ms"], tgt_comm["decode_ms"])

                src_tps = 1000 / tpot if tpot > 0 else 0
                tgt_tps = 1000 / t_tpot if t_tpot > 0 else 0

                print(f"{il:>8} {fmt_ms(ttft):>10} {fmt_ms(t_ttft):>10} "
                      f"{fmt_ms(tpot):>10} {fmt_ms(t_tpot):>10} "
                      f"{src_tps:>7.1f} {tgt_tps:>7.1f}")

                input_lens.append(il)
                src_ttfts.append(ttft); tgt_ttfts.append(t_ttft)
                src_tpots.append(tpot); tgt_tpots.append(t_tpot)

        # Save CSV
        csv_out = args.output.replace('.png', '_input_sweep.csv')
        with open(csv_out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["input_len", "src_ttft_ms", "tgt_ttft_ms",
                         "src_tpot_ms", "tgt_tpot_ms", "src_tps", "tgt_tps"])
            for i in range(len(input_lens)):
                w.writerow([input_lens[i],
                            f"{src_ttfts[i]:.2f}", f"{tgt_ttfts[i]:.2f}",
                            f"{src_tpots[i]:.3f}", f"{tgt_tpots[i]:.3f}",
                            f"{1000/src_tpots[i]:.1f}", f"{1000/tgt_tpots[i]:.1f}"])
        print(f"\nSaved CSV: {csv_out}")

        # Plot
        if input_lens:
            def fmt_len(n):
                return f"{n // 1024}K" if n >= 1024 else str(n)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            ax = axes[0]
            ax.plot(input_lens, src_ttfts, '-o', color='#378ADD', lw=2, ms=6,
                    label=args.src_label)
            ax.plot(input_lens, tgt_ttfts, '--^', color='#D85A30', lw=2, ms=7,
                    label=args.tgt_label)
            ax.set_xscale('log', base=2); ax.set_yscale('log')
            ax.set_xticks(input_lens)
            ax.set_xticklabels([fmt_len(il) for il in input_lens], rotation=45)
            ax.set_xlabel('Input Length (tokens)'); ax.set_ylabel('TTFT (ms, log)')
            ax.set_title('TTFT Median'); ax.grid(True, which='both', alpha=0.25)
            ax.legend(frameon=False)

            ax = axes[1]
            ax.plot(input_lens, src_tpots, '-o', color='#378ADD', lw=2, ms=6,
                    label=args.src_label)
            ax.plot(input_lens, tgt_tpots, '--^', color='#D85A30', lw=2, ms=7,
                    label=args.tgt_label)
            ax.set_xscale('log', base=2)
            ax.set_xticks(input_lens)
            ax.set_xticklabels([fmt_len(il) for il in input_lens], rotation=45)
            ax.set_xlabel('Input Length (tokens)'); ax.set_ylabel('TPOT (ms)')
            ax.set_title('TPOT Median'); ax.grid(True, which='both', alpha=0.25)
            ax.legend(frameon=False)

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {args.output}")

    # ========================================
    # Scenario scaling
    # ========================================
    if args.scenario_dir:
        print("\n=== Scenario Scaling ===")
        print(f"{'scenario':>20} {'src_ttft':>10} {'tgt_ttft':>10} "
              f"{'src_tpot':>10} {'tgt_tpot':>10}")
        print("-" * 65)

        for log_name in sorted(os.listdir(args.scenario_dir)):
            if not log_name.endswith(".log"):
                continue
            m = parse_scenario_log(os.path.join(args.scenario_dir, log_name))
            if "ttft" not in m or "tpot" not in m:
                continue
            name = log_name.replace(".log", "")
            t_ttft = scale_ttft(m["ttft"], args.src_flops, args.tgt_flops,
                                src_comm["prefill_ms"], tgt_comm["prefill_ms"])
            t_tpot = scale_tpot(m["tpot"], args.src_bw, args.tgt_bw,
                                src_comm["decode_ms"], tgt_comm["decode_ms"])
            print(f"{name:>20} {fmt_ms(m['ttft']):>10} {fmt_ms(t_ttft):>10} "
                  f"{fmt_ms(m['tpot']):>10} {fmt_ms(t_tpot):>10}")


if __name__ == "__main__":
    main()
