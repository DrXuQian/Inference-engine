#!/usr/bin/env python3
"""
Scale serving benchmark results to target platform.

TTFT (prefill, compute-bound): scales with FLOPS ratio
TPOT (decode, BW-bound):
  tpot_tgt = (tpot_src - src_link_latency) × (src_bw / tgt_bw) + tgt_link_latency

  link_latency = per-step communication latency for decode (TP>1).
  For decode, comm is latency-bound (small messages), NOT throughput-bound.

Usage:
    # Single target (link latency in us, BW in GB/s, FLOPS in TFLOPS)
    python scale_report.py \
        --input-csv results/input_sweep/summary.csv \
        --scenario-dir results/scenarios/ \
        --src-flops 312 --src-bw 2039 --src-link-latency 5 \
        --tgt-flops 100 --tgt-bw 680 --tgt-link-latency 26 \
        --name "ICN" --tp-size 2 \
        -o scaled_icn.json

    # Compare multiple targets (run multiple times, then plot)
    python scale_report.py ... --name "ICN" -o icn.json
    python scale_report.py ... --name "PCIe" -o pcie.json
    python scale_report.py --plot icn.json pcie.json -o comparison.png
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


def scale_ttft(ttft_src, src_flops, tgt_flops,
               src_link_bw=0, tgt_link_bw=0, prefill_comm_frac=0):
    """Scale TTFT: compute scales with FLOPS, prefill comm scales with link BW.
    ttft = compute + prefill_comm
    compute_tgt = compute_src × (src_flops / tgt_flops)
    comm_tgt = comm_src × (src_link_bw / tgt_link_bw)
    """
    if tgt_flops <= 0 or src_flops <= 0:
        return ttft_src
    compute = ttft_src * (1 - prefill_comm_frac)
    comm_src = ttft_src * prefill_comm_frac
    compute_tgt = compute * (src_flops / tgt_flops)
    if prefill_comm_frac > 0 and src_link_bw > 0 and tgt_link_bw > 0:
        comm_tgt = comm_src * (src_link_bw / tgt_link_bw)
    else:
        comm_tgt = comm_src
    return compute_tgt + comm_tgt


def scale_tpot(tpot_src_ms, src_bw, tgt_bw,
               src_ar_lat_us, tgt_ar_lat_us, n_ar_per_step):
    """Scale TPOT (ms).
    src/tgt_ar_lat_us: single all-reduce latency (us)
    n_ar_per_step: number of AR calls per decode step (= num_layers × 2)

    tpot = ddr_time + n_ar × ar_latency
    ddr_time scales with DDR BW, ar_latency replaced by target value.
    """
    if src_bw <= 0 or tgt_bw <= 0:
        return tpot_src_ms
    src_comm_ms = src_ar_lat_us * n_ar_per_step / 1000
    tgt_comm_ms = tgt_ar_lat_us * n_ar_per_step / 1000
    ddr_time = max(tpot_src_ms - src_comm_ms, 0)
    return ddr_time * (src_bw / tgt_bw) + tgt_comm_ms


def parse_log(path):
    """Extract median TTFT/TPOT from vllm bench serve log."""
    ttft = tpot = None
    with open(path) as f:
        for line in f:
            ll = line.lower().strip()
            if "median" in ll and "ttft" in ll:
                try:
                    ttft = float(line.split(":")[-1].strip().replace("ms", "").strip())
                except ValueError:
                    pass
            if "median" in ll and ("tpot" in ll or "inter-token" in ll):
                try:
                    tpot = float(line.split(":")[-1].strip().replace("ms", "").strip())
                except ValueError:
                    pass
    return ttft, tpot


def fmt_ms(v):
    if v < 1:
        return f"{v * 1000:.1f}us"
    if v < 1000:
        return f"{v:.2f}ms"
    if v < 60000:
        return f"{v / 1000:.2f}s"
    if v < 3600000:
        return f"{v / 60000:.2f}min"
    return f"{v / 3600000:.2f}h"


def fmt_len(n):
    return f"{n // 1024}K" if n >= 1024 else str(n)


COLORS = ['#378ADD', '#D85A30', '#2CA02C', '#9467BD', '#8C564B', '#E377C2']
MARKERS = ['-o', '--^', '-s', '--D', '-v', '--P']


def do_scale(args):
    """Scale and save results to JSON."""
    pf_comm_frac = args.prefill_comm_fraction
    if args.tp_size > 1 and pf_comm_frac == 0 and args.src_link_bw > 0:
        pf_comm_frac = 0.05  # rough default for TP>1

    result = {
        "name": args.name,
        "src_flops": args.src_flops, "src_bw": args.src_bw,
        "src_link_latency_us": args.src_link_latency,
        "src_link_bw": args.src_link_bw,
        "tgt_flops": args.tgt_flops, "tgt_bw": args.tgt_bw,
        "tgt_link_latency_us": args.tgt_link_latency,
        "tgt_link_bw": args.tgt_link_bw,
        "prefill_comm_fraction": pf_comm_frac,
        "tp_size": args.tp_size,
    }

    print(f"Name: {args.name}")
    print(f"Source: {args.src_flops} TFLOPS, {args.src_bw} GB/s, "
          f"link_lat={args.src_link_latency}us, link_bw={args.src_link_bw} GB/s")
    print(f"Target: {args.tgt_flops} TFLOPS, {args.tgt_bw} GB/s, "
          f"link_lat={args.tgt_link_latency}us, link_bw={args.tgt_link_bw} GB/s")
    print(f"TTFT scale: compute x{args.src_flops / args.tgt_flops:.2f}, "
          f"prefill_comm_frac={pf_comm_frac:.2f}")
    print(f"TPOT DDR scale: x{args.src_bw / args.tgt_bw:.2f}")
    print()

    # Input sweep
    if args.input_csv:
        print("=== Input Sweep ===")
        print(f"{'input':>8} {'tgt_ttft':>10} {'tgt_tpot':>10} {'tgt_tps':>8}")
        print("-" * 40)

        sweep = []
        with open(args.input_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                il = int(row["input_len"])
                ttft = float(row["ttft_median_ms"]) if row.get("ttft_median_ms") else None
                tpot = float(row["tpot_median_ms"]) if row.get("tpot_median_ms") else None
                if ttft is None or tpot is None:
                    continue
                t_ttft = scale_ttft(ttft, args.src_flops, args.tgt_flops, args.src_link_bw, args.tgt_link_bw, pf_comm_frac)
                n_ar = args.num_layers * 2
                t_tpot = scale_tpot(tpot, args.src_bw, args.tgt_bw,
                                    args.src_link_latency, args.tgt_link_latency, n_ar)
                t_tps = 1000 / t_tpot if t_tpot > 0 else 0
                print(f"{il:>8} {fmt_ms(t_ttft):>10} {fmt_ms(t_tpot):>10} {t_tps:>7.1f}")
                sweep.append({
                    "input_len": il,
                    "ttft_ms": round(t_ttft, 2),
                    "tpot_ms": round(t_tpot, 3),
                    "tps": round(t_tps, 1),
                })
        result["input_sweep"] = sweep
        print()

    # Scenarios
    if args.scenario_dir:
        print("=== Scenarios ===")
        print(f"{'scenario':>20} {'tgt_ttft':>10} {'tgt_tpot':>10}")
        print("-" * 45)

        scenarios = []
        for log_name in sorted(os.listdir(args.scenario_dir)):
            if not log_name.endswith(".log"):
                continue
            ttft, tpot = parse_log(os.path.join(args.scenario_dir, log_name))
            if ttft is None or tpot is None:
                continue
            name = log_name.replace(".log", "")
            t_ttft = scale_ttft(ttft, args.src_flops, args.tgt_flops, args.src_link_bw, args.tgt_link_bw, pf_comm_frac)
            t_tpot = scale_tpot(tpot, args.src_bw, args.tgt_bw,
                                args.src_link_latency, args.tgt_link_latency,
                                args.num_layers * 2)
            print(f"{name:>20} {fmt_ms(t_ttft):>10} {fmt_ms(t_tpot):>10}")
            scenarios.append({
                "name": name,
                "ttft_ms": round(t_ttft, 2),
                "tpot_ms": round(t_tpot, 3),
            })
        result["scenarios"] = scenarios

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.output}")


def do_plot(args):
    """Plot comparison from multiple scaled JSON files."""
    datasets = []
    for path in args.plot:
        with open(path) as f:
            datasets.append(json.load(f))

    # Check if input_sweep data exists
    has_sweep = any("input_sweep" in d for d in datasets)
    has_scenarios = any("scenarios" in d for d in datasets)

    if has_sweep:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # TTFT
        ax = axes[0]
        for i, d in enumerate(datasets):
            sweep = d.get("input_sweep", [])
            if not sweep:
                continue
            ils = [s["input_len"] for s in sweep]
            ttfts = [s["ttft_ms"] for s in sweep]
            ax.plot(ils, ttfts, MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)], lw=2, ms=6, label=d["name"])
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        all_ils = sorted(set(s["input_len"] for d in datasets for s in d.get("input_sweep", [])))
        if all_ils:
            ax.set_xticks(all_ils)
            ax.set_xticklabels([fmt_len(il) for il in all_ils], rotation=45)
        ax.set_xlabel('Input Length (tokens)')
        ax.set_ylabel('TTFT (ms, log)')
        ax.set_title('TTFT Median')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(frameon=False)

        # TPOT
        ax = axes[1]
        for i, d in enumerate(datasets):
            sweep = d.get("input_sweep", [])
            if not sweep:
                continue
            ils = [s["input_len"] for s in sweep]
            tpots = [s["tpot_ms"] for s in sweep]
            ax.plot(ils, tpots, MARKERS[i % len(MARKERS)],
                    color=COLORS[i % len(COLORS)], lw=2, ms=6, label=d["name"])
        ax.set_xscale('log', base=2)
        if all_ils:
            ax.set_xticks(all_ils)
            ax.set_xticklabels([fmt_len(il) for il in all_ils], rotation=45)
        ax.set_xlabel('Input Length (tokens)')
        ax.set_ylabel('TPOT (ms)')
        ax.set_title('TPOT Median')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {args.output}")

    if has_scenarios:
        # Print comparison table
        print("\n=== Scenario Comparison ===")
        names = sorted(set(s["name"] for d in datasets for s in d.get("scenarios", [])))
        header = f"{'scenario':>20}"
        for d in datasets:
            header += f" | {d['name']:>12} TTFT {d['name']:>12} TPOT"
        print(header)
        print("-" * len(header))
        for sname in names:
            row = f"{sname:>20}"
            for d in datasets:
                sc = next((s for s in d.get("scenarios", []) if s["name"] == sname), None)
                if sc:
                    row += f" | {fmt_ms(sc['ttft_ms']):>12} {fmt_ms(sc['tpot_ms']):>12}"
                else:
                    row += f" | {'N/A':>12} {'N/A':>12}"
            print(row)


def main():
    ap = argparse.ArgumentParser(description="Scale benchmark results / plot comparison")

    # Mode 1: Scale
    ap.add_argument("--src-flops", type=float, default=0)
    ap.add_argument("--src-bw", type=float, default=0)
    ap.add_argument("--src-link-latency", type=float, default=0,
                    help="Source decode link latency us (e.g. NVLink AR ~5us)")
    ap.add_argument("--src-link-bw", type=float, default=0,
                    help="Source inter-chip bandwidth GB/s (for prefill comm scaling)")
    ap.add_argument("--tgt-flops", type=float, default=0)
    ap.add_argument("--tgt-bw", type=float, default=0)
    ap.add_argument("--tgt-link-latency", type=float, default=0,
                    help="Target decode link latency us (e.g. PCIe AR ~26us)")
    ap.add_argument("--tgt-link-bw", type=float, default=0,
                    help="Target inter-chip bandwidth GB/s (for prefill comm scaling)")
    ap.add_argument("--prefill-comm-fraction", type=float, default=0.0,
                    help="Fraction of TTFT spent on prefill communication (0-1)")
    ap.add_argument("--input-csv", default=None)
    ap.add_argument("--scenario-dir", default=None)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=40,
                    help="Model num_hidden_layers (for AR count = layers×2)")
    ap.add_argument("--name", default="Target",
                    help="Label for this configuration (e.g. 'ICN', 'PCIe')")

    # Mode 2: Plot comparison
    ap.add_argument("--plot", nargs="+", default=None,
                    help="JSON files from previous runs to compare (e.g. icn.json pcie.json)")

    ap.add_argument("-o", "--output", default="scaled_report.json")
    args = ap.parse_args()

    if args.plot:
        do_plot(args)
    elif args.src_flops > 0:
        do_scale(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
