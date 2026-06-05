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


def calc_prefill_comm_ms(input_len, hidden_size, num_layers, link_bw_gbs):
    """Prefill comm from first principles (throughput-bound).
    Each layer has 2 AR ops. Message size = input_len × hidden_size × 2B (bf16).
    Time = num_layers × 2 × msg_bytes / (link_bw × 1e9) × 1000 ms.
    Returns 0 if link_bw not provided.
    """
    if link_bw_gbs <= 0 or hidden_size <= 0:
        return 0
    msg_bytes = input_len * hidden_size * 2  # bf16
    per_ar_s = msg_bytes / (link_bw_gbs * 1e9)
    return num_layers * 2 * per_ar_s * 1000  # ms


def calc_decode_comm_ms(num_layers, ar_latency_us):
    """Decode comm from first principles (latency-bound).
    Each layer has 2 AR ops. Time = num_layers × 2 × ar_latency_us / 1000.
    """
    if ar_latency_us <= 0:
        return 0
    return num_layers * 2 * ar_latency_us / 1000  # ms


def scale_ttft(ttft_src, src_flops, tgt_flops,
               src_prefill_comm_ms=0, tgt_prefill_comm_ms=0):
    """Scale TTFT: subtract src comm, scale compute with FLOPS, add tgt comm.
    Returns (ttft_tgt, tgt_prefill_comm_ms).
    """
    if tgt_flops <= 0 or src_flops <= 0:
        return ttft_src, tgt_prefill_comm_ms
    compute_src = max(ttft_src - src_prefill_comm_ms, 0)
    compute_tgt = compute_src * (src_flops / tgt_flops)
    return compute_tgt + tgt_prefill_comm_ms, tgt_prefill_comm_ms


def scale_tpot(tpot_src_ms, src_bw, tgt_bw,
               src_decode_comm_ms=0, tgt_decode_comm_ms=0):
    """Scale TPOT: subtract src comm, scale DDR time with BW, add tgt comm.
    Returns (tpot_tgt, tgt_decode_comm_ms).
    """
    if src_bw <= 0 or tgt_bw <= 0:
        return tpot_src_ms, tgt_decode_comm_ms
    ddr_time = max(tpot_src_ms - src_decode_comm_ms, 0)
    return ddr_time * (src_bw / tgt_bw) + tgt_decode_comm_ms, tgt_decode_comm_ms


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


DEFAULT_SCENARIO_INPUT_LENS = {
    "mainstream": 13400,
    "heavy_prefill": 79700,
    "heavy_decode": 600,
}


def do_scale(args):
    """Scale and save results to JSON."""
    nl = args.num_layers
    hs = args.hidden_size

    # Compute comm from first principles
    src_dec_comm = calc_decode_comm_ms(nl, args.src_link_latency)
    tgt_dec_comm = calc_decode_comm_ms(nl, args.tgt_link_latency)

    # Parse scenario input lengths
    sc_input_lens = dict(DEFAULT_SCENARIO_INPUT_LENS)
    if args.scenario_input_lens:
        for pair in args.scenario_input_lens.split(","):
            k, v = pair.split("=")
            sc_input_lens[k.strip()] = int(v.strip())

    result = {
        "name": args.name,
        "src_flops": args.src_flops, "src_bw": args.src_bw,
        "src_link_latency_us": args.src_link_latency,
        "src_link_bw": args.src_link_bw,
        "tgt_flops": args.tgt_flops, "tgt_bw": args.tgt_bw,
        "tgt_link_latency_us": args.tgt_link_latency,
        "tgt_link_bw": args.tgt_link_bw,
        "tp_size": args.tp_size,
        "num_layers": nl,
        "hidden_size": hs,
    }

    print(f"Name: {args.name}")
    print(f"Source: {args.src_flops} TFLOPS, {args.src_bw} GB/s, "
          f"AR_lat={args.src_link_latency}us, link_bw={args.src_link_bw} GB/s")
    print(f"Target: {args.tgt_flops} TFLOPS, {args.tgt_bw} GB/s, "
          f"AR_lat={args.tgt_link_latency}us, link_bw={args.tgt_link_bw} GB/s")
    print(f"Decode comm: src={fmt_ms(src_dec_comm)}/step  tgt={fmt_ms(tgt_dec_comm)}/step "
          f"({nl}L × 2AR × {args.tgt_link_latency}us)")
    if hs > 0:
        print(f"Prefill comm: {nl}L × 2AR × (input_len × {hs} × 2B) / link_bw")
    print()

    # Input sweep
    if args.input_csv:
        print("=== Input Sweep ===")
        print(f"{'input':>8} {'tgt_ttft':>10} {'tgt_tpot':>10} {'tgt_tps':>8}  {'pf_comm':>10} {'dec_comm':>10}")
        print("-" * 70)

        sweep = []
        with open(args.input_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                il = int(row["input_len"])
                ttft = float(row["ttft_median_ms"]) if row.get("ttft_median_ms") else None
                tpot = float(row["tpot_median_ms"]) if row.get("tpot_median_ms") else None
                if ttft is None or tpot is None:
                    continue
                src_pf_comm = calc_prefill_comm_ms(il, hs, nl, args.src_link_bw)
                tgt_pf_comm = calc_prefill_comm_ms(il, hs, nl, args.tgt_link_bw)
                t_ttft, _ = scale_ttft(ttft, args.src_flops, args.tgt_flops,
                                       src_pf_comm, tgt_pf_comm)
                t_tpot, _ = scale_tpot(tpot, args.src_bw, args.tgt_bw,
                                       src_dec_comm, tgt_dec_comm)
                t_tps = 1000 / t_tpot if t_tpot > 0 else 0
                print(f"{il:>8} {fmt_ms(t_ttft):>10} {fmt_ms(t_tpot):>10} {t_tps:>7.1f}  "
                      f"{fmt_ms(tgt_pf_comm):>10} {fmt_ms(tgt_dec_comm):>10}")
                sweep.append({
                    "input_len": il,
                    "ttft_ms": round(t_ttft, 2),
                    "tpot_ms": round(t_tpot, 3),
                    "tps": round(t_tps, 1),
                    "prefill_comm_ms": round(tgt_pf_comm, 3),
                    "decode_comm_ms": round(tgt_dec_comm, 4),
                })
        result["input_sweep"] = sweep
        print()

    # Scenarios
    if args.scenario_dir:
        print("=== Scenarios ===")
        print(f"{'scenario':>20} {'input':>8} {'tgt_ttft':>10} {'tgt_tpot':>10}  {'pf_comm':>10} {'dec_comm':>10}")
        print("-" * 80)

        scenarios = []
        for log_name in sorted(os.listdir(args.scenario_dir)):
            if not log_name.endswith(".log"):
                continue
            ttft, tpot = parse_log(os.path.join(args.scenario_dir, log_name))
            if ttft is None or tpot is None:
                continue
            name = log_name.replace(".log", "")
            input_len = sc_input_lens.get(name, 0)
            src_pf_comm = calc_prefill_comm_ms(input_len, hs, nl, args.src_link_bw)
            tgt_pf_comm = calc_prefill_comm_ms(input_len, hs, nl, args.tgt_link_bw)
            t_ttft, _ = scale_ttft(ttft, args.src_flops, args.tgt_flops,
                                   src_pf_comm, tgt_pf_comm)
            t_tpot, _ = scale_tpot(tpot, args.src_bw, args.tgt_bw,
                                   src_dec_comm, tgt_dec_comm)
            print(f"{name:>20} {input_len:>8} {fmt_ms(t_ttft):>10} {fmt_ms(t_tpot):>10}  "
                  f"{fmt_ms(tgt_pf_comm):>10} {fmt_ms(tgt_dec_comm):>10}")
            scenarios.append({
                "name": name,
                "ttft_ms": round(t_ttft, 2),
                "tpot_ms": round(t_tpot, 3),
                "prefill_comm_ms": round(tgt_pf_comm, 3),
                "decode_comm_ms": round(tgt_dec_comm, 4),
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
                    help="Target inter-chip bandwidth GB/s (for prefill comm, throughput-bound)")
    ap.add_argument("--input-csv", default=None)
    ap.add_argument("--scenario-dir", default=None)
    ap.add_argument("--scenario-input-lens", default=None,
                    help="Override input lens: mainstream=13400,heavy_prefill=79700,heavy_decode=600")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=40,
                    help="Model num_hidden_layers (AR count = layers×2)")
    ap.add_argument("--hidden-size", type=int, default=0,
                    help="Model hidden_size (for prefill AR msg = input_len×hidden×2B)")
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
