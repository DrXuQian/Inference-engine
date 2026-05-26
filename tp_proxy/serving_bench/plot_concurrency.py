#!/usr/bin/env python3
"""
Plot TTFT, TPOT, and throughput vs concurrency from serving benchmark logs.

Usage:
    python plot_concurrency.py --log-dir results/concurrency/ -o concurrency.png
    python plot_concurrency.py --log-dir results/concurrency/ --label "35B-A3B"
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(path: str) -> dict:
    """Extract metrics from vllm bench serve log."""
    metrics = {}
    with open(path) as f:
        text = f.read()

    # Try various output formats
    for line in text.split("\n"):
        ll = line.lower().strip()
        # Median TTFT
        if "median ttft" in ll or ("ttft" in ll and "median" in ll):
            m = re.search(r'([\d.]+)\s*ms', line)
            if m:
                metrics["ttft_median"] = float(m.group(1))
        # Median TPOT / Inter-token
        if ("median" in ll and "inter-token" in ll) or ("median" in ll and "tpot" in ll):
            m = re.search(r'([\d.]+)\s*ms', line)
            if m:
                metrics["tpot_median"] = float(m.group(1))
        # Throughput
        if "request throughput" in ll:
            m = re.search(r'([\d.]+)', line.split(":")[-1])
            if m:
                metrics["throughput_rps"] = float(m.group(1))
        if "output token throughput" in ll or "token throughput" in ll:
            m = re.search(r'([\d.]+)', line.split(":")[-1])
            if m:
                metrics["token_throughput"] = float(m.group(1))

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True, help="Directory with c1.log, c2.log, ...")
    ap.add_argument("--label", default="", help="Label for the plot")
    ap.add_argument("-o", "--output", default="concurrency_sweep.png")
    args = ap.parse_args()

    # Find all c*.log files
    logs = sorted(glob.glob(os.path.join(args.log_dir, "c*.log")))
    if not logs:
        print(f"No c*.log files found in {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    concurrencies = []
    ttfts = []
    tpots = []
    throughputs = []
    token_tps = []

    for log_path in logs:
        fname = os.path.basename(log_path)
        m = re.match(r'c(\d+)\.log', fname)
        if not m:
            continue
        c = int(m.group(1))
        metrics = parse_log(log_path)

        concurrencies.append(c)
        ttfts.append(metrics.get("ttft_median"))
        tpots.append(metrics.get("tpot_median"))
        throughputs.append(metrics.get("throughput_rps"))
        token_tps.append(metrics.get("token_throughput"))

    if not concurrencies:
        print("No valid data found", file=sys.stderr)
        sys.exit(1)

    # Sort by concurrency
    order = sorted(range(len(concurrencies)), key=lambda i: concurrencies[i])
    concurrencies = [concurrencies[i] for i in order]
    ttfts = [ttfts[i] for i in order]
    tpots = [tpots[i] for i in order]
    throughputs = [throughputs[i] for i in order]
    token_tps = [token_tps[i] for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    title_suffix = f" ({args.label})" if args.label else ""
    color = '#378ADD'

    # --- TTFT ---
    ax = axes[0]
    valid = [(c, v) for c, v in zip(concurrencies, ttfts) if v is not None]
    if valid:
        x, y = zip(*valid)
        ax.plot(x, y, '-o', color=color, lw=2, ms=6)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('TTFT Median (ms)')
    ax.set_title(f'TTFT vs Concurrency{title_suffix}', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_xticks(concurrencies)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.grid(True, alpha=0.25)

    # --- TPOT ---
    ax = axes[1]
    valid = [(c, v) for c, v in zip(concurrencies, tpots) if v is not None]
    if valid:
        x, y = zip(*valid)
        ax.plot(x, y, '-o', color='#D85A30', lw=2, ms=6)
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('TPOT Median (ms)')
    ax.set_title(f'TPOT vs Concurrency{title_suffix}', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_xticks(concurrencies)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.grid(True, alpha=0.25)

    # --- Throughput ---
    ax = axes[2]
    valid_rps = [(c, v) for c, v in zip(concurrencies, throughputs) if v is not None]
    valid_tps = [(c, v) for c, v in zip(concurrencies, token_tps) if v is not None]
    if valid_tps:
        x, y = zip(*valid_tps)
        ax.plot(x, y, '-s', color='#2CA02C', lw=2, ms=6, label='Token/s')
    if valid_rps:
        ax2 = ax.twinx()
        x, y = zip(*valid_rps)
        ax2.plot(x, y, '--^', color='#9467BD', lw=1.5, ms=5, label='Req/s')
        ax2.set_ylabel('Request Throughput (req/s)', color='#9467BD')
    ax.set_xlabel('Concurrency')
    ax.set_ylabel('Token Throughput (tok/s)')
    ax.set_title(f'Throughput vs Concurrency{title_suffix}', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_xticks(concurrencies)
    ax.set_xticklabels([str(c) for c in concurrencies])
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', frameon=False)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
