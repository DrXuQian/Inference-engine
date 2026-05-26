#!/usr/bin/env python3
"""
Plot TTFT and TPOT vs input length from serving benchmark results.
Supports overlaying multiple configs (e.g. 1-Die vs 2-Die TP=2).

Usage:
    # Single config
    python plot_input_sweep.py --csv results/input_sweep/summary.csv --label "1-Die"

    # Compare two configs
    python plot_input_sweep.py \
        --csv results_1die/input_sweep/summary.csv --label "1-Die" \
        --csv results_tp2/input_sweep/summary.csv --label "2-Die TP=2" \
        -o ttft_tpot_comparison.png
"""

import argparse
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_csv(path: str) -> dict:
    """Load summary.csv → {input_len: {ttft_median, tpot_median, ...}}."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            il = int(row["input_len"])
            data[il] = {
                "ttft_median": float(row["ttft_median_ms"]) if row.get("ttft_median_ms") else None,
                "ttft_p99": float(row["ttft_p99_ms"]) if row.get("ttft_p99_ms") else None,
                "tpot_median": float(row["tpot_median_ms"]) if row.get("tpot_median_ms") else None,
                "tpot_p99": float(row["tpot_p99_ms"]) if row.get("tpot_p99_ms") else None,
            }
    return data


def fmt_len(n: int) -> str:
    if n >= 1024:
        return f"{n // 1024}K"
    return str(n)


COLORS = ['#378ADD', '#D85A30', '#2CA02C', '#9467BD', '#8C564B', '#E377C2']
MARKERS = ['o', '^', 's', 'D', 'v', 'P']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="append", required=True, help="summary.csv path (can repeat)")
    ap.add_argument("--label", action="append", required=True, help="Label for each csv (same order)")
    ap.add_argument("-o", "--output", default="ttft_tpot_input_sweep.png")
    ap.add_argument("--tpot-ylim", type=float, nargs=2, default=None,
                    help="TPOT y-axis limits (e.g. 8 15)")
    args = ap.parse_args()

    if len(args.csv) != len(args.label):
        print("ERROR: --csv and --label count must match", file=sys.stderr)
        sys.exit(1)

    datasets = []
    for path, label in zip(args.csv, args.label):
        datasets.append((label, load_csv(path)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- TTFT (log-log) ---
    ax = axes[0]
    for i, (label, data) in enumerate(datasets):
        ils = sorted(data.keys())
        ttfts = [data[il]["ttft_median"] for il in ils]
        valid = [(il, t) for il, t in zip(ils, ttfts) if t is not None]
        if valid:
            x, y = zip(*valid)
            style = '-o' if i == 0 else '--' + MARKERS[i % len(MARKERS)]
            ax.plot(x, y, style, color=COLORS[i % len(COLORS)], lw=2, ms=6, label=label)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    all_ils = sorted(set(il for _, d in datasets for il in d))
    ax.set_xticks(all_ils)
    ax.set_xticklabels([fmt_len(il) for il in all_ils], rotation=45)
    ax.set_xlabel('Input Length (tokens)')
    ax.set_ylabel('TTFT (ms, log)')
    ax.set_title('TTFT Median', fontsize=12)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)

    # --- TPOT (log-x, linear-y) ---
    ax = axes[1]
    for i, (label, data) in enumerate(datasets):
        ils = sorted(data.keys())
        tpots = [data[il]["tpot_median"] for il in ils]
        valid = [(il, t) for il, t in zip(ils, tpots) if t is not None]
        if valid:
            x, y = zip(*valid)
            style = '-o' if i == 0 else '--' + MARKERS[i % len(MARKERS)]
            ax.plot(x, y, style, color=COLORS[i % len(COLORS)], lw=2, ms=6, label=label)

    ax.set_xscale('log', base=2)
    ax.set_xticks(all_ils)
    ax.set_xticklabels([fmt_len(il) for il in all_ils], rotation=45)
    ax.set_xlabel('Input Length (tokens)')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title('TPOT Median', fontsize=12)
    if args.tpot_ylim:
        ax.set_ylim(args.tpot_ylim)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)

    # Annotate gap at max input if 2+ datasets
    if len(datasets) >= 2:
        max_il = max(all_ils)
        vals = []
        for _, data in datasets:
            v = data.get(max_il, {}).get("tpot_median")
            if v is not None:
                vals.append(v)
        if len(vals) >= 2:
            gap = vals[-1] - vals[0]
            mid = (vals[0] + vals[-1]) / 2
            ax.annotate(f'gap = {gap:+.2f} ms',
                        xy=(max_il, mid),
                        xytext=(all_ils[-3] if len(all_ils) >= 3 else max_il // 2,
                                mid + abs(gap)),
                        fontsize=10, color='#555',
                        arrowprops=dict(arrowstyle='->', color='#888', lw=0.8))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
