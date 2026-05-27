#!/usr/bin/env python3
"""
Plot scenario results directly from benchmark logs (no scaling).

Draws horizontal stacked bars: Prefill (TTFT) + Decode (TPOT × tokens).

Usage:
    # Single config
    python plot_scenarios.py \
        --log-dir ./serving_results/scenarios/ \
        --name "TP=2" \
        -o scenarios.png

    # Compare multiple configs
    python plot_scenarios.py \
        --log-dir ./results_tp1/scenarios/ --name "TP=1" \
        --log-dir ./results_tp2/scenarios/ --name "TP=2" \
        -o scenarios_compare.png

    # Custom decode tokens
    python plot_scenarios.py \
        --log-dir ./serving_results/scenarios/ --name "TP=2" \
        --decode-tokens mainstream=500,heavy_prefill=200,heavy_decode=10000 \
        -o scenarios.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


DEFAULT_DECODE_TOKENS = {
    "mainstream": 500,
    "heavy_prefill": 200,
    "heavy_decode": 10000,
}

SCENARIO_INFO = {
    "mainstream": "Main Scenario:  Prompt=67K, Cache=80%, Prefill=13.4K, Decode=500 tok",
    "heavy_prefill": "Heavy Prefill:  Prompt=119K, Cache=33%, Prefill=79.7K, Decode=200 tok",
    "heavy_decode": "Heavy Decode:  Prompt=28K, Cache=98%, Prefill=0.6K, Decode=10K tok",
}

CONFIG_COLORS = [
    {"prefill": "#2166AC", "decode": "#67A9CF", "edge": "#2166AC"},
    {"prefill": "#B2182B", "decode": "#EF8A62", "edge": "#B2182B"},
    {"prefill": "#1B7837", "decode": "#7FBF7B", "edge": "#1B7837"},
    {"prefill": "#762A83", "decode": "#C2A5CF", "edge": "#762A83"},
]


def parse_log(path):
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


def fmt_time(ms):
    s = ms / 1000
    if s < 1:
        return f"{ms:.0f}ms"
    if s < 60:
        return f"{s:.2f}s"
    if s < 3600:
        return f"{s / 60:.2f}min"
    return f"{s / 3600:.2f}h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", action="append", required=True,
                    help="Scenario log directory (can repeat for comparison)")
    ap.add_argument("--name", action="append", required=True,
                    help="Label for each log-dir (same order)")
    ap.add_argument("--decode-tokens", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("-o", "--output", default="scenarios.png")
    args = ap.parse_args()

    if len(args.log_dir) != len(args.name):
        print("ERROR: --log-dir and --name count must match", file=sys.stderr)
        sys.exit(1)

    decode_tokens = dict(DEFAULT_DECODE_TOKENS)
    if args.decode_tokens:
        for pair in args.decode_tokens.split(","):
            k, v = pair.split("=")
            decode_tokens[k.strip()] = int(v.strip())

    # Load data
    configs = []
    for log_dir, name in zip(args.log_dir, args.name):
        scenarios = {}
        for fname in sorted(os.listdir(log_dir)):
            if not fname.endswith(".log"):
                continue
            sc_name = fname.replace(".log", "")
            ttft, tpot = parse_log(os.path.join(log_dir, fname))
            if ttft is not None and tpot is not None:
                scenarios[sc_name] = {"ttft_ms": ttft, "tpot_ms": tpot}
        configs.append({"name": name, "scenarios": scenarios})

    # Fixed order: mainstream first
    SCENARIO_ORDER = ["mainstream", "heavy_prefill", "heavy_decode"]
    all_scenarios = [s for s in SCENARIO_ORDER
                     if any(s in cfg["scenarios"] for cfg in configs)]
    # Append any extra scenarios not in the predefined order
    for cfg in configs:
        for sc in cfg["scenarios"]:
            if sc not in all_scenarios:
                all_scenarios.append(sc)

    if not all_scenarios:
        print("No scenarios found", file=sys.stderr)
        sys.exit(1)

    n_sc = len(all_scenarios)
    n_cfg = len(configs)

    fig, axes = plt.subplots(n_sc, 1, figsize=(14, 3.5 * n_sc + 1))
    if n_sc == 1:
        axes = [axes]

    if args.title:
        fig.suptitle(args.title, fontsize=12, fontweight='bold', y=0.98)

    for sc_idx, sc_name in enumerate(all_scenarios):
        ax = axes[sc_idx]
        n_tok = decode_tokens.get(sc_name, 500)
        sc_title = SCENARIO_INFO.get(sc_name, sc_name)
        ax.set_title(sc_title, fontsize=10, fontweight='bold', loc='left', color='#333')

        bar_height = 0.6
        y_positions = list(range(n_cfg - 1, -1, -1))
        totals = []

        for ci, cfg in enumerate(configs):
            sc_data = cfg["scenarios"].get(sc_name)
            if not sc_data:
                totals.append(None)
                continue

            ttft = sc_data["ttft_ms"]
            tpot = sc_data["tpot_ms"]
            decode_time = tpot * n_tok
            total = ttft + decode_time
            totals.append(total)

            colors = CONFIG_COLORS[ci % len(CONFIG_COLORS)]
            y = y_positions[ci]

            ax.barh(y, ttft / 1000, height=bar_height,
                    color=colors["prefill"], edgecolor='white', linewidth=0.5)
            ax.barh(y, decode_time / 1000, left=ttft / 1000, height=bar_height,
                    color=colors["decode"], edgecolor='white', linewidth=0.5)
            ax.barh(y, total / 1000, height=bar_height,
                    fill=False, edgecolor=colors["edge"], linewidth=1.5)

            if ttft / 1000 > total / 1000 * 0.08:
                ax.text(ttft / 1000 / 2, y, f"Prefill\n{fmt_time(ttft)}",
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            if decode_time / 1000 > total / 1000 * 0.08:
                ax.text(ttft / 1000 + decode_time / 1000 / 2, y,
                        f"Decode\n{fmt_time(decode_time)}",
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

            ax.text(-0.02, y, f"{cfg['name']}",
                    ha='right', va='center', fontsize=9,
                    transform=ax.get_yaxis_transform(), fontweight='bold',
                    color=colors["prefill"])

        max_total = max((t for t in totals if t), default=0)
        ref_total = totals[0] if totals and totals[0] else None

        for ci, total in enumerate(totals):
            if total is None:
                continue
            y = y_positions[ci]
            colors = CONFIG_COLORS[ci % len(CONFIG_COLORS)]
            label = f"{fmt_time(total)}"
            if ci > 0 and ref_total:
                delta = total - ref_total
                pct = delta / ref_total * 100
                sign = "+" if delta >= 0 else ""
                label += f"  ({sign}{fmt_time(abs(delta))}, {sign}{pct:.1f}%)"
                color = '#B2182B' if delta > 0 else '#2166AC'
            else:
                color = colors["prefill"]
            ax.text(total / 1000 + max_total / 1000 * 0.02, y,
                    label, ha='left', va='center', fontsize=9,
                    fontweight='bold', color=color)

        if ref_total:
            ax.axvline(x=ref_total / 1000, color='#999', linestyle='--',
                       linewidth=0.8, alpha=0.5)

        ax.set_yticks([])
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, max_total / 1000 * 1.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    handles = [
        mpatches.Patch(facecolor='#555', label='Prefill (TTFT)'),
        mpatches.Patch(facecolor='#AAA', label='Decode'),
    ]
    axes[0].legend(handles=handles, loc='upper right', frameon=True, fontsize=8)

    plt.tight_layout(rect=[0.12, 0, 1, 0.95])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
