#!/usr/bin/env python3
"""
Plot scenario comparison between different interconnect configurations.

Reads scaled JSON files (from scale_report.py) and draws horizontal
stacked bar charts: Prefill (TTFT) + Decode (TPOT × tokens).

Usage:
    python plot_scenario_compare.py \
        --json icn.json pcie.json \
        --model "Qwen3.5-122B-A10B" \
        --tp-size 2 \
        -o comparison.png

    # Custom scenario decode tokens:
    python plot_scenario_compare.py \
        --json icn.json pcie.json \
        --decode-tokens mainstream=500,heavy_prefill=200,heavy_decode=10000 \
        -o comparison.png
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Default decode token counts per scenario
DEFAULT_DECODE_TOKENS = {
    "mainstream": 500,
    "heavy_prefill": 200,
    "heavy_decode": 10000,
}

# Scenario display info
SCENARIO_INFO = {
    "mainstream": {
        "title": "Main Scenario:  Prompt=67K, Cache=80%, Prefill=13.4K, Decode=500 tok",
    },
    "heavy_prefill": {
        "title": "Heavy Prefill:  Prompt=119K, Cache=33%, Prefill=79.7K, Decode=200 tok",
    },
    "heavy_decode": {
        "title": "Heavy Decode:  Prompt=28K, Cache=98%, Prefill=0.6K, Decode=10K tok",
    },
}


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
    ap = argparse.ArgumentParser(description="Plot scenario comparison (Prefill + Decode bars)")
    ap.add_argument("--json", nargs="+", required=True,
                    help="Scaled JSON files from scale_report.py (e.g. icn.json pcie.json)")
    ap.add_argument("--decode-tokens", default=None,
                    help="Override decode tokens: mainstream=500,heavy_prefill=200,heavy_decode=10000")
    ap.add_argument("--model", default="", help="Model name for title")
    ap.add_argument("--tp-size", type=int, default=2, help="TP size for title")
    ap.add_argument("--title", default=None, help="Override full title")
    ap.add_argument("-o", "--output", default="scenario_compare.png")
    args = ap.parse_args()

    # Parse decode tokens
    decode_tokens = dict(DEFAULT_DECODE_TOKENS)
    if args.decode_tokens:
        for pair in args.decode_tokens.split(","):
            k, v = pair.split("=")
            decode_tokens[k.strip()] = int(v.strip())

    # Load configs
    configs = []
    for path in args.json:
        with open(path) as f:
            configs.append(json.load(f))

    if len(configs) < 1:
        print("Need at least 1 JSON file", file=sys.stderr)
        sys.exit(1)

    # Collect all scenario names
    all_scenarios = []
    for cfg in configs:
        for sc in cfg.get("scenarios", []):
            if sc["name"] not in all_scenarios:
                all_scenarios.append(sc["name"])

    if not all_scenarios:
        print("No scenarios found in JSON files", file=sys.stderr)
        sys.exit(1)

    n_scenarios = len(all_scenarios)
    n_configs = len(configs)

    # Colors
    config_colors = [
        {"prefill": "#2166AC", "decode": "#67A9CF", "edge": "#2166AC"},  # blue
        {"prefill": "#B2182B", "decode": "#EF8A62", "edge": "#B2182B"},  # red
        {"prefill": "#1B7837", "decode": "#7FBF7B", "edge": "#1B7837"},  # green
        {"prefill": "#762A83", "decode": "#C2A5CF", "edge": "#762A83"},  # purple
    ]

    fig, axes = plt.subplots(n_scenarios, 1, figsize=(14, 3.5 * n_scenarios + 1))
    if n_scenarios == 1:
        axes = [axes]

    # Build title
    if args.title:
        fig_title = args.title
    else:
        parts = []
        if args.model:
            parts.append(f"{args.model} · {args.tp_size}-Die TP")
        for i, cfg in enumerate(configs):
            name = cfg.get("name", f"Config-{chr(65+i)}")
            flops = cfg.get("tgt_flops", "?")
            bw = cfg.get("tgt_bw", "?")
            parts.append(f"{name}: {flops}T/{bw}GB/s")
        fig_title = "  |  ".join(parts)

    fig.suptitle(fig_title, fontsize=12, fontweight='bold', y=0.98)

    for sc_idx, sc_name in enumerate(all_scenarios):
        ax = axes[sc_idx]
        n_tok = decode_tokens.get(sc_name, 500)

        # Scenario title
        info = SCENARIO_INFO.get(sc_name, {})
        sc_title = info.get("title", sc_name)
        ax.set_title(sc_title, fontsize=10, fontweight='bold', loc='left', color='#333')

        bar_height = 0.6
        y_positions = list(range(n_configs - 1, -1, -1))  # top to bottom
        totals = []

        for ci, cfg in enumerate(configs):
            sc_data = next((s for s in cfg.get("scenarios", []) if s["name"] == sc_name), None)
            if not sc_data:
                totals.append(None)
                continue

            ttft = sc_data["ttft_ms"]
            tpot = sc_data["tpot_ms"]
            decode_time = tpot * n_tok
            total = ttft + decode_time
            totals.append(total)

            colors = config_colors[ci % len(config_colors)]
            y = y_positions[ci]

            # Prefill bar
            ax.barh(y, ttft / 1000, height=bar_height,
                    color=colors["prefill"], edgecolor='white', linewidth=0.5,
                    label='Prefill (TTFT)' if sc_idx == 0 and ci == 0 else "")
            # Decode bar
            ax.barh(y, decode_time / 1000, left=ttft / 1000, height=bar_height,
                    color=colors["decode"], edgecolor='white', linewidth=0.5,
                    label='Decode' if sc_idx == 0 and ci == 0 else "")

            # Add hatching for "simulated"
            ax.barh(y, total / 1000, height=bar_height,
                    fill=False, edgecolor=colors["edge"], linewidth=1.5,
                    hatch='///', alpha=0.3)

            # Text labels inside bars
            if ttft / 1000 > total / 1000 * 0.08:  # only if prefill bar wide enough
                ax.text(ttft / 1000 / 2, y, f"Prefill\n{fmt_time(ttft)}",
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
            if decode_time / 1000 > total / 1000 * 0.08:
                ax.text(ttft / 1000 + decode_time / 1000 / 2, y,
                        f"Decode\n{fmt_time(decode_time)}",
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')

            # Config label
            name = cfg.get("name", f"Config-{chr(65+ci)}")
            flops = cfg.get("tgt_flops", "?")
            bw = cfg.get("tgt_bw", "?")
            label = f"{name}\n({flops}T, {bw}GB/s)"
            ax.text(-0.02, y, label, ha='right', va='center', fontsize=8,
                    transform=ax.get_yaxis_transform(), fontweight='bold',
                    color=colors["prefill"])

        # Total time labels + delta
        max_total = max((t for t in totals if t), default=0)
        ref_total = totals[0] if totals[0] else None

        for ci, total in enumerate(totals):
            if total is None:
                continue
            y = y_positions[ci]
            colors = config_colors[ci % len(config_colors)]

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

        # Dashed reference line at first config's total
        if ref_total:
            ax.axvline(x=ref_total / 1000, color='#999', linestyle='--',
                       linewidth=0.8, alpha=0.5)

        ax.set_yticks([])
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, max_total / 1000 * 1.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # Legend (only on first subplot)
    if n_scenarios > 0:
        handles = [
            mpatches.Patch(facecolor='#555', label='Prefill (TTFT)'),
            mpatches.Patch(facecolor='#AAA', label='Decode'),
            mpatches.Patch(facecolor='none', edgecolor='#555', hatch='///', label='Simulated'),
        ]
        axes[0].legend(handles=handles, loc='upper right', frameon=True,
                       fontsize=8, framealpha=0.9)

    plt.tight_layout(rect=[0.12, 0, 1, 0.95])
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {args.output}")

    # Also save data summary
    summary_path = args.output.replace('.png', '_summary.json')
    summary = []
    for sc_name in all_scenarios:
        n_tok = decode_tokens.get(sc_name, 500)
        row = {"scenario": sc_name, "decode_tokens": n_tok}
        for ci, cfg in enumerate(configs):
            sc_data = next((s for s in cfg.get("scenarios", []) if s["name"] == sc_name), None)
            if sc_data:
                total = sc_data["ttft_ms"] + sc_data["tpot_ms"] * n_tok
                row[cfg.get("name", f"config_{ci}")] = {
                    "ttft_ms": sc_data["ttft_ms"],
                    "tpot_ms": sc_data["tpot_ms"],
                    "decode_ms": sc_data["tpot_ms"] * n_tok,
                    "total_ms": round(total, 2),
                }
        summary.append(row)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
