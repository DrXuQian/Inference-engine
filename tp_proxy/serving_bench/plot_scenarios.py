#!/usr/bin/env python3
"""
Plot scenario results directly from benchmark logs (no scaling).

Draws horizontal stacked bars: Prefill (TTFT) + Decode (TPOT × tokens).
When --comm-json is provided, each phase is split into compute + comm.

Usage:
    # Single config
    python plot_scenarios.py \
        --log-dir ./serving_results/scenarios/ \
        --name "TP=2" \
        -o scenarios.png

    # With communication breakdown
    python plot_scenarios.py \
        --log-dir ./serving_results/scenarios/ \
        --name "TP=2" \
        --comm-json comm.json \
        -o scenarios.png

    # Compare multiple configs (each with its own comm.json)
    python plot_scenarios.py \
        --log-dir ./results_tp1/scenarios/ --name "TP=1" --comm-json none \
        --log-dir ./results_tp2/scenarios/ --name "TP=2" --comm-json tp2_comm.json \
        -o scenarios_compare.png

    # Custom decode tokens
    python plot_scenarios.py \
        --log-dir ./serving_results/scenarios/ --name "TP=2" \
        --decode-tokens mainstream=500,heavy_prefill=200,heavy_decode=10000 \
        -o scenarios.png
"""

import argparse
import json
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
    {"prefill": "#2166AC", "decode": "#67A9CF", "pf_comm": "#4393C3", "dec_comm": "#92C5DE", "edge": "#2166AC"},
    {"prefill": "#B2182B", "decode": "#EF8A62", "pf_comm": "#D6604D", "dec_comm": "#F4A582", "edge": "#B2182B"},
    {"prefill": "#1B7837", "decode": "#7FBF7B", "pf_comm": "#4DAC26", "dec_comm": "#A6D96A", "edge": "#1B7837"},
    {"prefill": "#762A83", "decode": "#C2A5CF", "pf_comm": "#9970AB", "dec_comm": "#D4B9DA", "edge": "#762A83"},
]


def load_comm(path):
    """Load comm.json → {decode_ms, prefill_ms}."""
    if not path or path.lower() == "none":
        return None
    with open(path) as f:
        c = json.load(f)
    return {
        "decode_ms": c.get("decode_total_per_step_ms", c.get("total_per_step_ms", 0)),
        "prefill_ms": c.get("prefill_total_per_step_ms",
                            c.get("decode_total_per_step_ms",
                                  c.get("total_per_step_ms", 0))),
    }


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
    ap.add_argument("--comm-json", action="append", default=None,
                    help="comm.json per config (use 'none' to skip). "
                         "If given once, applies to all configs.")
    ap.add_argument("--decode-tokens", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("-o", "--output", default="scenarios.png")
    args = ap.parse_args()

    if len(args.log_dir) != len(args.name):
        print("ERROR: --log-dir and --name count must match", file=sys.stderr)
        sys.exit(1)

    # Load comm data per config
    comm_list = [None] * len(args.log_dir)
    if args.comm_json:
        if len(args.comm_json) == 1:
            c = load_comm(args.comm_json[0])
            comm_list = [c] * len(args.log_dir)
        elif len(args.comm_json) == len(args.log_dir):
            comm_list = [load_comm(p) for p in args.comm_json]
        else:
            print("ERROR: --comm-json count must be 1 or match --log-dir count",
                  file=sys.stderr)
            sys.exit(1)

    has_comm = any(c is not None for c in comm_list)

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
            comm = comm_list[ci]

            if comm:
                # 4-segment bar: [Pf compute | Pf comm | Dec compute | Dec comm]
                pf_comm = min(comm["prefill_ms"], ttft)
                pf_compute = ttft - pf_comm
                dec_comm_per_tok = min(comm["decode_ms"], tpot)
                dec_compute_per_tok = tpot - dec_comm_per_tok
                dec_comm_total = dec_comm_per_tok * n_tok
                dec_compute_total = dec_compute_per_tok * n_tok

                left = 0
                # Prefill compute
                ax.barh(y, pf_compute / 1000, left=left / 1000, height=bar_height,
                        color=colors["prefill"], edgecolor='white', linewidth=0.5)
                left += pf_compute
                # Prefill comm
                ax.barh(y, pf_comm / 1000, left=left / 1000, height=bar_height,
                        color=colors["pf_comm"], edgecolor='white', linewidth=0.5,
                        hatch='///', alpha=0.85)
                left += pf_comm
                # Decode compute
                ax.barh(y, dec_compute_total / 1000, left=left / 1000, height=bar_height,
                        color=colors["decode"], edgecolor='white', linewidth=0.5)
                left += dec_compute_total
                # Decode comm
                ax.barh(y, dec_comm_total / 1000, left=left / 1000, height=bar_height,
                        color=colors["dec_comm"], edgecolor='white', linewidth=0.5,
                        hatch='///', alpha=0.85)
                # Outline
                ax.barh(y, total / 1000, height=bar_height,
                        fill=False, edgecolor=colors["edge"], linewidth=1.5)

                # Text labels (show comm time explicitly)
                pf_total_s = ttft / 1000
                if pf_total_s > total / 1000 * 0.08:
                    ax.text(ttft / 1000 / 2, y,
                            f"Prefill {fmt_time(ttft)}\ncomm {fmt_time(pf_comm)}",
                            ha='center', va='center', fontsize=7, color='white',
                            fontweight='bold')
                dec_total_s = decode_time / 1000
                if dec_total_s > total / 1000 * 0.08:
                    ax.text(ttft / 1000 + decode_time / 1000 / 2, y,
                            f"Decode {fmt_time(decode_time)}\ncomm {fmt_time(dec_comm_total)}",
                            ha='center', va='center', fontsize=7, color='white',
                            fontweight='bold')
            else:
                # Original 2-segment bar
                ax.barh(y, ttft / 1000, height=bar_height,
                        color=colors["prefill"], edgecolor='white', linewidth=0.5)
                ax.barh(y, decode_time / 1000, left=ttft / 1000, height=bar_height,
                        color=colors["decode"], edgecolor='white', linewidth=0.5)
                ax.barh(y, total / 1000, height=bar_height,
                        fill=False, edgecolor=colors["edge"], linewidth=1.5)

                if ttft / 1000 > total / 1000 * 0.08:
                    ax.text(ttft / 1000 / 2, y, f"Prefill\n{fmt_time(ttft)}",
                            ha='center', va='center', fontsize=8, color='white',
                            fontweight='bold')
                if decode_time / 1000 > total / 1000 * 0.08:
                    ax.text(ttft / 1000 + decode_time / 1000 / 2, y,
                            f"Decode\n{fmt_time(decode_time)}",
                            ha='center', va='center', fontsize=8, color='white',
                            fontweight='bold')

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

    # Legend
    if has_comm:
        handles = [
            mpatches.Patch(facecolor='#555', label='Prefill compute'),
            mpatches.Patch(facecolor='#777', hatch='///', label='Prefill comm'),
            mpatches.Patch(facecolor='#AAA', label='Decode compute'),
            mpatches.Patch(facecolor='#CCC', hatch='///', label='Decode comm'),
        ]
    else:
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
