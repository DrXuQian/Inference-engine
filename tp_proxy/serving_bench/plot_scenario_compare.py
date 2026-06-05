#!/usr/bin/env python3
"""
Plot scenario comparison between different interconnect configurations.

Reads scaled JSON files (from scale_report.py) and draws horizontal
stacked bar charts: Prefill (TTFT) + Decode (TPOT × tokens).
When --comm-json is provided, each phase is split into compute + comm.

Usage:
    python plot_scenario_compare.py \
        --json icn.json pcie.json \
        --model "Qwen3.5-122B-A10B" \
        --tp-size 2 \
        -o comparison.png

    # With communication breakdown (one comm.json per JSON config):
    python plot_scenario_compare.py \
        --json icn.json pcie.json \
        --comm-json icn_comm.json pcie_comm.json \
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
    ap.add_argument("--comm-json", nargs="*", default=None,
                    help="comm.json per JSON config (use 'none' to skip one). "
                         "If one given, applies to all.")
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

    # Load comm data per config
    n_configs = len(configs)
    comm_list = [None] * n_configs
    if args.comm_json:
        if len(args.comm_json) == 1:
            c = load_comm(args.comm_json[0])
            comm_list = [c] * n_configs
        elif len(args.comm_json) == n_configs:
            comm_list = [load_comm(p) for p in args.comm_json]
        else:
            print("ERROR: --comm-json count must be 1 or match --json count",
                  file=sys.stderr)
            sys.exit(1)

    has_explicit_comm = any(c is not None for c in comm_list)

    # Check if JSONs contain embedded comm info
    has_json_comm = any(
        sc.get("prefill_comm_ms") or sc.get("decode_comm_ms")
        for cfg in configs for sc in cfg.get("scenarios", [])
    ) or any(
        cfg.get("tgt_decode_comm_ms", 0) > 0 or cfg.get("prefill_comm_fraction", 0) > 0
        for cfg in configs
    )
    has_comm = has_explicit_comm or has_json_comm

    # Fixed order: mainstream first
    SCENARIO_ORDER = ["mainstream", "heavy_prefill", "heavy_decode"]
    found = set()
    for cfg in configs:
        for sc in cfg.get("scenarios", []):
            found.add(sc["name"])
    all_scenarios = [s for s in SCENARIO_ORDER if s in found]
    for s in sorted(found):
        if s not in all_scenarios:
            all_scenarios.append(s)

    if not all_scenarios:
        print("No scenarios found in JSON files", file=sys.stderr)
        sys.exit(1)

    n_scenarios = len(all_scenarios)

    # Colors (with comm variants)
    config_colors = [
        {"prefill": "#2166AC", "decode": "#67A9CF", "pf_comm": "#4393C3", "dec_comm": "#92C5DE", "edge": "#2166AC"},
        {"prefill": "#B2182B", "decode": "#EF8A62", "pf_comm": "#D6604D", "dec_comm": "#F4A582", "edge": "#B2182B"},
        {"prefill": "#1B7837", "decode": "#7FBF7B", "pf_comm": "#4DAC26", "dec_comm": "#A6D96A", "edge": "#1B7837"},
        {"prefill": "#762A83", "decode": "#C2A5CF", "pf_comm": "#9970AB", "dec_comm": "#D4B9DA", "edge": "#762A83"},
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

        info = SCENARIO_INFO.get(sc_name, {})
        sc_title = info.get("title", sc_name)
        ax.set_title(sc_title, fontsize=10, fontweight='bold', loc='left', color='#333')

        bar_height = 0.6
        y_positions = list(range(n_configs - 1, -1, -1))
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

            # Resolve comm: --comm-json overrides, else read from JSON scenario
            comm = comm_list[ci]
            comm_source = "comm-json" if comm else None
            if not comm:
                sc_pf = sc_data.get("prefill_comm_ms")
                sc_dec = sc_data.get("decode_comm_ms")
                if sc_pf or sc_dec:
                    comm = {"prefill_ms": sc_pf or 0, "decode_ms": sc_dec or 0}
                    comm_source = "scenario"
            if not comm:
                tgt_dec_comm = cfg.get("tgt_decode_comm_ms", 0)
                pf_frac = cfg.get("prefill_comm_fraction", 0)
                if tgt_dec_comm > 0 or pf_frac > 0:
                    comm = {"prefill_ms": ttft * pf_frac, "decode_ms": tgt_dec_comm}
                    comm_source = "top-level"

            cfg_name = cfg.get("name", f"cfg{ci}")
            print(f"  [{cfg_name}] {sc_name}: ttft={ttft:.1f} tpot={tpot:.2f} "
                  f"comm_source={comm_source} "
                  f"comm={comm}")

            if comm:
                # 4-segment bar: [Pf compute | Pf comm | Dec compute | Dec comm]
                pf_comm = min(comm["prefill_ms"], ttft)
                pf_compute = ttft - pf_comm
                dec_comm_per_tok = min(comm["decode_ms"], tpot)
                dec_compute_per_tok = tpot - dec_comm_per_tok
                dec_comm_total = dec_comm_per_tok * n_tok
                dec_compute_total = dec_compute_per_tok * n_tok

                left = 0
                ax.barh(y, pf_compute / 1000, left=left / 1000, height=bar_height,
                        color=colors["prefill"], edgecolor='white', linewidth=0.5)
                left += pf_compute
                ax.barh(y, pf_comm / 1000, left=left / 1000, height=bar_height,
                        color=colors["pf_comm"], edgecolor='white', linewidth=0.5,
                        hatch='///', alpha=0.85)
                left += pf_comm
                ax.barh(y, dec_compute_total / 1000, left=left / 1000, height=bar_height,
                        color=colors["decode"], edgecolor='white', linewidth=0.5)
                left += dec_compute_total
                ax.barh(y, dec_comm_total / 1000, left=left / 1000, height=bar_height,
                        color=colors["dec_comm"], edgecolor='white', linewidth=0.5,
                        hatch='///', alpha=0.85)
                # Text labels with comm breakdown
                if ttft / 1000 > total / 1000 * 0.08:
                    ax.text(ttft / 1000 / 2, y,
                            f"Prefill {fmt_time(ttft)}\ncomm {fmt_time(pf_comm)}",
                            ha='center', va='center', fontsize=7, color='white',
                            fontweight='bold')
                if decode_time / 1000 > total / 1000 * 0.08:
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
                if ttft / 1000 > total / 1000 * 0.08:
                    ax.text(ttft / 1000 / 2, y, f"Prefill\n{fmt_time(ttft)}",
                            ha='center', va='center', fontsize=8, color='white',
                            fontweight='bold')
                if decode_time / 1000 > total / 1000 * 0.08:
                    ax.text(ttft / 1000 + decode_time / 1000 / 2, y,
                            f"Decode\n{fmt_time(decode_time)}",
                            ha='center', va='center', fontsize=8, color='white',
                            fontweight='bold')

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
    if n_scenarios > 0:
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
                ttft = sc_data["ttft_ms"]
                tpot = sc_data["tpot_ms"]
                total = ttft + tpot * n_tok
                entry = {
                    "ttft_ms": ttft,
                    "tpot_ms": tpot,
                    "decode_ms": tpot * n_tok,
                    "total_ms": round(total, 2),
                }
                comm = comm_list[ci] if ci < len(comm_list) else None
                if comm:
                    pf_comm = min(comm["prefill_ms"], ttft)
                    dec_comm = min(comm["decode_ms"], tpot)
                    entry["prefill_comm_ms"] = round(pf_comm, 3)
                    entry["decode_comm_per_tok_ms"] = round(dec_comm, 3)
                    entry["decode_comm_total_ms"] = round(dec_comm * n_tok, 2)
                    entry["total_comm_ms"] = round(pf_comm + dec_comm * n_tok, 2)
                row[cfg.get("name", f"config_{ci}")] = entry
        summary.append(row)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
