#!/usr/bin/env python3
"""
Scale serving benchmark results from source platform to target platform.

Given source platform specs (FP16 TFLOPS, DDR BW, inter-chip BW) and
target platform specs, scale TTFT/TPOT accordingly:

  TTFT (prefill) is compute-bound:
    TTFT_target = TTFT_source × (src_flops / tgt_flops)

  TPOT (decode) is bandwidth-bound:
    TPOT = weight_time + kv_time + comm_time
    weight_time scales with DDR BW
    kv_time scales with DDR BW
    comm_time scales with inter-chip BW (for TP>1)

Usage:
    # Scale input sweep results
    python scale_report.py \
        --input-csv results/input_sweep/summary.csv \
        --src-flops 312 --src-bw 2039 --src-link-bw 600 \
        --tgt-flops 100 --tgt-bw 680 --tgt-link-bw 200 \
        --model-dir /path/to/model --tp-size 2 \
        -o report_scaled.png

    # Scale scenario results
    python scale_report.py \
        --scenario-dir results/scenarios/ \
        --src-flops 312 --src-bw 2039 --src-link-bw 600 \
        --tgt-flops 100 --tgt-bw 680 --tgt-link-bw 200 \
        -o scenario_scaled.txt
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_model_config(model_dir: str) -> dict | None:
    p = os.path.join(model_dir, "config.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    layer_types = tc.get("layer_types", [])
    n_layers = tc.get("num_hidden_layers", 0)
    n_full = sum(1 for lt in layer_types[:n_layers] if lt == "full_attention") if layer_types else n_layers
    return {
        "num_hidden_layers": n_layers,
        "num_key_value_heads": tc.get("num_key_value_heads", 2),
        "head_dim": tc.get("head_dim", 256),
        "n_full_attn_layers": n_full,
        "n_lin_attn_layers": n_layers - n_full,
        "linear_num_key_heads": tc.get("linear_num_key_heads", 0),
        "linear_key_head_dim": tc.get("linear_key_head_dim", 0),
    }


def compute_kv_fraction(cfg: dict, seq_len: int, tp_size: int,
                        total_decode_bytes: float) -> float:
    """Estimate what fraction of decode bandwidth is KV cache vs weights."""
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    n_full = cfg["n_full_attn_layers"]
    kv_heads_per_rank = n_kv if tp_size > n_kv else n_kv // tp_size

    full_kv = n_full * 2 * kv_heads_per_rank * head_dim * seq_len * 2
    lin_qk = cfg.get("linear_num_key_heads", 0)
    lin_kd = cfg.get("linear_key_head_dim", 0)
    lin_kv = cfg["n_lin_attn_layers"] * lin_qk * lin_kd * lin_kd * 2 if lin_qk > 0 else 0

    kv_total = full_kv + lin_kv
    if total_decode_bytes > 0:
        return kv_total / total_decode_bytes
    return 0.3  # fallback estimate


def scale_ttft(ttft_src: float, src_flops: float, tgt_flops: float) -> float:
    """Scale TTFT (compute-bound): inversely proportional to FLOPS."""
    if tgt_flops <= 0 or src_flops <= 0:
        return ttft_src
    return ttft_src * (src_flops / tgt_flops)


def scale_tpot(tpot_src: float, src_bw: float, tgt_bw: float,
               src_link: float, tgt_link: float,
               comm_fraction: float = 0.0) -> float:
    """Scale TPOT (bandwidth-bound): memory part scales with DDR BW,
    comm part scales with link BW.

    tpot = ddr_time + comm_time
    ddr_time = tpot * (1 - comm_fraction)
    comm_time = tpot * comm_fraction
    """
    if src_bw <= 0 or tgt_bw <= 0:
        return tpot_src
    ddr_time = tpot_src * (1 - comm_fraction) * (src_bw / tgt_bw)
    if comm_fraction > 0 and src_link > 0 and tgt_link > 0:
        comm_time = tpot_src * comm_fraction * (src_link / tgt_link)
    else:
        comm_time = tpot_src * comm_fraction
    return ddr_time + comm_time


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


def main():
    ap = argparse.ArgumentParser(description="Scale benchmark results to target platform")

    # Source platform
    ap.add_argument("--src-flops", type=float, required=True,
                    help="Source platform FP16 TFLOPS per GPU")
    ap.add_argument("--src-bw", type=float, required=True,
                    help="Source platform DDR bandwidth GB/s per GPU")
    ap.add_argument("--src-link-bw", type=float, default=0,
                    help="Source platform inter-chip bandwidth GB/s (e.g. NVLink)")

    # Target platform
    ap.add_argument("--tgt-flops", type=float, required=True,
                    help="Target platform FP16 TFLOPS per GPU")
    ap.add_argument("--tgt-bw", type=float, required=True,
                    help="Target platform DDR bandwidth GB/s per GPU")
    ap.add_argument("--tgt-link-bw", type=float, default=0,
                    help="Target platform inter-chip bandwidth GB/s")

    # Data inputs
    ap.add_argument("--input-csv", default=None,
                    help="Input sweep summary.csv from bench_input_sweep.sh")
    ap.add_argument("--scenario-dir", default=None,
                    help="Directory with scenario logs (mainstream.log, etc.)")
    ap.add_argument("--concurrency-dir", default=None,
                    help="Directory with concurrency logs (c1.log, etc.)")
    ap.add_argument("--model-dir", default=None,
                    help="Model config dir (for KV cache fraction estimation)")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--comm-fraction", type=float, default=0.0,
                    help="Fraction of TPOT spent on inter-chip communication (auto-estimated if model-dir provided)")

    ap.add_argument("-o", "--output", default="scaled_report.png")
    ap.add_argument("--src-label", default="Source")
    ap.add_argument("--tgt-label", default="Target")
    args = ap.parse_args()

    # Comm fraction for TP>1
    comm_frac = args.comm_fraction
    if args.tp_size > 1 and comm_frac == 0 and args.src_link_bw > 0:
        comm_frac = 0.05  # rough default: ~5% of TPOT is comm for TP=2

    print(f"Source: {args.src_flops} TFLOPS, {args.src_bw} GB/s DDR, {args.src_link_bw} GB/s link")
    print(f"Target: {args.tgt_flops} TFLOPS, {args.tgt_bw} GB/s DDR, {args.tgt_link_bw} GB/s link")
    print(f"TP={args.tp_size}, comm_fraction={comm_frac:.2f}")
    print(f"TTFT scale: ×{args.src_flops / args.tgt_flops:.2f}")
    print(f"TPOT DDR scale: ×{args.src_bw / args.tgt_bw:.2f}")
    if args.src_link_bw > 0 and args.tgt_link_bw > 0:
        print(f"TPOT link scale: ×{args.src_link_bw / args.tgt_link_bw:.2f}")
    print()

    # ========================================
    # Input sweep scaling
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

                t_ttft = scale_ttft(ttft, args.src_flops, args.tgt_flops)
                t_tpot = scale_tpot(tpot, args.src_bw, args.tgt_bw,
                                    args.src_link_bw, args.tgt_link_bw, comm_frac)

                src_tps = 1000 / tpot if tpot > 0 else 0
                tgt_tps = 1000 / t_tpot if t_tpot > 0 else 0

                print(f"{il:>8} {fmt_ms(ttft):>10} {fmt_ms(t_ttft):>10} "
                      f"{fmt_ms(tpot):>10} {fmt_ms(t_tpot):>10} "
                      f"{src_tps:>7.1f} {tgt_tps:>7.1f}")

                input_lens.append(il)
                src_ttfts.append(ttft); tgt_ttfts.append(t_ttft)
                src_tpots.append(tpot); tgt_tpots.append(t_tpot)

        # Save scaled CSV
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
        print(f"\nSaved: {csv_out}")

        # Plot
        if input_lens:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

            def fmt_len(n):
                return f"{n // 1024}K" if n >= 1024 else str(n)

            ax = axes[0]
            ax.plot(input_lens, src_ttfts, '-o', color='#378ADD', lw=2, ms=6,
                    label=args.src_label)
            ax.plot(input_lens, tgt_ttfts, '--^', color='#D85A30', lw=2, ms=7,
                    label=args.tgt_label)
            ax.set_xscale('log', base=2)
            ax.set_yscale('log')
            ax.set_xticks(input_lens)
            ax.set_xticklabels([fmt_len(il) for il in input_lens], rotation=45)
            ax.set_xlabel('Input Length (tokens)')
            ax.set_ylabel('TTFT (ms, log)')
            ax.set_title('TTFT Median', fontsize=12)
            ax.grid(True, which='both', alpha=0.25)
            ax.legend(frameon=False)

            ax = axes[1]
            ax.plot(input_lens, src_tpots, '-o', color='#378ADD', lw=2, ms=6,
                    label=args.src_label)
            ax.plot(input_lens, tgt_tpots, '--^', color='#D85A30', lw=2, ms=7,
                    label=args.tgt_label)
            ax.set_xscale('log', base=2)
            ax.set_xticks(input_lens)
            ax.set_xticklabels([fmt_len(il) for il in input_lens], rotation=45)
            ax.set_xlabel('Input Length (tokens)')
            ax.set_ylabel('TPOT (ms)')
            ax.set_title('TPOT Median', fontsize=12)
            ax.grid(True, which='both', alpha=0.25)
            ax.legend(frameon=False)

            if len(input_lens) >= 2:
                gap = tgt_tpots[-1] - src_tpots[-1]
                mid = (src_tpots[-1] + tgt_tpots[-1]) / 2
                ax.annotate(f'gap = {gap:+.2f} ms',
                            xy=(input_lens[-1], mid),
                            xytext=(input_lens[-3] if len(input_lens) >= 3 else input_lens[0], mid + abs(gap) * 0.5),
                            fontsize=10, color='#555',
                            arrowprops=dict(arrowstyle='->', color='#888', lw=0.8))

            plt.tight_layout()
            plt.savefig(args.output, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {args.output}")

    # ========================================
    # Scenario scaling
    # ========================================
    if args.scenario_dir:
        import re
        print("\n=== Scenario Scaling ===")
        print(f"{'scenario':>20} {'src_ttft':>10} {'tgt_ttft':>10} "
              f"{'src_tpot':>10} {'tgt_tpot':>10}")
        print("-" * 65)

        for log_name in sorted(os.listdir(args.scenario_dir)):
            if not log_name.endswith(".log"):
                continue
            log_path = os.path.join(args.scenario_dir, log_name)
            name = log_name.replace(".log", "")

            with open(log_path) as f:
                text = f.read()

            ttft = tpot = None
            for line in text.split("\n"):
                ll = line.lower().strip()
                if "median ttft" in ll or ("ttft" in ll and "median" in ll):
                    m = re.search(r'([\d.]+)\s*ms', line)
                    if m:
                        ttft = float(m.group(1))
                if ("median" in ll and "inter-token" in ll) or ("median" in ll and "tpot" in ll):
                    m = re.search(r'([\d.]+)\s*ms', line)
                    if m:
                        tpot = float(m.group(1))

            if ttft is not None and tpot is not None:
                t_ttft = scale_ttft(ttft, args.src_flops, args.tgt_flops)
                t_tpot = scale_tpot(tpot, args.src_bw, args.tgt_bw,
                                    args.src_link_bw, args.tgt_link_bw, comm_frac)
                print(f"{name:>20} {fmt_ms(ttft):>10} {fmt_ms(t_ttft):>10} "
                      f"{fmt_ms(tpot):>10} {fmt_ms(t_tpot):>10}")


if __name__ == "__main__":
    main()
