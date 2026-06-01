#!/usr/bin/env python3
"""
VLA pipeline summary report with per-component precision scaling.

Reads VIT/DiT results (from vla_bench.py --output-json) and LLM trace,
generates a summary table with FP16/FP8/FP4 projections for each component.

Usage:
    # From bench results (JSON)
    python vla_report.py \
        --vit-json results/vit/results.json \
        --dit-json results/dit/results.json \
        --llm-ms 15.0

    # From trace sqlite (auto GEMM analysis)
    python vla_report.py \
        --vit-trace results/vit/trace.sqlite \
        --dit-trace results/dit/trace.sqlite \
        --llm-trace results/llm/trace.sqlite

    # Specify precision per component
    python vla_report.py \
        --vit-json results/vit/results.json --vit-precision fp8 \
        --dit-json results/dit/results.json --dit-precision fp4 \
        --llm-ms 15.0 --llm-precision fp8

    # All combos
    python vla_report.py \
        --vit-json results/vit/results.json \
        --dit-json results/dit/results.json \
        --llm-ms 15.0 \
        --show-all-combos
"""

import argparse
import json
import sys
import os

# Add parent dir for trace_gemm_scale
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


PRECISION_SPEEDUP = {
    "fp16": 1.0,
    "bf16": 1.0,
    "fp8": 2.0,
    "fp4": 4.0,
}


def analyze_trace(sqlite_path, nvtx_filter=None):
    """Extract GEMM/FA/Other time from trace. Returns {gemm_ms, fa_ms, other_ms, total_ms}."""
    from trace_gemm_scale import find_kernel_table, find_nvtx_table, get_nvtx_range, GEMM_RE, FA_RE
    import sqlite3

    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    kt = find_kernel_table(c)
    if not kt:
        conn.close()
        return None

    nvtx_start, nvtx_end = None, None
    if nvtx_filter:
        nvtx_table = find_nvtx_table(c)
        if nvtx_table:
            nvtx_start, nvtx_end = get_nvtx_range(c, nvtx_table, nvtx_filter)
        if not nvtx_start:
            print(f"  WARNING: NVTX '{nvtx_filter}' not found in trace, skipping")
            conn.close()
            return None

    query = f'''SELECT k."end" - k.start AS dur, s.value AS name
        FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id'''
    if nvtx_start and nvtx_end:
        query += f' WHERE k.start >= {nvtx_start} AND k."end" <= {nvtx_end}'

    c.execute(query)
    rows = c.fetchall()
    conn.close()

    if not rows:
        return None

    gemm_ns = fa_ns = other_ns = 0
    for d, n in rows:
        if FA_RE.search(n):
            fa_ns += d
        elif GEMM_RE.search(n):
            gemm_ns += d
        else:
            other_ns += d

    return {
        "gemm_ms": gemm_ns / 1e6,
        "fa_ms": fa_ns / 1e6,
        "other_ms": other_ns / 1e6,
        "total_ms": (gemm_ns + fa_ns + other_ns) / 1e6,
    }


def scale_time(c, gemm_speedup, fa_speedup=1.0):
    """Project time: GEMM and FA scale independently. Other unchanged."""
    return c["gemm_ms"] / gemm_speedup + c.get("fa_ms", 0) / fa_speedup + c["other_ms"]


def load_component(json_path=None, trace_path=None, nvtx_filter=None,
                   time_key="vit_ms", manual_ms=None):
    """Load component timing. Returns {gemm_ms, other_ms, total_ms}."""
    if manual_ms is not None:
        # Assume 50% GEMM, 20% FA, 30% Other (typical for transformer with FlashAttn)
        return {"gemm_ms": manual_ms * 0.5, "fa_ms": manual_ms * 0.2,
                "other_ms": manual_ms * 0.3, "total_ms": manual_ms}

    if trace_path and os.path.exists(trace_path):
        result = analyze_trace(trace_path, nvtx_filter)
        if result:
            return result

    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        total = d.get("component_results", d).get(time_key, 0)
        return {"gemm_ms": total * 0.5, "fa_ms": total * 0.2,
                "other_ms": total * 0.3, "total_ms": total}

    return None


def main():
    ap = argparse.ArgumentParser(description="VLA pipeline summary with precision scaling")
    # VIT
    ap.add_argument("--vit-json", default=None)
    ap.add_argument("--vit-trace", default=None)
    ap.add_argument("--vit-ms", type=float, default=None, help="Manual VIT time (ms)")
    # LLM
    ap.add_argument("--llm-trace", default=None)
    ap.add_argument("--llm-ms", type=float, default=None, help="Manual LLM prefill time (ms)")
    # DiT
    ap.add_argument("--dit-json", default=None)
    ap.add_argument("--dit-trace", default=None)
    ap.add_argument("--dit-ms", type=float, default=None, help="Manual DiT total time (ms)")
    # Options
    ap.add_argument("--dit-steps", type=int, default=10)
    ap.add_argument("--peak-tflops", type=float, default=500,
                    help="Peak BF16 tensor TFLOPS for MFU calculation")
    ap.add_argument("--peak-bw", type=float, default=680,
                    help="Peak memory bandwidth GB/s for BW utilization")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    # Load components
    print("Loading components from traces...")
    vit = load_component(args.vit_json, args.vit_trace, "VIT_0", "vit_ms", args.vit_ms)
    if vit:
        print(f"  VIT: {vit['total_ms']:.2f}ms (gemm={vit['gemm_ms']:.2f}, other={vit['other_ms']:.2f})")
    llm = load_component(None, args.llm_trace, "prefill", None, args.llm_ms)
    if llm:
        print(f"  LLM: {llm['total_ms']:.2f}ms (gemm={llm['gemm_ms']:.2f}, other={llm['other_ms']:.2f})")
    # DiT: use full 50-step NVTX range directly (kernel sum = real time)
    dit = None
    if args.dit_ms is not None:
        dit = load_component(None, None, None, None, args.dit_ms)
    elif args.dit_trace:
        # Try DiT_50steps_0 first (full run), then DiT_1step_0 × N
        dit = load_component(None, args.dit_trace, f"DiT_{args.dit_steps}steps_0", None, None)
        if dit:
            print(f"  DiT {args.dit_steps}steps: {dit['total_ms']:.2f}ms (from trace)")
        else:
            dit_1step = load_component(None, args.dit_trace, "DiT_1step_0", None, None)
            if dit_1step:
                print(f"  DiT 1step: {dit_1step['total_ms']:.2f}ms → ×{args.dit_steps} = {dit_1step['total_ms']*args.dit_steps:.2f}ms")
                dit = {k: dit_1step.get(k, 0) * args.dit_steps for k in ("gemm_ms", "fa_ms", "other_ms", "total_ms")}
    if not dit:
        dit = load_component(args.dit_json, None, None, "dit_full_ms", None)

    print("=" * 75)
    print("VLA Pipeline Summary")
    print("=" * 75)

    # Estimate FLOPs per sub-component
    # Transformer FLOPs helper: 2 × (QKV + Attn + Out + FFN) per layer
    def transformer_flops(batch, seq, layers, hidden, heads, head_dim, ffn_dim):
        per_layer = (
            3 * 2 * batch * seq * hidden * hidden      # QKV proj
            + 2 * 2 * batch * heads * seq * seq * head_dim  # QK^T + AV
            + 2 * batch * seq * hidden * hidden          # out proj
            + 2 * 2 * batch * seq * hidden * ffn_dim     # FFN
        )
        return per_layer * layers

    def cross_attn_flops(batch, seq_q, seq_kv, layers, hidden, heads, head_dim):
        per_layer = (
            2 * batch * seq_q * hidden * hidden          # Q proj
            + 2 * 2 * batch * seq_kv * hidden * hidden   # KV proj
            + 2 * 2 * batch * heads * seq_q * seq_kv * head_dim  # QK^T + AV
            + 2 * batch * seq_q * hidden * hidden        # out proj
        )
        return per_layer * layers

    # ①a VIT Target (Spatial): 3 cams × 900 tokens (225×4, pre-merge), 24L
    vit_target = transformer_flops(3, 900, 24, 1024, 16, 64, 4096)
    # ①b VIT Current (Spatial): 3 cams × 900 tokens, 24L
    vit_current = transformer_flops(3, 900, 24, 1024, 16, 64, 4096)
    # ①b+ Temporal Cross-Attention: Q=900/cam, KV=3825/cam (17f×225), 3cam, 6L
    # Per-camera independent, no FFN. QKV proj only for current frame.
    Nq_t, Nkv_t, cams_t = 900, 17*225, 3  # Q=900 patches, KV=3825 history
    vit_temporal = 6 * cams_t * (
        2 * Nq_t * 1024 * 1024               # Q proj
        + 2 * Nkv_t * 1024 * 1024            # K proj (history)
        + 2 * Nkv_t * 1024 * 1024            # V proj (history)
        + 2 * 16 * Nq_t * Nkv_t * 64         # QK^T
        + 2 * 16 * Nq_t * Nkv_t * 64         # softmax(QK^T) × V
        + 2 * Nq_t * 1024 * 1024             # out proj
    )
    vit_flops = vit_target + vit_current + vit_temporal

    # ② LLM Prefill: 1550 tokens, 36L, Qwen3-VL-4B
    # GQA: Q=32heads×128=4096, KV=8heads×128=1024
    # FFN: SwiGLU = 3 projections (gate + up + down) × 2560×9728
    llm_flops = 36 * (
        2 * 1550 * 2560 * 4096          # Q proj (2560→4096)
        + 2 * 1550 * 2560 * 1024        # K proj (2560→1024)
        + 2 * 1550 * 2560 * 1024        # V proj (2560→1024)
        + 2 * 32 * 1550 * 1550 * 128    # QK^T
        + 2 * 32 * 1550 * 1550 * 128    # AV
        + 2 * 1550 * 4096 * 2560        # out proj (4096→2560)
        + 3 * 2 * 1550 * 2560 * 9728    # FFN SwiGLU (gate+up+down)
    )

    # ③ DiT: 51 tokens, 18L, cross-attn with LLM KV (1550 tokens), ×N steps
    dit_1step_flops = 18 * (
        3 * 2 * 51 * 1024 * 1024 + 2 * 2 * 8 * 51 * 51 * 128 + 2 * 51 * 1024 * 1024  # self-attn
        + 2 * 51 * 1024 * 1024 + 2 * 2 * 1550 * 1024 * 1024 + 2 * 2 * 8 * 51 * 1550 * 128 + 2 * 51 * 1024 * 1024  # cross-attn
        + 2 * 2 * 51 * 1024 * 4096  # FFN
    ) + 2 * 1550 * 2560 * 1024  # kv_proj (once)
    dit_flops = dit_1step_flops * args.dit_steps

    comp_flops = {"VIT": vit_flops, "LLM": llm_flops, "DiT": dit_flops}

    # Print detailed FLOPs breakdown
    print(f"\nFLOPs Breakdown:")
    print(f"  {'Stage':<30s} {'Batch':>5s} {'Seq':>5s} {'Layers':>6s} {'Hidden':>6s} {'TFLOPs':>8s}")
    print(f"  {'-'*68}")
    print(f"  {'①a VIT Target (Spatial)':<30s} {'3':>5s} {'900':>5s} {'24':>6s} {'1024':>6s} {vit_target/1e12:>8.3f}")
    print(f"  {'①b VIT Current (Spatial)':<30s} {'3':>5s} {'900':>5s} {'24':>6s} {'1024':>6s} {vit_current/1e12:>8.3f}")
    print(f"  {'①b+ Temporal CrossAttn':<30s} {'Q=900':>5s} {'K=3825':>6s} {'6':>6s} {'1024':>6s} {vit_temporal/1e12:>8.3f}")
    print(f"  {'② LLM Prefill':<30s} {'1':>5s} {'1550':>5s} {'36':>6s} {'2560':>6s} {llm_flops/1e12:>8.3f}")
    print(f"  {'③ DiT (×' + str(args.dit_steps) + ')':<30s} {'1':>5s} {'51':>5s} {'18':>6s} {'1024':>6s} {dit_flops/1e12:>8.3f}")
    print(f"  {'-'*68}")
    total_est = vit_flops + llm_flops + dit_flops
    print(f"  {'Total':<30s} {'':>5s} {'':>5s} {'':>6s} {'':>6s} {total_est/1e12:>8.3f}")
    peak = args.peak_tflops
    peak_bw = args.peak_bw

    # Weight params per component (for BW utilization: bytes read from memory)
    # VIT: ~335M params, read once
    vit_params = 24*(3*1024*1024 + 1024*1024 + 2*1024*4096) + \
                 6*(3*1024*1024 + 1024*1024) + \
                 1024*4*1024 + 1024*2560 + 1024*1024
    # LLM: prefill is compute-bound, weight read once
    # Qwen3-4B GQA: Q=2560→4096, K=2560→1024, V=2560→1024, O=4096→2560, FFN=2×2560×9728
    llm_params = 36*(2560*4096 + 2*2560*1024 + 4096*2560 + 2*2560*9728)
    # DiT: ~305M params, weight re-read every denoising step
    dit_params = 18*(3*1024*1024 + 1024*1024 + 3*1024*1024 + 1024*1024 + 2*1024*4096) + \
                 2560*1024 + 62*1024 + 1024*62  # kv_proj + proj_in + proj_out

    comp_weight_bytes = {
        "VIT": vit_params * 2,                         # bf16, read once
        "LLM": llm_params * 2,                         # bf16, read once (prefill)
        "DiT": dit_params * 2 * args.dit_steps,        # bf16, ×50 steps
    }

    # Component breakdown
    print(f"\n{'Comp':<6s} {'GEMM':>7s} {'FA':>7s} {'Other':>7s} {'Total':>8s} "
          f"{'GFLOPs':>7s} {'TFLOPS':>6s} {'MFU':>5s} {'WtMB':>6s} {'BW%':>5s}")
    print("-" * 78)
    components = {"VIT": vit, "LLM": llm, "DiT": dit}
    total_ms = 0
    for name, c in components.items():
        if c:
            fa = c.get("fa_ms", 0)
            gflops = comp_flops[name] / 1e9
            tflops = comp_flops[name] / 1e12 / (c["total_ms"] / 1000) if c["total_ms"] > 0 else 0
            mfu = tflops / peak * 100 if peak > 0 else 0
            wb = comp_weight_bytes[name]
            bw_util = (wb / 1e9) / (c["total_ms"] / 1000) / peak_bw * 100 if c["total_ms"] > 0 and peak_bw > 0 else 0
            print(f"{name:<6s} {c['gemm_ms']:>7.1f} {fa:>7.1f} {c['other_ms']:>7.1f} {c['total_ms']:>8.1f} "
                  f"{gflops:>7.0f} {tflops:>6.1f} {mfu:>4.1f}% {wb/1e6:>6.0f} {bw_util:>4.0f}%")
            total_ms += c["total_ms"]
        else:
            print(f"{name:<6s} {'N/A':>7s} {'N/A':>7s} {'N/A':>7s} {'N/A':>8s}")
    print("-" * 78)
    total_flops = sum(comp_flops.values())
    total_tflops = total_flops / 1e12 / (total_ms / 1000) if total_ms > 0 else 0
    total_mfu = total_tflops / peak * 100 if peak > 0 else 0
    print(f"{'TOTAL':<6s} {'':>7s} {'':>7s} {'':>7s} {total_ms:>8.1f} "
          f"{total_flops/1e9:>7.0f} {total_tflops:>6.1f} {total_mfu:>4.1f}%")
    if total_ms > 0:
        print(f"{'FPS':<6s} {'':>7s} {'':>7s} {'':>7s} {1000/total_ms:>8.1f}")

    print(f"\n  (peak: {peak} TFLOPS, {peak_bw} GB/s)")

    # Recommended precision config
    # VIT: GEMM=FP8, FA=FP8 (entire VIT in FP8)
    # LLM: GEMM=FP4, FA=FP8
    # DiT: GEMM=FP16, FA=FP16 (all FP16)
    RECOMMENDED = {
        "VIT": {"gemm": "fp8",  "fa": "fp8",  "gemm_sp": 2.0, "fa_sp": 2.0},
        "LLM": {"gemm": "fp4",  "fa": "fp8",  "gemm_sp": 4.0, "fa_sp": 2.0},
        "DiT": {"gemm": "fp16", "fa": "fp16", "gemm_sp": 1.0, "fa_sp": 1.0},
    }

    print(f"\n{'='*75}")
    print("Recommended Precision")
    print("=" * 75)
    print(f"  {'Comp':<6s} {'GEMM':>6s} {'FA':>6s}  {'Before':>8s} {'After':>8s} {'Speedup':>8s}")
    print(f"  {'-'*48}")

    scaled = {}
    for name, c in components.items():
        if c:
            r = RECOMMENDED[name]
            t = scale_time(c, r["gemm_sp"], r["fa_sp"])
            scaled[name] = t
            sp = c["total_ms"] / t if t > 0 else 0
            print(f"  {name:<6s} {r['gemm']:>6s} {r['fa']:>6s}  {c['total_ms']:>7.2f}ms {t:>7.2f}ms {sp:>7.2f}x")
        else:
            scaled[name] = 0

    scaled_total = sum(scaled.values())
    print(f"  {'-'*48}")
    print(f"  {'TOTAL':<6s} {'':>6s} {'':>6s}  {total_ms:>7.2f}ms {scaled_total:>7.2f}ms {total_ms/scaled_total if scaled_total>0 else 0:>7.2f}x")
    if scaled_total > 0:
        print(f"  {'FPS':<6s} {'':>6s} {'':>6s}  {1000/total_ms:>7.1f}   {1000/scaled_total:>7.1f}")

    # Save
    if args.output_json:
        out = {
            "components": {
                name: {**c, "precision": prec, "scaled_ms": scaled.get(name, 0)}
                for name, c, prec in [("VIT", vit, args.vit_precision),
                                       ("LLM", llm, args.llm_precision),
                                       ("DiT", dit, args.dit_precision)]
                if c
            },
            "total_ms": total_ms,
            "scaled_total_ms": scaled_total,
            "fps": 1000 / total_ms if total_ms > 0 else 0,
            "scaled_fps": 1000 / scaled_total if scaled_total > 0 else 0,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {args.output_json}")


if __name__ == "__main__":
    main()
