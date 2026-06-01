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
    """Extract GEMM vs non-GEMM time from trace. Returns {gemm_ms, other_ms, total_ms}."""
    from trace_gemm_scale import find_kernel_table, find_nvtx_table, get_nvtx_range, GEMM_RE
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

    gemm_ns = sum(d for d, n in rows if GEMM_RE.search(n))
    other_ns = sum(d for d, n in rows if not GEMM_RE.search(n))

    return {
        "gemm_ms": gemm_ns / 1e6,
        "other_ms": other_ns / 1e6,
        "total_ms": (gemm_ns + other_ns) / 1e6,
    }


def scale_time(gemm_ms, other_ms, speedup):
    """Project time with GEMM speedup."""
    return gemm_ms / speedup + other_ms


def load_component(json_path=None, trace_path=None, nvtx_filter=None,
                   time_key="vit_ms", manual_ms=None):
    """Load component timing. Returns {gemm_ms, other_ms, total_ms}."""
    if manual_ms is not None:
        # Assume 70% GEMM (typical for transformer)
        return {"gemm_ms": manual_ms * 0.7, "other_ms": manual_ms * 0.3, "total_ms": manual_ms}

    if trace_path and os.path.exists(trace_path):
        result = analyze_trace(trace_path, nvtx_filter)
        if result:
            return result

    if json_path and os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        total = d.get("component_results", d).get(time_key, 0)
        # Without trace, assume 70% GEMM
        return {"gemm_ms": total * 0.7, "other_ms": total * 0.3, "total_ms": total}

    return None


def main():
    ap = argparse.ArgumentParser(description="VLA pipeline summary with precision scaling")
    # VIT
    ap.add_argument("--vit-json", default=None)
    ap.add_argument("--vit-trace", default=None)
    ap.add_argument("--vit-ms", type=float, default=None, help="Manual VIT time (ms)")
    ap.add_argument("--vit-precision", default="fp16", choices=PRECISION_SPEEDUP.keys())
    # LLM
    ap.add_argument("--llm-trace", default=None)
    ap.add_argument("--llm-ms", type=float, default=None, help="Manual LLM prefill time (ms)")
    ap.add_argument("--llm-precision", default="fp16", choices=PRECISION_SPEEDUP.keys())
    # DiT
    ap.add_argument("--dit-json", default=None)
    ap.add_argument("--dit-trace", default=None)
    ap.add_argument("--dit-ms", type=float, default=None, help="Manual DiT 50-step time (ms)")
    ap.add_argument("--dit-precision", default="fp16", choices=PRECISION_SPEEDUP.keys())
    # Options
    ap.add_argument("--dit-steps", type=int, default=10)
    ap.add_argument("--show-all-combos", action="store_true",
                    help="Show all FP16/FP8/FP4 combinations")
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
                dit = {k: dit_1step[k] * args.dit_steps for k in ("gemm_ms", "other_ms", "total_ms")}
    if not dit:
        dit = load_component(args.dit_json, None, None, "dit_full_ms", None)

    print("=" * 75)
    print("VLA Pipeline Summary")
    print("=" * 75)

    # Estimate FLOPs per component (from architecture params)
    # VIT: 24L×1024, per-cam 900 tokens self-attn, 900×3825 cross-attn
    vit_flops = (
        24 * (3*2*900*1024*1024 + 2*2*16*900*900*64 + 2*900*1024*1024 + 2*2*900*1024*4096)  # self-attn
        + 6 * (2*900*1024*1024 + 2*2*3825*1024*1024 + 2*2*16*900*3825*64 + 2*900*1024*1024)  # cross-attn
    ) * 6  # 6 cameras
    # LLM: 36L×2560, 1550 tokens prefill
    llm_flops = 36 * (3*2*1550*2560*2560 + 2*2*32*1550*1550*128 + 2*1550*2560*2560 + 2*2*1550*2560*9728)
    # DiT: 18L×1024, 51 query × 1550 KV, ×50 steps
    dit_1step_flops = 18 * (
        3*2*51*1024*1024 + 2*2*8*51*51*128 + 2*51*1024*1024  # self-attn
        + 2*51*1024*1024 + 2*2*1550*1024*1024 + 2*2*8*51*1550*128 + 2*51*1024*1024  # cross-attn
        + 2*2*51*1024*4096  # FFN
    ) + 2*1550*2560*1024  # kv_proj
    dit_flops = dit_1step_flops * args.dit_steps

    comp_flops = {"VIT": vit_flops, "LLM": llm_flops, "DiT": dit_flops}
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
    print(f"\n{'Comp':<6s} {'GEMM':>7s} {'Other':>7s} {'Total':>8s} {'GEMM%':>5s} "
          f"{'GFLOPs':>7s} {'TFLOPS':>6s} {'MFU':>5s} {'WeightMB':>8s} {'BW%':>5s}")
    print("-" * 80)
    components = {"VIT": vit, "LLM": llm, "DiT": dit}
    total_ms = 0
    for name, c in components.items():
        if c:
            pct = c["gemm_ms"] / c["total_ms"] * 100 if c["total_ms"] > 0 else 0
            gflops = comp_flops[name] / 1e9
            tflops = comp_flops[name] / 1e12 / (c["total_ms"] / 1000) if c["total_ms"] > 0 else 0
            mfu = tflops / peak * 100 if peak > 0 else 0
            wb = comp_weight_bytes[name]
            bw_util = (wb / 1e9) / (c["total_ms"] / 1000) / peak_bw * 100 if c["total_ms"] > 0 and peak_bw > 0 else 0
            print(f"{name:<6s} {c['gemm_ms']:>7.1f} {c['other_ms']:>7.1f} {c['total_ms']:>8.1f} {pct:>4.0f}% "
                  f"{gflops:>7.0f} {tflops:>6.1f} {mfu:>4.1f}% {wb/1e6:>8.0f} {bw_util:>4.0f}%")
            total_ms += c["total_ms"]
        else:
            print(f"{name:<6s} {'N/A':>7s} {'N/A':>7s} {'N/A':>8s}")
    print("-" * 80)
    total_flops = sum(comp_flops.values())
    total_tflops = total_flops / 1e12 / (total_ms / 1000) if total_ms > 0 else 0
    total_mfu = total_tflops / peak * 100 if peak > 0 else 0
    print(f"{'TOTAL':<6s} {'':>7s} {'':>7s} {total_ms:>8.1f} {'':>5s} "
          f"{total_flops/1e9:>7.0f} {total_tflops:>6.1f} {total_mfu:>4.1f}%")
    if total_ms > 0:
        print(f"{'FPS':<6s} {'':>7s} {'':>7s} {1000/total_ms:>8.1f}")

    print(f"\n  (peak: {peak} TFLOPS, {peak_bw} GB/s)")

    # Selected precision
    print(f"\n{'='*75}")
    print(f"Selected Precision: VIT={args.vit_precision} LLM={args.llm_precision} DiT={args.dit_precision}")
    print("=" * 75)

    scaled = {}
    for name, c, prec in [("VIT", vit, args.vit_precision),
                           ("LLM", llm, args.llm_precision),
                           ("DiT", dit, args.dit_precision)]:
        if c:
            sp = PRECISION_SPEEDUP[prec]
            t = scale_time(c["gemm_ms"], c["other_ms"], sp)
            scaled[name] = t
            speedup = c["total_ms"] / t if t > 0 else 0
            print(f"  {name}: {c['total_ms']:.2f}ms → {t:.2f}ms ({speedup:.2f}x with {prec})")
        else:
            scaled[name] = 0

    scaled_total = sum(scaled.values())
    print(f"\n  Pipeline: {total_ms:.2f}ms → {scaled_total:.2f}ms")
    if scaled_total > 0:
        print(f"  FPS: {1000/total_ms:.1f} → {1000/scaled_total:.1f}")

    # All combos table
    if args.show_all_combos:
        precisions = ["fp16", "fp8", "fp4"]
        print(f"\n{'='*75}")
        print("All Precision Combinations")
        print("=" * 75)
        print(f"{'VIT':>6s} {'LLM':>6s} {'DiT':>6s}  {'VIT(ms)':>8s} {'LLM(ms)':>8s} {'DiT(ms)':>8s}  {'Total':>8s} {'FPS':>6s} {'vs base':>8s}")
        print("-" * 75)

        base_total = total_ms
        for vp in precisions:
            for lp in precisions:
                for dp in precisions:
                    vt = scale_time(vit["gemm_ms"], vit["other_ms"], PRECISION_SPEEDUP[vp]) if vit else 0
                    lt = scale_time(llm["gemm_ms"], llm["other_ms"], PRECISION_SPEEDUP[lp]) if llm else 0
                    dt = scale_time(dit["gemm_ms"], dit["other_ms"], PRECISION_SPEEDUP[dp]) if dit else 0
                    tt = vt + lt + dt
                    fps = 1000 / tt if tt > 0 else 0
                    sp = base_total / tt if tt > 0 else 0
                    print(f"{vp:>6s} {lp:>6s} {dp:>6s}  {vt:>8.2f} {lt:>8.2f} {dt:>8.2f}  {tt:>8.2f} {fps:>6.1f} {sp:>7.2f}x")

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
