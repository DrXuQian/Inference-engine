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
    ap.add_argument("--dit-steps", type=int, default=50)
    ap.add_argument("--show-all-combos", action="store_true",
                    help="Show all FP16/FP8/FP4 combinations")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    # Load components
    vit = load_component(args.vit_json, args.vit_trace, "VIT_0", "vit_ms", args.vit_ms)
    llm = load_component(None, args.llm_trace, "prefill", None, args.llm_ms)
    # DiT: use single step from trace, multiply by dit_steps for total
    dit_1step = load_component(args.dit_json, args.dit_trace, "DiT_1step_0", "dit_1step_ms", None)
    if args.dit_ms is not None:
        dit = load_component(None, None, None, None, args.dit_ms)
    elif dit_1step:
        # Scale single step to full denoising
        factor = args.dit_steps
        dit = {
            "gemm_ms": dit_1step["gemm_ms"] * factor,
            "other_ms": dit_1step["other_ms"] * factor,
            "total_ms": dit_1step["total_ms"] * factor,
        }
    else:
        dit = load_component(args.dit_json, None, None, "dit_full_ms", None)

    print("=" * 75)
    print("VLA Pipeline Summary")
    print("=" * 75)

    # Component breakdown
    print(f"\n{'Component':<12s} {'GEMM(ms)':>10s} {'Other(ms)':>10s} {'Total(ms)':>10s} {'GEMM%':>7s}")
    print("-" * 55)
    components = {"VIT": vit, "LLM": llm, "DiT": dit}
    total_ms = 0
    for name, c in components.items():
        if c:
            pct = c["gemm_ms"] / c["total_ms"] * 100 if c["total_ms"] > 0 else 0
            print(f"{name:<12s} {c['gemm_ms']:>10.2f} {c['other_ms']:>10.2f} {c['total_ms']:>10.2f} {pct:>6.1f}%")
            total_ms += c["total_ms"]
        else:
            print(f"{name:<12s} {'N/A':>10s} {'N/A':>10s} {'N/A':>10s}")
    print("-" * 55)
    print(f"{'TOTAL':<12s} {'':>10s} {'':>10s} {total_ms:>10.2f}")
    if total_ms > 0:
        print(f"{'FPS':<12s} {'':>10s} {'':>10s} {1000/total_ms:>10.1f}")

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
