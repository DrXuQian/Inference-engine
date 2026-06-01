#!/usr/bin/env python3
"""
Analyze per-kernel-type bandwidth utilization from a single decode step.

Extracts one decode step (CUDA Graph region) from trace, then breaks down:
  - Marlin kernels (INT4 GEMV, attention projections)
  - moe_wna16_gemm kernels (MoE expert FFN)
  - Other kernels (layernorm, softmax, etc.)

For each type, computes:
  - Kernel time
  - Estimated weight bytes read
  - Bandwidth utilization = weight_bytes / kernel_time / peak_BW

Usage:
    python analyze_decode_bw.py trace.sqlite --peak-bw 680
    python analyze_decode_bw.py trace.sqlite --peak-bw 680 --tp 2
"""

import argparse
import re
import sqlite3
import sys
from collections import defaultdict


def find_kernel_table(c):
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for r in c.fetchall():
        if 'KERNEL' in r[0].upper() and 'ACTIVITY' in r[0].upper():
            return r[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite", help="trace sqlite file")
    ap.add_argument("--peak-bw", type=float, default=680, help="Peak BW GB/s")
    ap.add_argument("--tp", type=int, default=2, help="TP size")
    # Model params (Qwen3-30B-A3B defaults)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--q-heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--moe-intermediate", type=int, default=768)
    ap.add_argument("--num-experts", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--layers", type=int, default=48)
    args = ap.parse_args()

    conn = sqlite3.connect(args.sqlite)
    c = conn.cursor()
    kt = find_kernel_table(c)
    if not kt:
        print("ERROR: no kernel table found")
        sys.exit(1)

    # Check graphNodeId
    c.execute(f'PRAGMA table_info("{kt}")')
    cols = [col[1] for col in c.fetchall()]
    has_graph = "graphNodeId" in cols

    if not has_graph:
        print("WARNING: no graphNodeId column, cannot isolate decode steps")

    # Get all kernels
    c.execute(f'''
        SELECT k.start, k."end" - k.start AS dur, k."end",
               s.value AS name
               {', k.graphNodeId' if has_graph else ''}
        FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    ''')
    all_evts = c.fetchall()
    conn.close()

    if not all_evts:
        print("No kernels found")
        return

    # Find decode steps (CUDA Graph regions)
    if has_graph:
        # Split into graph/gap segments
        segs = []
        ct = "graph" if (all_evts[0][4] or 0) > 0 else "gap"
        ce = [all_evts[0]]
        for e in all_evts[1:]:
            t = "graph" if (e[4] or 0) > 0 else "gap"
            if t == ct:
                ce.append(e)
            else:
                segs.append((ct, ce))
                ct, ce = t, [e]
        segs.append((ct, ce))

        # Find (graph, gap) pairs = decode steps
        steps = []
        for i in range(len(segs) - 1):
            if segs[i][0] == "graph" and segs[i+1][0] == "gap":
                steps.append(segs[i][1] + segs[i+1][1])

        # Use mode kernel count to filter consistent steps
        from collections import Counter
        graph_counts = Counter(
            sum(1 for e in s if (e[4] or 0) > 0)
            for s in steps
        )
        if graph_counts:
            mode = graph_counts.most_common(1)[0][0]
            steps = [s for s in steps if sum(1 for e in s if (e[4] or 0) > 0) == mode]

        if not steps:
            print("No consistent decode steps found")
            return

        # Use last step (most representative)
        decode_kernels = steps[-1]
        print(f"Found {len(steps)} consistent decode steps, using last one ({len(decode_kernels)} kernels)")
    else:
        # No graph info, use all kernels (less accurate)
        decode_kernels = all_evts
        print(f"Using all {len(decode_kernels)} kernels (no graphNodeId)")

    # Classify kernels
    marlin_re = re.compile(r"marlin", re.IGNORECASE)
    moe_re = re.compile(r"moe_wna16|moe.*gemm", re.IGNORECASE)
    fa_re = re.compile(r"flash_fwd|flash_bwd|fmha|FlashAttn", re.IGNORECASE)

    categories = defaultdict(lambda: {"count": 0, "time_ns": 0, "kernels": defaultdict(lambda: [0, 0])})

    for evt in decode_kernels:
        name = evt[3]
        dur = evt[1]

        if marlin_re.search(name):
            cat = "marlin (INT4 attn GEMV)"
        elif moe_re.search(name):
            cat = "moe_wna16 (MoE expert)"
        elif fa_re.search(name):
            cat = "flash_attn"
        else:
            cat = "other"

        categories[cat]["count"] += 1
        categories[cat]["time_ns"] += dur
        short = name[:80]
        categories[cat]["kernels"][short][0] += dur
        categories[cat]["kernels"][short][1] += 1

    # Weight bytes estimation per category per decode step
    H = args.hidden
    qd = args.q_heads * args.head_dim
    kvd = args.kv_heads * args.head_dim
    tp = args.tp
    L = args.layers
    moe_ffn = args.moe_intermediate
    top_k = args.top_k

    # Marlin: attention Q/K/V/O projections, INT4 packed (0.5B/param effective)
    # Per layer: Q(H→qd) + K(H→kvd) + V(H→kvd) + O(qd→H) params
    attn_params_per_layer = (H * qd + H * kvd + H * kvd + qd * H)
    attn_total_params = attn_params_per_layer * L / tp
    marlin_bytes = attn_total_params * 0.5  # INT4

    # MoE: top-k experts, each has gate+up+down, INT4
    expert_params = 3 * H * moe_ffn
    moe_params_per_layer = top_k * expert_params
    moe_total_params = moe_params_per_layer * L / tp
    moe_bytes = moe_total_params * 0.5  # INT4

    weight_bytes = {
        "marlin (INT4 attn GEMV)": marlin_bytes,
        "moe_wna16 (MoE expert)": moe_bytes,
    }

    # Print results
    total_ns = sum(c["time_ns"] for c in categories.values())
    total_ms = total_ns / 1e6

    print(f"\n{'='*80}")
    print(f"Single Decode Step Breakdown (TP={tp})")
    print(f"{'='*80}")
    print(f"Total decode step: {total_ms:.3f} ms ({sum(c['count'] for c in categories.values())} kernels)")
    print()

    print(f"{'Category':<30s} {'Time(ms)':>8s} {'%':>6s} {'Count':>6s} {'Weight':>8s} {'BW(GB/s)':>9s} {'BW%':>6s}")
    print("-" * 80)

    for cat in ["marlin (INT4 attn GEMV)", "moe_wna16 (MoE expert)", "flash_attn", "other"]:
        if cat not in categories:
            continue
        info = categories[cat]
        t_ms = info["time_ns"] / 1e6
        pct = info["time_ns"] / total_ns * 100 if total_ns > 0 else 0
        wb = weight_bytes.get(cat, 0)
        if wb > 0 and t_ms > 0:
            bw = (wb / 1e9) / (t_ms / 1000)
            bw_pct = bw / args.peak_bw * 100
            print(f"{cat:<30s} {t_ms:>8.3f} {pct:>5.1f}% {info['count']:>6d} "
                  f"{wb/1e9:.2f}GB {bw:>8.1f} {bw_pct:>5.1f}%")
        else:
            print(f"{cat:<30s} {t_ms:>8.3f} {pct:>5.1f}% {info['count']:>6d} "
                  f"{'—':>8s} {'—':>9s} {'—':>6s}")

    print("-" * 80)
    total_wb = sum(weight_bytes.values())
    total_bw = (total_wb / 1e9) / (total_ms / 1000) if total_ms > 0 else 0
    print(f"{'TOTAL':<30s} {total_ms:>8.3f} {'100%':>6s} "
          f"{sum(c['count'] for c in categories.values()):>6d} "
          f"{total_wb/1e9:.2f}GB {total_bw:>8.1f} {total_bw/args.peak_bw*100:>5.1f}%")

    # Top kernels per category
    for cat in ["marlin (INT4 attn GEMV)", "moe_wna16 (MoE expert)", "flash_attn", "other"]:
        if cat not in categories:
            continue
        info = categories[cat]
        print(f"\n  [{cat}] top kernels:")
        sorted_k = sorted(info["kernels"].items(), key=lambda x: -x[1][0])
        for name, (t, cnt) in sorted_k[:5]:
            print(f"    {name:<70s} {t/1e6:>7.3f}ms ×{cnt}")


if __name__ == "__main__":
    main()
