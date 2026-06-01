#!/usr/bin/env python3
"""
Extract GEMM kernels from trace and simulate FP8/FP4 speedup.

Reads nsys/asys sqlite trace, classifies kernels as GEMM vs non-GEMM,
then computes projected time if GEMMs were run at FP8 (2x) or FP4 (4x).

Usage:
    python trace_gemm_scale.py trace.sqlite
    python trace_gemm_scale.py trace.sqlite --nvtx-filter "VIT_"
    python trace_gemm_scale.py trace.sqlite --nvtx-filter "DiT_1step" --fp8-speedup 2.0 --fp4-speedup 4.0
"""

import argparse
import re
import sqlite3
import sys


# GEMM kernel name patterns
GEMM_PATTERNS = [
    r"gemm", r"gemv", r"cutlass", r"cublas", r"cublasLt",
    r"sm\d+_xmma", r"volta_.*gemm", r"ampere_.*gemm", r"hopper_.*gemm",
    r"matmul", r"dot_kernel", r"batch_matmul",
    # Marlin / GPTQ quantized GEMM
    r"marlin", r"gptq",
    # Triton matmul
    r"triton.*matmul", r"tt_dot",
]
GEMM_RE = re.compile("|".join(GEMM_PATTERNS), re.IGNORECASE)


def find_kernel_table(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    for t in tables:
        if "KERNEL" in t.upper() and "ACTIVITY" in t.upper():
            return t
    # nsys fallback
    for t in tables:
        if "CUPTI" in t.upper() and "KERNEL" in t.upper():
            return t
    return None


def find_nvtx_table(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    for t in tables:
        if ("HGTX" in t.upper() or "NVTX" in t.upper()) and "EVENT" in t.upper():
            return t
    return None


def get_nvtx_range(cursor, nvtx_table, nvtx_filter):
    """Find time range of NVTX marker matching filter."""
    if not nvtx_table or not nvtx_filter:
        return None, None

    cursor.execute(f'PRAGMA table_info("{nvtx_table}")')
    cols = [c[1] for c in cursor.fetchall()]

    start_col = next((c for c in cols if c.lower() in ('start', 'timestamp', 'starttime')), None)
    end_col = next((c for c in cols if c.lower() in ('end', 'endtime', 'endtimestamp')), None)
    text_col = next((c for c in cols if c.lower() in ('text', 'message', 'name', 'value')), None)
    text_id_col = next((c for c in cols if c.lower() == 'textid'), None)

    if not start_col:
        return None, None

    # Build query: text may be inline or via textId → StringIds
    if text_id_col:
        where = "s.value LIKE ?"
        join = f'JOIN StringIds s ON n."{text_id_col}" = s.id'
    elif text_col:
        where = f'n."{text_col}" LIKE ?'
        join = ""
    else:
        return None, None

    if end_col:
        q = f'SELECT n."{start_col}", n."{end_col}" FROM "{nvtx_table}" n {join} WHERE {where} ORDER BY n."{start_col}" DESC LIMIT 1'
    else:
        dur_col = next((c for c in cols if c.lower() in ('duration', 'dur')), None)
        if not dur_col:
            return None, None
        q = f'SELECT n."{start_col}", n."{start_col}" + n."{dur_col}" FROM "{nvtx_table}" n {join} WHERE {where} ORDER BY n."{start_col}" DESC LIMIT 1'

    cursor.execute(q, (f"%{nvtx_filter}%",))
    row = cursor.fetchone()
    if row:
        return row[0], row[1]
    return None, None


def analyze(sqlite_path, nvtx_filter=None, fp8_speedup=2.0, fp4_speedup=4.0, top_n=20):
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    kt = find_kernel_table(c)
    if not kt:
        print("ERROR: no kernel table found in sqlite")
        sys.exit(1)

    # Get NVTX time range filter
    nvtx_start, nvtx_end = None, None
    if nvtx_filter:
        nvtx_table = find_nvtx_table(c)
        nvtx_start, nvtx_end = get_nvtx_range(c, nvtx_table, nvtx_filter)
        if nvtx_start:
            print(f"NVTX filter: '{nvtx_filter}' → [{nvtx_start}, {nvtx_end}]")
        else:
            print(f"WARNING: NVTX '{nvtx_filter}' not found, using all kernels")

    # Query kernels
    query = f'''
        SELECT k.start, k."end" - k.start AS dur, s.value AS name
        FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
    '''
    conditions = []
    if nvtx_start and nvtx_end:
        conditions.append(f'k.start >= {nvtx_start} AND k."end" <= {nvtx_end}')
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY k.start"

    c.execute(query)
    rows = c.fetchall()
    conn.close()

    if not rows:
        print("No kernels found")
        return

    # Classify
    gemm_kernels = []
    other_kernels = []
    for start, dur, name in rows:
        if GEMM_RE.search(name):
            gemm_kernels.append((dur, name))
        else:
            other_kernels.append((dur, name))

    gemm_time = sum(d for d, _ in gemm_kernels)
    other_time = sum(d for d, _ in other_kernels)
    total_time = gemm_time + other_time

    # Wall time (first kernel start → last kernel end)
    wall_ns = rows[-1][0] + rows[-1][1] - rows[0][0]

    print(f"\n{'='*70}")
    print(f"Kernel Analysis: {len(rows)} kernels")
    if nvtx_filter:
        print(f"  NVTX range: {nvtx_filter}")
    print(f"{'='*70}")
    print(f"  Total kernel time: {total_time/1e6:.2f} ms")
    print(f"  Wall time:         {wall_ns/1e6:.2f} ms")
    print(f"  GEMM kernels:      {len(gemm_kernels):>6d} ({gemm_time/1e6:.2f} ms, {gemm_time/total_time*100:.1f}%)")
    print(f"  Other kernels:     {len(other_kernels):>6d} ({other_time/1e6:.2f} ms, {other_time/total_time*100:.1f}%)")

    # Top GEMM kernels by total time
    from collections import defaultdict
    gemm_by_name = defaultdict(lambda: [0, 0])
    for dur, name in gemm_kernels:
        gemm_by_name[name][0] += dur
        gemm_by_name[name][1] += 1

    print(f"\n  Top {top_n} GEMM kernels:")
    print(f"  {'Kernel':<60s} {'Time(ms)':>10s} {'Count':>6s} {'%':>6s}")
    print(f"  {'-'*85}")
    sorted_gemm = sorted(gemm_by_name.items(), key=lambda x: -x[1][0])
    for name, (t, cnt) in sorted_gemm[:top_n]:
        short = name[:58] + ".." if len(name) > 60 else name
        print(f"  {short:<60s} {t/1e6:>10.2f} {cnt:>6d} {t/total_time*100:>5.1f}%")

    # Top non-GEMM
    other_by_name = defaultdict(lambda: [0, 0])
    for dur, name in other_kernels:
        other_by_name[name][0] += dur
        other_by_name[name][1] += 1

    print(f"\n  Top {top_n} non-GEMM kernels:")
    print(f"  {'Kernel':<60s} {'Time(ms)':>10s} {'Count':>6s} {'%':>6s}")
    print(f"  {'-'*85}")
    sorted_other = sorted(other_by_name.items(), key=lambda x: -x[1][0])
    for name, (t, cnt) in sorted_other[:top_n]:
        short = name[:58] + ".." if len(name) > 60 else name
        print(f"  {short:<60s} {t/1e6:>10.2f} {cnt:>6d} {t/total_time*100:>5.1f}%")

    # Projected times
    print(f"\n{'='*70}")
    print(f"Projected Times (GEMM scaling, non-GEMM unchanged)")
    print(f"{'='*70}")
    print(f"  {'Precision':<12s} {'GEMM(ms)':>10s} {'Other(ms)':>10s} {'Total(ms)':>10s} {'Speedup':>8s}")
    print(f"  {'-'*55}")

    for label, speedup in [("FP16 (base)", 1.0), (f"FP8 ({fp8_speedup}x)", fp8_speedup),
                            (f"FP4 ({fp4_speedup}x)", fp4_speedup)]:
        g = gemm_time / speedup
        t = g + other_time
        sp = total_time / t if t > 0 else 0
        print(f"  {label:<12s} {g/1e6:>10.2f} {other_time/1e6:>10.2f} {t/1e6:>10.2f} {sp:>7.2f}x")

    # Also estimate wall time scaling (assume gaps don't change)
    gap_time = wall_ns - total_time
    print(f"\n  Wall time estimate (gaps={gap_time/1e6:.2f}ms unchanged):")
    for label, speedup in [("FP16", 1.0), ("FP8", fp8_speedup), ("FP4", fp4_speedup)]:
        g = gemm_time / speedup
        wall = g + other_time + gap_time
        sp = wall_ns / wall if wall > 0 else 0
        print(f"    {label:<6s}: {wall/1e6:.2f} ms ({sp:.2f}x)")


def main():
    ap = argparse.ArgumentParser(description="Extract GEMM from trace, project FP8/FP4 speedup")
    ap.add_argument("sqlite", help="nsys/asys trace sqlite file")
    ap.add_argument("--nvtx-filter", default=None,
                    help="Filter kernels within NVTX range (e.g. 'VIT_0', 'DiT_1step_0')")
    ap.add_argument("--fp8-speedup", type=float, default=2.0,
                    help="FP8 GEMM speedup over FP16 (default: 2.0)")
    ap.add_argument("--fp4-speedup", type=float, default=4.0,
                    help="FP4 GEMM speedup over FP16 (default: 4.0)")
    ap.add_argument("--top", type=int, default=20, help="Top N kernels to show")
    args = ap.parse_args()

    analyze(args.sqlite, args.nvtx_filter, args.fp8_speedup, args.fp4_speedup, args.top)


if __name__ == "__main__":
    main()
