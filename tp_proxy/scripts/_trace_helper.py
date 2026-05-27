"""Temporary test: new trace analysis logic"""
import sqlite3

def measure_tail_from_trace(sqlite_path, lm_head_kernel):
    """Extract per-step decode breakdown using graphNodeId.

    Decode step structure:
      [CUDA Graph region: encoder, graphNodeId>0] [gap: lm_head+sampling, graphNodeId=0]

    Only analyzes steps where the graph region has consistent kernel count
    (= correct batch size). Skips partial-batch steps.
    """
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    kt = ""
    for r in cursor.fetchall():
        if "KERNEL" in r[0].upper() and "ACTIVITY" in r[0].upper():
            kt = r[0]; break
    if not kt:
        conn.close(); return None

    cursor.execute(f'PRAGMA table_info("{kt}")')
    columns = [c[1] for c in cursor.fetchall()]
    if "graphNodeId" not in columns:
        conn.close()
        print("  WARNING: no graphNodeId column")
        return None

    cursor.execute(f'''
        SELECT k.start, k."end" - k.start AS dur, k."end",
               s.value AS name, k.graphNodeId
        FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    ''')
    all_events = [(r[0], r[1], r[2], r[3], r[4] or 0) for r in cursor.fetchall()]
    conn.close()

    n_graph = sum(1 for *_, g in all_events if g > 0)
    print(f"  Total: {len(all_events)} kernels ({n_graph} graph, {len(all_events)-n_graph} non-graph)")

    # Split into alternating graph/gap segments
    segments = []
    cur_type = "graph" if all_events[0][4] > 0 else "gap"
    cur_evts = [all_events[0]]
    for ev in all_events[1:]:
        t = "graph" if ev[4] > 0 else "gap"
        if t == cur_type:
            cur_evts.append(ev)
        else:
            segments.append((cur_type, cur_evts))
            cur_type = t
            cur_evts = [ev]
    segments.append((cur_type, cur_evts))

    # Pair: each decode step = (graph_seg, gap_seg)
    # graph_seg = encoder, gap_seg = lm_head + sampling
    steps = []
    for i in range(len(segments) - 1):
        if segments[i][0] == "graph" and segments[i+1][0] == "gap":
            steps.append((segments[i][1], segments[i+1][1]))

    if not steps:
        print("  WARNING: no (graph, gap) pairs found")
        return None
    print(f"  Decode steps (graph→gap pairs): {len(steps)}")

    # Filter to consistent batch size: most common graph kernel count
    graph_counts = [len(s[0]) for s in steps]
    from collections import Counter
    count_freq = Counter(graph_counts)
    most_common_count = count_freq.most_common(1)[0][0]
    consistent_steps = [(g, gap) for g, gap in steps if len(g) == most_common_count]
    print(f"  Graph kernel count mode: {most_common_count} "
          f"({len(consistent_steps)}/{len(steps)} steps)")

    if not consistent_steps:
        return None

    # Use last consistent step (most stable)
    graph_evts, gap_evts = consistent_steps[-1]

    # Encoder = sum of graph kernel durations
    encoder_ns = sum(d for _, d, _, _, _ in graph_evts)
    encoder_wall_ns = graph_evts[-1][2] - graph_evts[0][0]

    # Gap = lm_head + sampling (between two CUDA Graph regions)
    # sampling = gap total - lm_head
    gap_total_ns = sum(d for _, d, _, _, _ in gap_evts)
    lm_head_ns = 0
    for ev in gap_evts:
        if lm_head_kernel in ev[3]:
            lm_head_ns = ev[1]
            break
    sampling_ns = gap_total_ns - lm_head_ns

    if lm_head_ns == 0:
        print(f"  WARNING: '{lm_head_kernel}' not found in gap. Gap kernels:")
        for ev in gap_evts[:5]:
            print(f"    {ev[3][:70]}  dur={ev[1]/1e3:.1f}us")
        if gap_evts:
            largest = max(gap_evts, key=lambda e: e[1])
            lm_head_ns = largest[1]
            sampling_ns = gap_total_ns - lm_head_ns

    # Overhead
    gap_wall_ns = gap_evts[-1][2] - gap_evts[0][0] if gap_evts else 0
    step_wall_ns = encoder_wall_ns + gap_wall_ns
    step_kernel_ns = encoder_ns + lm_head_ns + sampling_ns
    overhead_ns = max(step_wall_ns - step_kernel_ns, 0)

    print(f"  encoder (CUDA Graph): {encoder_ns/1e6:.4f} ms ({len(graph_evts)} kernels)")
    print(f"  lm_head:  {lm_head_ns/1e6:.4f} ms")
    print(f"  sampling: {sampling_ns/1e6:.4f} ms ({len(gap_evts)-1 if lm_head_ns>0 else len(gap_evts)} kernels)")
    print(f"  step wall: {step_wall_ns/1e6:.4f} ms")
    print(f"  overhead:  {overhead_ns/1e6:.4f} ms")

    return {
        "lm_head_ms": round(lm_head_ns / 1e6, 4),
        "sampling_ms": round(sampling_ns / 1e6, 4),
        "encoder_ms": round(encoder_ns / 1e6, 4),
        "overhead_ms": round(overhead_ns / 1e6, 4),
        "tail_per_step_ms": round((lm_head_ns + sampling_ns) / 1e6, 4),
    }

# Quick test
if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "trace.sqlite"
    kernel = sys.argv[2] if len(sys.argv) > 2 else "gemvt_op"
    print(f"DB: {db}, lm_head_kernel: {kernel}")
    r = measure_tail_from_trace(db, kernel)
    if r:
        print(f"\nResult: {r}")
