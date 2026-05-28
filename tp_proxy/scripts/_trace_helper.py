"""
Simple trace analysis: one decode step = one CUDA Graph + gap to next CUDA Graph.

  encoder = CUDA Graph wall time (graph_end - graph_start)
  tail    = gap wall time (gap_end - next_graph_start or gap_end)
  tpot    = encoder + tail

That's it.
"""
import sqlite3
from collections import Counter


def measure_tail_from_trace(sqlite_path, lm_head_kernel):
    conn = sqlite3.connect(sqlite_path)
    c = conn.cursor()

    # Find kernel table
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    kt = [r[0] for r in c.fetchall() if 'KERNEL' in r[0].upper() and 'ACTIVITY' in r[0].upper()]
    if not kt:
        conn.close(); return None
    kt = kt[0]

    c.execute(f'''
        SELECT k.start, k."end" - k.start AS dur, k."end",
               s.value AS name, k.graphNodeId
        FROM "{kt}" k JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    ''')
    evts = [(r[0], r[1], r[2], r[3], r[4] or 0) for r in c.fetchall()]
    conn.close()

    n_g = sum(1 for *_, g in evts if g > 0)
    print(f"  kernels: {len(evts)} ({n_g} graph, {len(evts)-n_g} non-graph)")

    # Split into graph/gap segments
    segs = []
    ct = "graph" if evts[0][4] > 0 else "gap"
    ce = [evts[0]]
    for e in evts[1:]:
        t = "graph" if e[4] > 0 else "gap"
        if t == ct:
            ce.append(e)
        else:
            segs.append((ct, ce))
            ct, ce = t, [e]
    segs.append((ct, ce))

    # Each decode step = (graph_seg, gap_seg)
    steps = []
    for i in range(len(segs) - 1):
        if segs[i][0] == "graph" and segs[i+1][0] == "gap":
            g_evts = segs[i][1]
            gap_evts = segs[i+1][1]
            g_start = g_evts[0][0]
            g_end = g_evts[-1][2]
            gap_end = gap_evts[-1][2]
            # Find lm_head in gap
            lm_dur = 0
            for ev in gap_evts:
                if lm_head_kernel in ev[3]:
                    lm_dur = ev[1]
                    break
            steps.append({
                "encoder_wall": g_end - g_start,
                "tail_wall": gap_end - g_end,
                "step_wall": gap_end - g_start,
                "lm_head_dur": lm_dur,
                "graph_kernels": len(g_evts),
                "gap_kernels": len(gap_evts),
            })

    print(f"  decode steps: {len(steps)}")
    if not steps:
        return None

    # Filter: consistent graph kernel count
    counts = Counter(s["graph_kernels"] for s in steps)
    mode_count = counts.most_common(1)[0][0]
    filtered = [s for s in steps if s["graph_kernels"] == mode_count]
    print(f"  graph kernel mode: {mode_count} ({len(filtered)}/{len(steps)} steps)")

    # Filter: has matching lm_head
    matched = [s for s in filtered if s["lm_head_dur"] > 0]
    if matched:
        print(f"  with lm_head '{lm_head_kernel}': {len(matched)}")
        # Pick top 25% by lm_head duration (= largest batch)
        matched.sort(key=lambda s: s["lm_head_dur"])
        threshold = matched[len(matched) * 3 // 4]["lm_head_dur"] if len(matched) > 4 else matched[0]["lm_head_dur"]
        top = [s for s in matched if s["lm_head_dur"] >= threshold]
        print(f"  lm_head range: {matched[0]['lm_head_dur']/1e3:.1f}-{matched[-1]['lm_head_dur']/1e3:.1f}us, "
              f"p75={threshold/1e3:.1f}us, top={len(top)}")
        selected = top
    else:
        print(f"  WARNING: '{lm_head_kernel}' not found, using all")
        selected = filtered

    # Use last step
    s = selected[-1]
    encoder_ms = s["encoder_wall"] / 1e6
    tail_ms = s["tail_wall"] / 1e6
    lm_head_ms = s["lm_head_dur"] / 1e6
    sampling_ms = tail_ms - lm_head_ms
    tpot_ms = s["step_wall"] / 1e6

    # Prefill: everything before first consistent graph
    first_graph_start = filtered[0]["step_wall"]  # wrong, need actual timestamp
    # Recalculate from raw data
    first_filtered_idx = next(i for i, st in enumerate(steps) if st["graph_kernels"] == mode_count)
    # Find the actual graph segment for this step
    graph_seg_idx = 0
    step_count = 0
    for si, (stype, sevts) in enumerate(segs):
        if stype == "graph" and si + 1 < len(segs) and segs[si+1][0] == "gap":
            if step_count == first_filtered_idx:
                first_graph_time = sevts[0][0]
                break
            step_count += 1
    else:
        first_graph_time = evts[0][0]

    prefill_wall_ns = first_graph_time - evts[0][0]
    ttft_ms = prefill_wall_ns / 1e6

    print(f"\n  === Result ===")
    print(f"  encoder (graph wall): {encoder_ms:.4f} ms")
    print(f"  tail (to next graph): {tail_ms:.4f} ms")
    print(f"    lm_head: {lm_head_ms:.4f} ms")
    print(f"    sampling: {sampling_ms:.4f} ms")
    print(f"  TPOT (step wall):     {tpot_ms:.4f} ms")
    print(f"  TTFT (prefill wall):  {ttft_ms:.2f} ms")
    print(f"  check: encoder + tail = {encoder_ms + tail_ms:.4f} ms")

    return {
        "encoder_ms": round(encoder_ms, 4),
        "lm_head_ms": round(lm_head_ms, 4),
        "sampling_ms": round(max(sampling_ms, 0), 4),
        "tpot_ms": round(tpot_ms, 4),
        "ttft_ms": round(ttft_ms, 2),
        "overhead_ms": 0,
        "tail_per_step_ms": round(tail_ms, 4),
    }


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "trace.sqlite"
    kernel = sys.argv[2] if len(sys.argv) > 2 else "gemvt_op"
    print(f"DB: {db}\nlm_head: {kernel}\n")
    r = measure_tail_from_trace(db, kernel)
    if r:
        print(f"\n{r}")
