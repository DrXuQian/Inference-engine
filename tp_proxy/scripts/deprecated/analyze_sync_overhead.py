#!/usr/bin/env python3
"""
Analyze all-reduce synchronization overhead from asys/nsys sqlite trace.

For each all-reduce call, computes:
  - kernel_time = GPU kernel execution duration
  - sync_time = max(device0, device1) wall time (includes waiting)
  - sync_overhead = sync_time - kernel_time

Separates prefill (eager, large tensors) and decode (CUDA Graph, small tensors).

Usage:
    # From asys sqlite
    python analyze_sync_overhead.py \
        --sqlite result.sqlite \
        --start-ns 1000000000 --end-ns 5000000000

    # From nsys sqlite
    python analyze_sync_overhead.py \
        --sqlite report.sqlite --platform nvidia \
        --start-ns 1000000000 --end-ns 5000000000

    # Auto-detect serving window (no timestamps needed)
    python analyze_sync_overhead.py --sqlite result.sqlite
"""

import argparse
import sqlite3
import sys
from collections import defaultdict


def load_ar_events(sqlite_path: str, platform: str,
                   start_ns: int | None, end_ns: int | None) -> list[dict]:
    """Load all-reduce kernel events from sqlite trace."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    if platform == "ppu":
        kernel_table = "HGPTI_ACTIVITY_KIND_KERNEL"
    else:
        kernel_table = "CUPTI_ACTIVITY_KIND_KERNEL"

    # Check which table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    if kernel_table not in tables:
        # Try the other one
        for t in tables:
            if "KERNEL" in t and "ACTIVITY" in t:
                kernel_table = t
                break

    print(f"Using kernel table: {kernel_table}")

    # Get all kernels with names
    query = f"""
        SELECT k.start, k."end", k."end" - k.start AS duration,
               k.deviceId, s.value AS name
        FROM {kernel_table} k
        JOIN StringIds s ON k.demangledName = s.id
    """
    conditions = []
    if start_ns is not None:
        conditions.append(f"k.start >= {start_ns}")
    if end_ns is not None:
        conditions.append(f"k.start <= {end_ns}")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY k.start"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    events = []
    for start, end, dur, dev, name in rows:
        events.append({
            "start": start, "end": end, "dur": dur,
            "device": dev, "name": name,
        })
    return events


def find_ar_events(events: list[dict]) -> list[dict]:
    """Filter to all-reduce / cross_device_reduce events."""
    ar_keywords = ["allreduce", "all_reduce", "cross_device_reduce",
                   "pccl_allreduce", "nccl"]
    # Exclude all-gather
    exclude = ["allgather", "all_gather"]

    ar_events = []
    for e in events:
        nl = e["name"].lower()
        if any(kw in nl for kw in exclude):
            continue
        if any(kw in nl for kw in ar_keywords):
            ar_events.append(e)
    return ar_events


def pair_ar_events(ar_events: list[dict]) -> list[dict]:
    """Pair all-reduce calls across two devices by closest start time."""
    # Separate by device
    by_device = defaultdict(list)
    for e in ar_events:
        by_device[e["device"]].append(e)

    devices = sorted(by_device.keys())
    if len(devices) < 2:
        print(f"WARNING: only found {len(devices)} device(s), need 2 for sync analysis")
        return []

    dev0, dev1 = devices[0], devices[1]
    events0 = sorted(by_device[dev0], key=lambda e: e["start"])
    events1 = sorted(by_device[dev1], key=lambda e: e["start"])

    print(f"Device {dev0}: {len(events0)} AR calls")
    print(f"Device {dev1}: {len(events1)} AR calls")

    # Pair by index (both devices execute same sequence of AR calls)
    n = min(len(events0), len(events1))
    pairs = []
    for i in range(n):
        e0 = events0[i]
        e1 = events1[i]
        # Sync time = max(end) - min(start) for the pair
        pair_start = min(e0["start"], e1["start"])
        pair_end = max(e0["end"], e1["end"])
        sync_wall = pair_end - pair_start
        kernel_max = max(e0["dur"], e1["dur"])
        sync_overhead = sync_wall - kernel_max
        start_diff = abs(e0["start"] - e1["start"])

        pairs.append({
            "idx": i,
            "dev0_start": e0["start"],
            "dev0_dur_us": e0["dur"] / 1e3,
            "dev1_start": e1["start"],
            "dev1_dur_us": e1["dur"] / 1e3,
            "sync_wall_us": sync_wall / 1e3,
            "kernel_max_us": kernel_max / 1e3,
            "sync_overhead_us": sync_overhead / 1e3,
            "start_diff_us": start_diff / 1e3,
            "name": e0["name"],
        })
    return pairs


def classify_prefill_decode(pairs: list[dict]) -> tuple[list, list]:
    """Separate prefill (large kernel time) and decode (small kernel time) AR calls.

    Heuristic: prefill AR kernels are much longer (>100us) than decode (<50us).
    Also check for gaps between consecutive AR calls to find request boundaries.
    """
    if not pairs:
        return [], []

    # Use kernel duration to classify
    # Prefill: large tensor → longer kernel time
    # Decode: small tensor → shorter kernel time
    # Threshold: median kernel time (bimodal distribution)
    kernel_times = sorted(p["kernel_max_us"] for p in pairs)
    median_kt = kernel_times[len(kernel_times) // 2]

    # If distribution is bimodal, find the gap
    # Simple: threshold at 10x the minimum
    min_kt = kernel_times[0]
    threshold = max(min_kt * 10, 50)  # at least 50us

    prefill = [p for p in pairs if p["kernel_max_us"] > threshold]
    decode = [p for p in pairs if p["kernel_max_us"] <= threshold]

    return prefill, decode


def print_stats(pairs: list[dict], label: str):
    """Print all-reduce timing statistics."""
    if not pairs:
        print(f"\n{label}: no data")
        return

    sync_walls = sorted(p["sync_wall_us"] for p in pairs)
    kernel_maxs = sorted(p["kernel_max_us"] for p in pairs)

    n = len(pairs)
    med = lambda xs: xs[len(xs) // 2]
    mean = lambda xs: sum(xs) / len(xs)

    print(f"\n{label} ({n} calls):")
    print(f"  mean(max(dev0,dev1)): {mean(sync_walls):.1f} us")
    print(f"  median:               {med(sync_walls):.1f} us")
    print(f"  kernel_only mean:     {mean(kernel_maxs):.1f} us")
    print(f"  kernel_only median:   {med(kernel_maxs):.1f} us")


def main():
    ap = argparse.ArgumentParser(description="Analyze all-reduce sync overhead")
    ap.add_argument("--sqlite", required=True, help="Path to sqlite trace file")
    ap.add_argument("--platform", choices=["nvidia", "ppu"], default="ppu")
    ap.add_argument("--start-ns", type=int, default=None,
                    help="Start timestamp (ns). If omitted, analyze full trace.")
    ap.add_argument("--end-ns", type=int, default=None,
                    help="End timestamp (ns). If omitted, analyze full trace.")
    args = ap.parse_args()

    print(f"Loading trace: {args.sqlite}")
    events = load_ar_events(args.sqlite, args.platform, args.start_ns, args.end_ns)
    print(f"Total kernel events in window: {len(events)}")

    ar_events = find_ar_events(events)
    print(f"All-reduce events: {len(ar_events)}")

    if not ar_events:
        # Show available kernel names for debugging
        names = set(e["name"] for e in events[:1000])
        print("\nNo AR events found. Sample kernel names:")
        for n in sorted(names)[:20]:
            print(f"  {n}")
        sys.exit(1)

    # Show AR kernel name variants
    ar_names = set(e["name"] for e in ar_events)
    print(f"AR kernel variants: {len(ar_names)}")
    for n in ar_names:
        count = sum(1 for e in ar_events if e["name"] == n)
        print(f"  {count:6d}  {n[:80]}")

    # Pair across devices
    pairs = pair_ar_events(ar_events)
    if not pairs:
        sys.exit(1)

    # Classify prefill vs decode
    prefill, decode = classify_prefill_decode(pairs)
    print(f"\nClassified: {len(prefill)} prefill, {len(decode)} decode")

    # Print stats
    print_stats(prefill, "PREFILL all-reduce")
    print_stats(decode, "DECODE all-reduce")
    print_stats(pairs, "ALL all-reduce (combined)")


if __name__ == "__main__":
    main()
