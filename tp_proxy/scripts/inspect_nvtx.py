#!/usr/bin/env python3
"""Dump / verify NVTX markers from an asys/nsys trace.sqlite.

Shows which markers exist (decode_0, prefill, and the 'bs=' batch markers from
patch_vllm_batch_nvtx.py) and the actual decode batch size, so you can confirm
whether multi-batch decode really ran.

Usage:
    python inspect_nvtx.py /path/to/trace.sqlite            # dump markers
    python inspect_nvtx.py /path/to/trace.sqlite --expect 4 # verify decode bs==4
                                                            # exit 1 on mismatch
"""
import re
import sqlite3
import sys
from collections import Counter

from compensate_ppu import _find_nvtx_table


def decode_batch(path: str):
    """Return (mode_batch, {batch: count}) from decode-phase 'bs=' markers.

    Prefill markers ('prefill bs=...') are excluded so the result reflects the
    decode batch. Returns (None, {}) if no decode 'bs=' marker is present.
    """
    info = _find_nvtx_table(path)
    if not info:
        return None, {}
    table, _sc, _ec, tc = info
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(f'SELECT "{tc}" FROM "{table}" WHERE "{tc}" LIKE \'%bs=%\'')
    vals = []
    for (txt,) in c.fetchall():
        if not txt or "prefill" in txt.lower():
            continue
        m = re.search(r"bs=(\d+)", txt)
        if m:
            vals.append(int(m.group(1)))
    conn.close()
    if not vals:
        return None, {}
    cnt = Counter(vals)
    return cnt.most_common(1)[0][0], dict(sorted(cnt.items()))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    expect = None
    if "--expect" in sys.argv:
        expect = int(sys.argv[sys.argv.index("--expect") + 1])
    if not args:
        print(__doc__)
        sys.exit(1)
    path = args[0]

    info = _find_nvtx_table(path)
    if not info:
        print("ERROR: no NVTX table found in trace")
        sys.exit(1)
    table, start_col, end_col, text_col = info
    print(f"NVTX table: {table}  (start={start_col}, end={end_col}, text={text_col})")

    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(f'SELECT "{text_col}", COUNT(*) FROM "{table}" GROUP BY "{text_col}" '
              f'ORDER BY COUNT(*) DESC')
    rows = c.fetchall()
    conn.close()

    print(f"\nDistinct NVTX texts ({len(rows)}):")
    for txt, n in rows[:60]:
        print(f"  {n:>6}  {txt}")
    if len(rows) > 60:
        print(f"  ... ({len(rows) - 60} more)")

    mode, dist = decode_batch(path)
    print(f"\nDecode batch (from 'decode bs=N' markers): {mode}")
    if dist:
        print(f"  distribution {{batch: steps}}: {dist}")

    if expect is not None:
        print(f"\nExpected decode batch: {expect}")
        if mode is None:
            print("FAIL: no 'decode bs=N' marker in trace — cannot confirm batching. "
                  "Was patch_vllm_batch_nvtx.py loaded (PYTHONPATH=_sitepatch)?")
            sys.exit(1)
        if mode != expect:
            print(f"FAIL: decode ran at batch={mode}, not {expect} "
                  f"(vLLM split the batch).")
            sys.exit(1)
        print(f"OK: decode ran at batch={expect}.")
        sys.exit(0)


if __name__ == "__main__":
    main()
