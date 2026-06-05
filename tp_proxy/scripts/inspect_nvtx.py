#!/usr/bin/env python3
"""Dump NVTX markers from an asys/nsys trace.sqlite for diagnosis.

Shows which markers exist (decode_0, prefill, and the 'bs=' batch markers
from patch_vllm_batch_nvtx.py) so we can tell whether the batch patch was
captured.

Usage:
    python inspect_nvtx.py /path/to/trace.sqlite
"""
import sqlite3
import sys

from compensate_ppu import _find_nvtx_table


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = sys.argv[1]

    info = _find_nvtx_table(path)
    if not info:
        print("ERROR: no NVTX table found in trace")
        sys.exit(1)
    table, start_col, end_col, text_col = info
    print(f"NVTX table: {table}  (start={start_col}, end={end_col}, text={text_col})")

    conn = sqlite3.connect(path)
    c = conn.cursor()

    # Distinct marker texts with counts
    c.execute(f'SELECT "{text_col}", COUNT(*) FROM "{table}" GROUP BY "{text_col}" '
              f'ORDER BY COUNT(*) DESC')
    rows = c.fetchall()
    print(f"\nDistinct NVTX texts ({len(rows)}):")
    for txt, n in rows[:60]:
        print(f"  {n:>6}  {txt}")
    if len(rows) > 60:
        print(f"  ... ({len(rows) - 60} more)")

    # Specifically: any 'bs=' markers?
    c.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{text_col}" LIKE \'%bs=%\'')
    n_bs = c.fetchone()[0]
    print(f"\n'bs=' markers total: {n_bs}")

    # decode_0 range + bs= markers within it
    c.execute(f'SELECT "{start_col}", "{end_col}" FROM "{table}" '
              f'WHERE "{text_col}" = \'decode_0\'')
    row = c.fetchone()
    if row:
        s, e = row
        c.execute(f'SELECT COUNT(*) FROM "{table}" '
                  f'WHERE "{start_col}" >= ? AND "{end_col}" <= ? '
                  f'AND "{text_col}" LIKE \'%bs=%\'', (s, e))
        print(f"'bs=' markers inside decode_0: {c.fetchone()[0]}")
    else:
        print("decode_0 marker: NOT FOUND")

    conn.close()


if __name__ == "__main__":
    main()
