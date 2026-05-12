#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "select 1 from sqlite_master where type='table' and name=?", (table,)
    ).fetchone()
    return row is not None


def string_expr(alias: str = "s") -> str:
    return f"coalesce({alias}.value, '<unknown>')"


def summarize_cuda_api(con: sqlite3.Connection, limit: int) -> list[dict[str, object]]:
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        return []
    rows = con.execute(
        f"""
        select {string_expr()} as name,
               count(*) as calls,
               sum(r.end - r.start) / 1e6 as total_ms,
               avg(r.end - r.start) / 1e3 as avg_us,
               max(r.end - r.start) / 1e3 as max_us
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        left join StringIds s on r.nameId = s.id
        group by name
        order by total_ms desc
        limit ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "name": row[0],
            "calls": row[1],
            "total_ms": round(row[2], 3),
            "avg_us": round(row[3], 3),
            "max_us": round(row[4], 3),
        }
        for row in rows
    ]


def summarize_osrt(con: sqlite3.Connection, limit: int) -> list[dict[str, object]]:
    if not table_exists(con, "OSRT_API"):
        return []
    rows = con.execute(
        f"""
        select {string_expr()} as name,
               count(*) as calls,
               sum(o.end - o.start) / 1e6 as total_ms,
               avg(o.end - o.start) / 1e3 as avg_us,
               max(o.end - o.start) / 1e3 as max_us
        from OSRT_API o
        left join StringIds s on o.nameId = s.id
        group by name
        order by total_ms desc
        limit ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "name": row[0],
            "calls": row[1],
            "total_ms": round(row[2], 3),
            "avg_us": round(row[3], 3),
            "max_us": round(row[4], 3),
        }
        for row in rows
    ]


def summarize_kernels(con: sqlite3.Connection, limit: int) -> list[dict[str, object]]:
    if not table_exists(con, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return []
    rows = con.execute(
        f"""
        select {string_expr()} as name,
               count(*) as launches,
               sum(k.end - k.start) / 1e6 as total_ms,
               avg(k.end - k.start) / 1e3 as avg_us,
               max(k.end - k.start) / 1e3 as max_us
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on k.demangledName = s.id
        group by name
        order by total_ms desc
        limit ?
        """,
        (limit,),
    ).fetchall()
    return [
        {
            "name": row[0],
            "launches": row[1],
            "total_ms": round(row[2], 3),
            "avg_us": round(row[3], 3),
            "max_us": round(row[4], 3),
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize selected Nsight Systems SQLite tables.")
    parser.add_argument("sqlite_file")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite_file)
    con = sqlite3.connect(sqlite_path)
    tables = [
        row[0]
        for row in con.execute(
            "select name from sqlite_master where type='table' order by name"
        )
    ]
    summary = {
        "sqlite_file": str(sqlite_path),
        "tables": tables,
        "cuda_api": summarize_cuda_api(con, args.top),
        "osrt_api": summarize_osrt(con, args.top),
        "cuda_kernels": summarize_kernels(con, args.top),
    }
    Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
