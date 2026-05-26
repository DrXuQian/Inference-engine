#!/usr/bin/env python3
"""
Convert asys trace sqlite to CSV.

Usage:
    python sqlite_to_csv.py trace.sqlite                    # export all tables
    python sqlite_to_csv.py trace.sqlite -o output_dir/     # specify output dir
    python sqlite_to_csv.py trace.sqlite -t KERNEL           # only tables matching keyword
"""

import argparse
import csv
import os
import sqlite3
import sys


def main():
    ap = argparse.ArgumentParser(description="Export asys trace sqlite tables to CSV")
    ap.add_argument("sqlite", help="Path to trace.sqlite")
    ap.add_argument("-o", "--output-dir", default=None,
                    help="Output directory (default: same dir as sqlite)")
    ap.add_argument("-t", "--table-filter", default=None,
                    help="Only export tables whose name contains this keyword (case-insensitive)")
    ap.add_argument("--list", action="store_true",
                    help="List all tables and exit")
    args = ap.parse_args()

    if not os.path.exists(args.sqlite):
        print(f"ERROR: {args.sqlite} not found", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(args.sqlite)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cursor.fetchall()]

    if args.list:
        for t in tables:
            cursor.execute(f"SELECT COUNT(*) FROM \"{t}\"")
            count = cursor.fetchone()[0]
            print(f"  {t:50s} ({count} rows)")
        conn.close()
        return

    # Filter
    if args.table_filter:
        kw = args.table_filter.lower()
        tables = [t for t in tables if kw in t.lower()]
        if not tables:
            print(f"No tables matching '{args.table_filter}'", file=sys.stderr)
            conn.close()
            sys.exit(1)

    out_dir = args.output_dir or os.path.dirname(args.sqlite) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Check if StringIds table exists for name resolution
    has_string_ids = "StringIds" in [t for t in tables]
    # Also check for lowercase variant
    if not has_string_ids:
        for t in tables:
            if t.lower() == "stringids":
                has_string_ids = True
                break

    for table in tables:
        # For kernel/activity tables, JOIN with StringIds to resolve names
        cursor.execute(f"PRAGMA table_info(\"{table}\")")
        col_info = cursor.fetchall()
        col_names_lower = {c[1].lower(): c[1] for c in col_info}

        # Detect string ID columns that should be resolved
        str_id_cols = []
        if has_string_ids:
            for candidate in ["demangledName", "mangledName", "shortName",
                              "registeredName", "name"]:
                if candidate in col_names_lower.values() or candidate.lower() in col_names_lower:
                    actual = col_names_lower.get(candidate.lower(), candidate)
                    str_id_cols.append(actual)

        if str_id_cols and table.lower() != "stringids":
            # Build query with JOINs to resolve string IDs
            joins = []
            selects = []
            cursor.execute(f"PRAGMA table_info(\"{table}\")")
            for c in cursor.fetchall():
                col = c[1]
                if col in str_id_cols:
                    alias = f"s_{col}"
                    selects.append(f'{alias}.value AS "{col}"')
                    joins.append(
                        f'LEFT JOIN StringIds {alias} ON t."{col}" = {alias}.id')
                else:
                    selects.append(f't."{col}"')

            query = f"SELECT {', '.join(selects)} FROM \"{table}\" t {' '.join(joins)} ORDER BY t.start" if "start" in col_names_lower else \
                    f"SELECT {', '.join(selects)} FROM \"{table}\" t {' '.join(joins)}"
            try:
                cursor.execute(query)
            except Exception:
                # Fallback to raw query if JOIN fails
                cursor.execute(f"SELECT * FROM \"{table}\"")
        else:
            cursor.execute(f"SELECT * FROM \"{table}\"")

        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]

        out_path = os.path.join(out_dir, f"{table}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            w.writerows(rows)

        print(f"  {table} -> {out_path} ({len(rows)} rows)")

    conn.close()
    print(f"\nExported {len(tables)} table(s) to {out_dir}")


if __name__ == "__main__":
    main()
