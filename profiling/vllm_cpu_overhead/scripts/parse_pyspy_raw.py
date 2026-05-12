#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_line(line: str) -> tuple[list[str], int] | None:
    line = line.strip()
    if not line:
        return None
    try:
        stack_text, count_text = line.rsplit(" ", 1)
        count = int(count_text)
    except ValueError:
        return None
    return stack_text.split(";"), count


def simplify(frame: str) -> str:
    frame = frame.strip()
    for prefix in (
        "/root/autodl-tmp/qwen35-quant-venv/lib/python3.12/site-packages/",
        "/root/autodl-tmp/qwen35-compress-venv/lib/python3.12/site-packages/",
    ):
        frame = frame.replace(prefix, "")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize py-spy raw stack samples.")
    parser.add_argument("raw_file")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    raw_path = Path(args.raw_file)
    leaf_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    stack_counts: Counter[str] = Counter()
    total_samples = 0

    for line in raw_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = parse_line(line)
        if parsed is None:
            continue
        stack, count = parsed
        stack = [simplify(frame) for frame in stack]
        total_samples += count
        if stack:
            leaf_counts[stack[-1]] += count
        for frame in set(stack):
            frame_counts[frame] += count
        stack_counts[";".join(stack)] += count

    def top(counter: Counter[str]) -> list[dict[str, object]]:
        return [
            {
                "samples": count,
                "percent": round(100.0 * count / total_samples, 2) if total_samples else 0,
                "name": name,
            }
            for name, count in counter.most_common(args.top)
        ]

    summary = {
        "raw_file": str(raw_path),
        "total_samples": total_samples,
        "top_leaf_functions": top(leaf_counts),
        "top_frames_anywhere_in_stack": top(frame_counts),
        "top_stacks": top(stack_counts),
    }
    Path(args.json_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
