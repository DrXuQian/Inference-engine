#!/usr/bin/env python3
"""
Generate performance report table from compensated_ppu.json results.

Usage:
    python generate_report.py [--results-dir ./results] [--format markdown|csv]
"""

import argparse
import json
import os
import sys


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_ms(val_ms: float) -> str:
    """Format milliseconds to human-readable string."""
    if val_ms < 1:
        return f"{val_ms * 1000:.1f}us"
    if val_ms < 1000:
        return f"{val_ms:.2f}ms"
    return f"{val_ms / 1000:.2f}s"


def get_metrics(data: dict) -> dict | None:
    """Extract comp_ttft, comp_tpot, tps, total from compensated json."""
    results = data.get("results", [])
    # Take first non-error result (or average if multiple input_lens)
    valid = [r for r in results if "error" not in r and "comp_tpot_ms" in r]
    if not valid:
        return None

    # Use the result matching the scenario's target input_len (usually only one)
    r = valid[0]
    ttft = r["comp_ttft_ms"]
    tpot = r["comp_tpot_ms"]
    tps = 1000.0 / tpot if tpot > 0 else 0
    total = r.get("comp_total_ms", ttft + (r.get("output_tokens", 64) - 1) * tpot)
    output_tokens = r.get("output_tokens", data.get("results", [{}])[0].get("output_len", 64))

    return {
        "ttft": ttft,
        "tpot": tpot,
        "tps": tps,
        "total": total,
        "output_tokens": output_tokens,
    }


def print_scenario(title: str, rows: list[tuple[str, dict | None]], fmt: str):
    """Print a scenario block with model rows."""
    if fmt == "csv":
        for model_name, m in rows:
            if m:
                print(f"{title},{model_name},{m['ttft']:.2f},{m['tpot']:.3f},"
                      f"{m['tps']:.1f},{m['total']:.1f}")
            else:
                print(f"{title},{model_name},N/A,N/A,N/A,N/A")
    else:
        print(f"\n### {title}\n")
        print(f"| 模型 | TTFT | TPOT | TPS | 总延迟 |")
        print(f"|------|------|------|-----|--------|")
        for model_name, m in rows:
            if m:
                print(f"| {model_name} | {fmt_ms(m['ttft'])} | {fmt_ms(m['tpot'])} | "
                      f"{m['tps']:.1f} tok/s | {fmt_ms(m['total'])} |")
            else:
                print(f"| {model_name} | N/A | N/A | N/A | N/A |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    args = ap.parse_args()
    rd = args.results_dir
    fmt = args.format

    if fmt == "csv":
        print("场景,模型,TTFT(ms),TPOT(ms),TPS(tok/s),总延迟(ms)")

    # =========================================================================
    # 1. Code Completion (15K input, 50 output)
    # =========================================================================
    d01 = load_json(os.path.join(rd, "01_code_completion_35B", "compensated.json"))
    m01 = get_metrics(d01) if d01 else None
    print_scenario(
        "Code Completion (15K input, 50 output)",
        [("Qwen3.5-35B-A3B-GPTQ-INT4", m01)],
        fmt,
    )

    # =========================================================================
    # 2. Chat Q&A (25K input, 1K output)
    # =========================================================================
    d02 = load_json(os.path.join(rd, "02_chat_27B", "compensated.json"))
    m02 = get_metrics(d02) if d02 else None

    d03_tp1 = load_json(os.path.join(rd, "03_chat_122B", "tp1", "compensated.json"))
    m03_tp1 = get_metrics(d03_tp1) if d03_tp1 else None

    d03_tp2 = load_json(os.path.join(rd, "03_chat_122B", "tp2", "compensated.json"))
    m03_tp2 = get_metrics(d03_tp2) if d03_tp2 else None

    print_scenario(
        "Chat Q&A (25K input, 1K output)",
        [
            ("Qwen3.5-27B", m02),
            ("Qwen3.5-122B-A10B-GPTQ-INT4 TP=1", m03_tp1),
            ("Qwen3.5-122B-A10B-GPTQ-INT4 TP=2", m03_tp2),
        ],
        fmt,
    )

    # =========================================================================
    # 3. Agent Full Task (100K first + 9×20K hit)
    # =========================================================================
    # First call (100K input, 30K output)
    agent_first = {}
    for scenario, tp_list in [("04_agent_122B", [1, 2]), ("05_agent_397B", [2, 4])]:
        for tp in tp_list:
            d = load_json(os.path.join(rd, scenario, f"tp{tp}", "compensated.json"))
            agent_first[(scenario, tp)] = get_metrics(d) if d else None

    # Hit calls (20K input, 30K output)
    agent_hit = {}
    for scenario, tp_list in [("04b_agent_hit_122B", [1, 2]), ("05b_agent_hit_397B", [2, 4])]:
        for tp in tp_list:
            d = load_json(os.path.join(rd, scenario, f"tp{tp}", "compensated.json"))
            agent_hit[(scenario, tp)] = get_metrics(d) if d else None

    # Map hit scenarios to first-call scenarios
    hit_map = {
        ("04b_agent_hit_122B", 1): ("04_agent_122B", 1),
        ("04b_agent_hit_122B", 2): ("04_agent_122B", 2),
        ("05b_agent_hit_397B", 2): ("05_agent_397B", 2),
        ("05b_agent_hit_397B", 4): ("05_agent_397B", 4),
    }

    # Model display names
    model_names = {
        ("04_agent_122B", 1): "Qwen3.5-122B-A10B-GPTQ-INT4 TP=1",
        ("04_agent_122B", 2): "Qwen3.5-122B-A10B-GPTQ-INT4 TP=2",
        ("05_agent_397B", 2): "Qwen3.5-397B-A17B-GPTQ-INT4 TP=2",
        ("05_agent_397B", 4): "Qwen3.5-397B-A17B-GPTQ-INT4 TP=4",
    }

    # Compute full task total latency: first + 9 × hit
    if fmt == "markdown":
        print(f"\n### Agent Full Task (1M tokens = 100K first + 9×100K@80%hit)")
        print(f"\n**全任务总延迟:**\n")
        print(f"| 模型 | 第一次(100K) | 后续×9(20K) | 全任务总延迟 |")
        print(f"|------|-------------|-------------|-------------|")
    else:
        print()

    for first_key in [("04_agent_122B", 1), ("04_agent_122B", 2),
                       ("05_agent_397B", 2), ("05_agent_397B", 4)]:
        hit_key = [hk for hk, fk in hit_map.items() if fk == first_key]
        hit_key = hit_key[0] if hit_key else None

        mf = agent_first.get(first_key)
        mh = agent_hit.get(hit_key) if hit_key else None
        name = model_names[first_key]

        if mf and mh:
            full_total = mf["total"] + 9 * mh["total"]
            if fmt == "csv":
                print(f"Agent全任务,{name},{mf['total']:.1f},{mh['total']:.1f},{full_total:.1f},")
            else:
                print(f"| {name} | {fmt_ms(mf['total'])} | {fmt_ms(mh['total'])} | {fmt_ms(full_total)} |")
        else:
            if fmt == "csv":
                print(f"Agent全任务,{name},N/A,N/A,N/A,")
            else:
                print(f"| {name} | N/A | N/A | N/A |")

    # First call detail
    print_scenario(
        "Agent 第一次调用 (100K input, 30K output)",
        [(model_names[k], agent_first[k]) for k in
         [("04_agent_122B", 1), ("04_agent_122B", 2),
          ("05_agent_397B", 2), ("05_agent_397B", 4)]],
        fmt,
    )

    # Hit call detail
    hit_names = {
        ("04b_agent_hit_122B", 1): "Qwen3.5-122B-A10B-GPTQ-INT4 TP=1",
        ("04b_agent_hit_122B", 2): "Qwen3.5-122B-A10B-GPTQ-INT4 TP=2",
        ("05b_agent_hit_397B", 2): "Qwen3.5-397B-A17B-GPTQ-INT4 TP=2",
        ("05b_agent_hit_397B", 4): "Qwen3.5-397B-A17B-GPTQ-INT4 TP=4",
    }
    print_scenario(
        "Agent 后续调用 (20K input@80%hit, 30K output)",
        [(hit_names[k], agent_hit[k]) for k in
         [("04b_agent_hit_122B", 1), ("04b_agent_hit_122B", 2),
          ("05b_agent_hit_397B", 2), ("05b_agent_hit_397B", 4)]],
        fmt,
    )

    # =========================================================================
    # 4. RAG Repo Understanding (800K input, 3K output)
    # =========================================================================
    d06 = load_json(os.path.join(rd, "06_rag_35B", "compensated.json"))
    m06 = get_metrics(d06) if d06 else None
    print_scenario(
        "RAG Repo Understanding (800K input, 3K output)",
        [("Qwen3.5-35B-A3B-GPTQ-INT4", m06)],
        fmt,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    if fmt == "markdown":
        print("\n---")
        print("*Generated by generate_report.py from compensated results*")


if __name__ == "__main__":
    main()
