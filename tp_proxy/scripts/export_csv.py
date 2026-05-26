#!/usr/bin/env python3
"""
Export compensated results to CSV matching the deliverable table format.

Usage:
    python export_csv.py [--results-dir ./results] [-o report.csv]
"""

import argparse
import csv
import json
import os
import sys


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_metrics(data: dict) -> dict | None:
    results = data.get("results", [])
    valid = [r for r in results if "error" not in r and "comp_tpot_ms" in r]
    if not valid:
        return None
    r = valid[0]
    ttft = r["comp_ttft_ms"]
    tpot = r["comp_tpot_ms"]
    tps = 1000.0 / tpot if tpot > 0 else 0
    output_tokens = r.get("output_tokens", 64)
    total = r.get("comp_total_ms", ttft + (output_tokens - 1) * tpot)
    return {"ttft": ttft, "tpot": tpot, "tps": tps, "total": total}


def fmt(val, unit="ms", prec=2):
    if val is None:
        return ""
    if unit == "ms":
        if val >= 1000:
            return f"{val / 1000:.{prec}f}s"
        return f"{val:.{prec}f}ms"
    if unit == "tps":
        return f"{val:.1f}"
    return str(val)


def row(m, include_total=True):
    """Return [TTFT, TPOT, TPS, 总延迟] or [TTFT, TPOT, TPS]."""
    if m is None:
        return [""] * (4 if include_total else 3)
    cells = [fmt(m["ttft"]), fmt(m["tpot"], prec=3), fmt(m["tps"], unit="tps")]
    if include_total:
        cells.append(fmt(m["total"]))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("-o", "--output", default=None,
                    help="Output CSV path (default: stdout)")
    args = ap.parse_args()
    rd = args.results_dir

    out = open(args.output, "w", newline="", encoding="utf-8-sig") if args.output else sys.stdout
    w = csv.writer(out)

    # === Code Completion ===
    w.writerow(["Code Completion"])
    w.writerow(["子场景", "单次上下文(token)", "单次产出(token)", "模型"])
    w.writerow(["代码补全", "15K", "50", "Qwen3.5 35B-A3B"])
    w.writerow([])

    d01 = load_json(os.path.join(rd, "01_code_completion_35B", "compensated.json"))
    m01 = get_metrics(d01) if d01 else None
    w.writerow(["模型", "TTFT", "TPOT", "TPS", "总延迟"])
    w.writerow(["Qwen3.5-35B-GPTQ-INT4"] + row(m01))
    w.writerow([])

    # === Chat Q&A ===
    w.writerow(["Chat Q&A"])
    w.writerow(["子场景", "单次上下文(token)", "单次产出(token)", "模型"])
    w.writerow(["Chat问答", "25K", "1K", "27B, 122B-A10B"])
    w.writerow([])

    d02 = load_json(os.path.join(rd, "02_chat_27B", "compensated.json"))
    m02 = get_metrics(d02) if d02 else None
    d03_tp1 = load_json(os.path.join(rd, "03_chat_122B", "tp1", "compensated.json"))
    m03_tp1 = get_metrics(d03_tp1) if d03_tp1 else None
    d03_tp2 = load_json(os.path.join(rd, "03_chat_122B", "tp2", "compensated.json"))
    m03_tp2 = get_metrics(d03_tp2) if d03_tp2 else None

    w.writerow(["模型", "TTFT", "TPOT", "TPS", "总延迟"])
    w.writerow(["Qwen3.5-27B"] + row(m02))
    w.writerow(["Qwen3.5-122B-A10B-GPTQ-INT4-TP1"] + row(m03_tp1))
    w.writerow(["Qwen3.5-122B-A10B-GPTQ-INT4-TP2"] + row(m03_tp2))
    w.writerow([])

    # === Agent Full Task ===
    w.writerow(["Agent Full Task (1M tokens)"])
    w.writerow(["子场景", "单次上下文(token)", "单次产出(token)", "模型"])
    w.writerow(["Agent长程调用", "单次100K, 全任务1M = first 100K(0%hit) + 9*100K(80%hit)", "30K",
                "122B-A10B, 397B-A17B"])
    w.writerow([])

    # Load all agent data
    agent_first = {}
    for scenario, tp_list in [("04_agent_122B", [1, 2]), ("05_agent_397B", [2, 4])]:
        for tp in tp_list:
            d = load_json(os.path.join(rd, scenario, f"tp{tp}", "compensated.json"))
            agent_first[(scenario, tp)] = get_metrics(d) if d else None

    agent_hit = {}
    for scenario, tp_list in [("04b_agent_hit_122B", [1, 2]), ("05b_agent_hit_397B", [2, 4])]:
        for tp in tp_list:
            d = load_json(os.path.join(rd, scenario, f"tp{tp}", "compensated.json"))
            agent_hit[(scenario, tp)] = get_metrics(d) if d else None

    hit_map = {
        ("04_agent_122B", 1): ("04b_agent_hit_122B", 1),
        ("04_agent_122B", 2): ("04b_agent_hit_122B", 2),
        ("05_agent_397B", 2): ("05b_agent_hit_397B", 2),
        ("05_agent_397B", 4): ("05b_agent_hit_397B", 4),
    }

    model_names = {
        ("04_agent_122B", 1): "Qwen3.5-122B-A10B-GPTQ-INT4-TP1",
        ("04_agent_122B", 2): "Qwen3.5-122B-A10B-GPTQ-INT4-TP2",
        ("05_agent_397B", 2): "Qwen3.5-397B-A17B-GPTQ-INT4-TP2",
        ("05_agent_397B", 4): "Qwen3.5-397B-A17B-GPTQ-INT4-TP4",
    }

    # Full task total
    w.writerow(["全任务总延迟:"])
    w.writerow(["模型", "第一次调用总延迟", "后续调用总延迟(×1)", "全任务总延迟"])
    for fk in [("04_agent_122B", 1), ("04_agent_122B", 2),
               ("05_agent_397B", 2), ("05_agent_397B", 4)]:
        hk = hit_map[fk]
        mf = agent_first.get(fk)
        mh = agent_hit.get(hk)
        if mf and mh:
            full = mf["total"] + 9 * mh["total"]
            w.writerow([model_names[fk], fmt(mf["total"]), fmt(mh["total"]), fmt(full)])
        else:
            w.writerow([model_names[fk], "", "", ""])
    w.writerow([])

    # First call detail
    w.writerow(["第一次调用:"])
    w.writerow(["模型", "TTFT", "TPOT", "TPS", "总延迟"])
    for fk in [("04_agent_122B", 1), ("04_agent_122B", 2),
               ("05_agent_397B", 2), ("05_agent_397B", 4)]:
        w.writerow([model_names[fk]] + row(agent_first.get(fk)))
    w.writerow([])

    # Hit call detail
    w.writerow(["后续调用:"])
    w.writerow(["模型", "TTFT", "TPOT", "TPS", "总延迟"])
    for fk in [("04_agent_122B", 1), ("04_agent_122B", 2),
               ("05_agent_397B", 2), ("05_agent_397B", 4)]:
        hk = hit_map[fk]
        w.writerow([model_names[fk]] + row(agent_hit.get(hk)))
    w.writerow([])

    # === RAG ===
    w.writerow(["RAG Repo Understanding"])
    w.writerow(["子场景", "单次上下文(token)", "单次产出(token)", "模型"])
    w.writerow(["RAG仓库理解", "800K", "3K", "Qwen3.5 35B-A3B"])
    w.writerow([])

    d06 = load_json(os.path.join(rd, "06_rag_35B", "compensated.json"))
    m06 = get_metrics(d06) if d06 else None
    w.writerow(["模型", "TTFT", "TPOT", "TPS"])
    w.writerow(["Qwen3.5-35B-GPTQ-INT4"] + row(m06, include_total=False))

    if args.output:
        out.close()
        print(f"CSV saved to: {args.output}")


if __name__ == "__main__":
    main()
