#!/bin/bash
# Agent hit: Qwen 397B-A17B, TP=2 and TP=4, input=20K
# Reuses split model from 05_agent_397B
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_05=./results/05_agent_397B
OUT=./results/05b_agent_hit_397B

for TP in 2 4; do
    echo "--- TP=$TP ---"
    PRUNED=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_05/tp${TP}/model" 2>/dev/null || echo "")}
    if [ -z "$PRUNED" ]; then
        echo "ERROR: MODEL not set and no split model found for TP=$TP. Either:"
        echo "  1. Set MODEL=/path/to/model env var"
        echo "  2. Run bench_scenarios/05_agent_397B.sh first to split model"
        exit 1
    fi
    TRACE_DIR="$OUT/tp${TP}/trace"
    bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" 20480 3072 "$TRACE_DIR" 5
    echo ""
done
