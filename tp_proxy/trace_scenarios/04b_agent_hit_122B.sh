#!/bin/bash
# Agent hit: Qwen3.5-122B-A10B, TP=1 and TP=2, input=20K
# Reuses split model from 04_agent_122B
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_04=./results/04_agent_122B
OUT=./results/04b_agent_hit_122B

for TP in 1 2; do
    echo "--- TP=$TP ---"
    PRUNED=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model" 2>/dev/null || echo "")}
    if [ -z "$PRUNED" ]; then
        echo "ERROR: MODEL not set and no split model found for TP=$TP. Either:"
        echo "  1. Set MODEL=/path/to/model env var"
        echo "  2. Run bench_scenarios/04_agent_122B.sh first to split model"
        exit 1
    fi
    TRACE_DIR="$OUT/tp${TP}/trace"
    bash "$SCRIPT_DIR/capture_trace.sh" "$PRUNED" 20480 3072 "$TRACE_DIR" 5
    echo ""
done
