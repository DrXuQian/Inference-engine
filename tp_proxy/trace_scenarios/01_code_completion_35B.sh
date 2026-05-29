#!/bin/bash
# 代码补全: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
# Input: 1.5K, Output: 50
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "./results/01_code_completion_35B/model" 2>/dev/null || echo "")}
if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL not set and no split model found. Either:"
    echo "  1. Set MODEL=/path/to/model env var"
    echo "  2. Run bench_scenarios/01_code_completion_35B.sh first to split model"
    exit 1
fi
OUT=./results/01_code_completion_35B/trace

bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 1536 50 "$OUT" 10
