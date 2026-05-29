#!/bin/bash
# Chat问答: 27B FP16, TP=1
# Input: 25K, Output: 1K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "./results/02_chat_27B/model" 2>/dev/null || echo "")}
if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL not set and no split model found. Either:"
    echo "  1. Set MODEL=/path/to/model env var"
    echo "  2. Run bench_scenarios/02_chat_27B.sh first to split model"
    exit 1
fi
OUT=./results/02_chat_27B/trace

bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 25600 1024 "$OUT" 10
