#!/bin/bash
# RAG仓库理解: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
# Input: 800K, Output: 3K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "./results/06_rag_35B/model" 2>/dev/null || echo "")}
if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL not set and no split model found. Either:"
    echo "  1. Set MODEL=/path/to/model env var"
    echo "  2. Run bench_scenarios/06_rag_35B.sh first to split model"
    exit 1
fi
OUT=./results/06_rag_35B/trace

bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 819200 3072 "$OUT" 3
