#!/bin/bash
# RAG仓库理解: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
# Input: 800K, Output: 3K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-./results/06_rag_35B/model/rank_0_*L}
MODEL=$(ls -d $MODEL 2>/dev/null | head -1)
OUT=./results/06_rag_35B/trace

bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 819200 3072 "$OUT" 3
