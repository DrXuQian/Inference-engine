#!/bin/bash
# 代码补全: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
# Input: 1.5K, Output: 50
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-./results/01_code_completion_35B/model/rank_0_*L}
MODEL=$(ls -d $MODEL 2>/dev/null | head -1)
OUT=./results/01_code_completion_35B/trace

bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 1536 50 "$OUT" 10
