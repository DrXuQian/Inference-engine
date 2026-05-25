#!/bin/bash
# Chat问答: 27B FP16, TP=1
# Input: 25K, Output: 1K
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-./results/02_chat_27B/model/rank_0_*L}
MODEL=$(ls -d $MODEL 2>/dev/null | head -1)
OUT=./results/02_chat_27B/trace

bash "$SCRIPT_DIR/capture_trace.sh" "$MODEL" 25600 1024 "$OUT" 10
