#!/bin/bash
# Chat问答: 27B FP16, TP=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/02_chat_27B
MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/model")

python3 "$SCRIPT_DIR/compensate_ppu.py" \
    --bench-results "$BASE/bench.json" \
    --model-dir "$MODEL" \
    --asys-sqlite "$BASE/trace/trace.sqlite" \
    --comm-json "./results/02_chat_27B/comm.json" \
    --output-json "$BASE/compensated.json"
