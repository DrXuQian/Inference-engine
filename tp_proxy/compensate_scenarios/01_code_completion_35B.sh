#!/bin/bash
# 代码补全: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/01_code_completion_35B
MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/model")

python3 "$SCRIPT_DIR/compensate_ppu.py" \
    --bench-results "$BASE/bench.json" \
    --model-dir "$MODEL" \
    --asys-sqlite "$BASE/trace/trace.sqlite" \
    --comm-json "./results/01_code_completion_35B/comm.json" \
    --output-json "$BASE/compensated.json"
