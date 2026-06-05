#!/bin/bash
# 代码补全: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/01_code_completion_35B
MODEL=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/model" 2>/dev/null || echo "")}
if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL not set. Set MODEL=/path/to/model env var"
    exit 1
fi

python3 "$SCRIPT_DIR/compensate_ppu.py" \
    --model-dir "$MODEL" \
    --asys-sqlite "$BASE/trace/trace.sqlite" \
    --output-len 50 \
    --output-json "$BASE/compensated.json"
