#!/bin/bash
# RAG仓库理解: Qwen3.5-35B-A3B GPTQ-Int4, TP=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/06_rag_35B
MODEL=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE/model" 2>/dev/null || echo "")}
if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL not set. Set MODEL=/path/to/model env var"
    exit 1
fi

COMM="$BASE/comm.json"; [ -f "$COMM" ] && COMM_ARG="--comm-json $COMM" || COMM_ARG=""

python3 "$SCRIPT_DIR/compensate_ppu.py" \
    --model-dir "$MODEL" \
    --asys-sqlite "$BASE/trace/trace.sqlite" \
    --output-len 3072 \
    $COMM_ARG \
    --output-json "$BASE/compensated.json"
