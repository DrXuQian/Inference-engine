#!/bin/bash
# Chat问答: Qwen3.5-122B-A10B GPTQ-Int4, TP=1 and TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/03_chat_122B

for TP in 1 2; do
    echo "--- TP=$TP ---"
    DIR="$BASE/tp${TP}"
    MODEL_TP=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model" 2>/dev/null || echo "")}
    if [ -z "$MODEL_TP" ]; then
        echo "ERROR: Model not found for TP=$TP, skipping"
        continue
    fi

    COMM="$DIR/comm.json"; [ -f "$COMM" ] && COMM_ARG="--comm-json $COMM" || COMM_ARG=""

    python3 "$SCRIPT_DIR/compensate_ppu.py" \
        --model-dir "$MODEL_TP" \
        --asys-sqlite "$DIR/trace/trace.sqlite" \
        --output-len 1024 \
        $COMM_ARG \
        --output-json "$DIR/compensated.json"
    echo ""
done
