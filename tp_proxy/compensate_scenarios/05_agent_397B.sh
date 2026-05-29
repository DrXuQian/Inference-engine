#!/bin/bash
# Agent长程调用: Qwen 397B-A17B GPTQ-Int4, TP=2 and TP=4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/05_agent_397B

for TP in 2; do
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
        --output-len 3072 \
        $COMM_ARG \
        --output-json "$DIR/compensated.json"
    echo ""
done
