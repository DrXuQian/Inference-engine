#!/bin/bash
# Agent hit: Qwen 397B-A17B, TP=2 and TP=4
# Reuses split model and comm.json from 05_agent_397B
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_05=./results/05_agent_397B
OUT=./results/05b_agent_hit_397B

for TP in 2 4; do
    echo "--- TP=$TP ---"
    DIR="$OUT/tp${TP}"

    MODEL_TP=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_05/tp${TP}/model" 2>/dev/null || echo "")}
    if [ -z "$MODEL_TP" ]; then
        echo "ERROR: Model not found for TP=$TP, skipping"
        continue
    fi

    COMM="$BASE_05/tp${TP}/comm.json"; [ -f "$COMM" ] && COMM_ARG="--comm-json $COMM" || COMM_ARG=""

    python3 "$SCRIPT_DIR/compensate_ppu.py" \
        --model-dir "$MODEL_TP" \
        --asys-sqlite "$DIR/trace/trace.sqlite" \
        --output-len 3072 \
        --input-len 20480 \
        --actual-seq-len 102400 \
        $COMM_ARG \
        --output-json "$DIR/compensated.json"
    echo ""
done
