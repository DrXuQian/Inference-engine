#!/bin/bash
# Chat问答: Qwen3.5-122B-A10B GPTQ-Int4, TP=1 and TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/03_chat_122B

for TP in 1 2; do
    echo "--- TP=$TP ---"
    DIR="$BASE/tp${TP}"
    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model")
    [ -z "$MODEL" ] && echo "Model not found for TP=$TP" && continue

    python3 "$SCRIPT_DIR/compensate_ppu.py" \
        --bench-results "$DIR/bench.json" \
        --model-dir "$MODEL" \
        --asys-sqlite "$DIR/trace/trace.sqlite" \
        --comm-json "$DIR/comm.json" \
        --output-json "$DIR/compensated.json"
    echo ""
done
