#!/bin/bash
# Agent hit: Qwen 397B-A17B, TP=2 and TP=4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/05b_agent_hit_397B

for TP in 2; do
    echo "--- TP=$TP ---"
    DIR="$BASE/tp${TP}"
    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model")
    [ -z "$MODEL" ] && echo "Model not found for TP=$TP" && continue

    python3 "$SCRIPT_DIR/compensate_ppu.py" \
        --bench-results "$DIR/bench.json" \
        --model-dir "$MODEL" \
        --asys-sqlite "$DIR/trace/trace.sqlite" \
        --comm-json "$DIR/comm.json" \
        --actual-seq-len 102400 \
        --output-json "$DIR/compensated.json"
    echo ""
done
