#!/bin/bash
# Compensate agent batch sweep
# Uses batch=1 trace (tail from 05_agent), applies batch scaling:
#   lm_head × 1.1^batch, sampling × batch
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_04=./results/05_agent_397B
OUT=./results/05c_agent_batch_397B
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

for TP in 2; do
    echo "--- TP=$TP ---"
    DIR="$OUT/tp${TP}"
    [ ! -d "$DIR" ] && echo "  Not found" && continue

    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
    [ -z "$MODEL" ] && MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model")
    [ -z "$MODEL" ] && echo "  Model not found" && continue

    # Use batch=1 trace from 05_agent (tail baseline)
    TRACE="$BASE_04/tp${TP}/trace/trace.sqlite"
    COMM="$BASE_04/tp${TP}/comm.json"

    for B in $BATCH_LIST; do
        BENCH="$DIR/bench_batch${B}.json"
        [ ! -f "$BENCH" ] && echo "  batch=$B: no bench result" && continue

        echo "  batch=$B"
        COMP_ARGS=""
        [ -f "$TRACE" ] && COMP_ARGS="$COMP_ARGS --asys-sqlite $TRACE"
        [ -f "$COMM" ] && COMP_ARGS="$COMP_ARGS --comm-json $COMM"

        python3 "$SCRIPT_DIR/compensate_ppu.py" \
            --bench-results "$BENCH" \
            --model-dir "$MODEL" \
            --batch-size $B \
            $COMP_ARGS \
            --output-json "$DIR/compensated_batch${B}.json"
        echo ""
    done
done
