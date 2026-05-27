#!/bin/bash
# Compensate agent batch sweep: per-batch trace and comm
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_04=./results/04_agent_122B
OUT=./results/04c_agent_batch_122B
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"

for TP in 1 2; do
    echo "--- TP=$TP ---"
    DIR="$OUT/tp${TP}"
    [ ! -d "$DIR" ] && echo "  Not found" && continue

    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
    [ -z "$MODEL" ] && MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model")
    [ -z "$MODEL" ] && echo "  Model not found" && continue

    for B in $BATCH_LIST; do
        LOG="$DIR/batch${B}.log"
        [ ! -f "$LOG" ] && echo "  batch=$B: no bench log" && continue

        # Extract TPOT/TTFT from bench log
        TPOT=$(grep -i "median.*tpot\|median.*inter-token" "$LOG" | tail -1 | awk -F: '{print $NF}' | tr -d ' ')
        TTFT=$(grep -i "median.*ttft" "$LOG" | tail -1 | awk -F: '{print $NF}' | tr -d ' ')
        [ -z "$TPOT" ] && echo "  batch=$B: no TPOT" && continue

        echo "  batch=$B: raw TPOT=${TPOT}ms TTFT=${TTFT:-0}ms"

        # Create bench.json
        python3 -c "
import json
r = {'input_len': 4096, 'output_len': 1500, 'output_tokens': 1500,
     'tpot_median_ms': $TPOT, 'ttft_median_ms': ${TTFT:-0}}
json.dump({'results': [r]}, open('$DIR/bench_batch${B}.json', 'w'), indent=2)
"

        # Per-batch trace and comm
        TRACE="$DIR/trace_batch${B}/trace.sqlite"
        COMM="$DIR/comm_batch${B}.json"
        COMP_ARGS=""
        [ -f "$TRACE" ] && COMP_ARGS="$COMP_ARGS --asys-sqlite $TRACE"
        [ -f "$COMM" ] && COMP_ARGS="$COMP_ARGS --comm-json $COMM"

        python3 "$SCRIPT_DIR/compensate_ppu.py" \
            --bench-results "$DIR/bench_batch${B}.json" \
            --model-dir "$MODEL" \
            $COMP_ARGS \
            --output-json "$DIR/compensated_batch${B}.json"
        echo ""
    done
done
