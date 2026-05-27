#!/bin/bash
# Compensate agent batch sweep
# Uses per-batch trace: batch=1 uses gemvt_op, batch>1 uses gemm kernel
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE_04=./results/04_agent_122B
OUT=./results/04c_agent_batch_122B
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"
# lm_head kernel name changes with batch size
LM_HEAD_B1="gemvt_op"
LM_HEAD_BN="gemm_ktype0_aiu1_mtype1_dtypeBF16xBF16xFP32xBF16xBF16"

for TP in 1 2; do
    echo "--- TP=$TP ---"
    DIR="$OUT/tp${TP}"
    [ ! -d "$DIR" ] && echo "  Not found" && continue

    MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
    [ -z "$MODEL" ] && MODEL=$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model")
    [ -z "$MODEL" ] && echo "  Model not found" && continue

    COMM="$BASE_04/tp${TP}/comm.json"

    for B in $BATCH_LIST; do
        BENCH="$DIR/bench_batch${B}.json"
        [ ! -f "$BENCH" ] && echo "  batch=$B: no bench result" && continue

        # Use per-batch trace if available, fallback to batch=1 trace
        TRACE="$DIR/trace_batch${B}/trace.sqlite"
        [ ! -f "$TRACE" ] && TRACE="$BASE_04/tp${TP}/trace/trace.sqlite"

        # lm_head kernel name: gemvt for batch=1, gemm for batch>1
        if [ $B -eq 1 ]; then
            LM_HEAD="$LM_HEAD_B1"
        else
            LM_HEAD="$LM_HEAD_BN"
        fi

        echo "  batch=$B (lm_head_kernel=$LM_HEAD)"
        COMP_ARGS=""
        [ -f "$TRACE" ] && COMP_ARGS="$COMP_ARGS --asys-sqlite $TRACE"
        [ -f "$COMM" ] && COMP_ARGS="$COMP_ARGS --comm-json $COMM"

        python3 "$SCRIPT_DIR/compensate_ppu.py" \
            --bench-results "$BENCH" \
            --model-dir "$MODEL" \
            --batch-size $B \
            --lm-head-kernel "$LM_HEAD" \
            $COMP_ARGS \
            --output-json "$DIR/compensated_batch${B}.json"
        echo ""
    done
done
