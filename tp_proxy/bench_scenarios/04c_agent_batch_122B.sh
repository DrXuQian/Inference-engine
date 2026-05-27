#!/bin/bash
# Agent batch sweep: Qwen3.5-122B-A10B GPTQ-Int4, TP=1 and TP=2
# Tests batch=1,2,4,8 with input=4096, output=1500
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4}
GPU_MEM=${GPU_MEM:-16}
BASE_04=./results/04_agent_122B
OUT=./results/04c_agent_batch_122B
INPUT_LEN=${INPUT_LEN:-4096}
OUTPUT_LEN=${OUTPUT_LEN:-1500}
NUM_PROMPTS=${NUM_PROMPTS:-20}
REQUEST_RATE=${REQUEST_RATE:-10}
BATCH_LIST="${BATCH_LIST:-1 2 4 8}"
PORT=8200

echo "=== Agent Batch Sweep: Qwen3.5-122B-A10B ==="
echo "Input=$INPUT_LEN, Output=$OUTPUT_LEN, Batch: $BATCH_LIST"

for TP in 1 2; do
    echo ""
    echo "--- TP=$TP ---"
    TP_DIR="$OUT/tp${TP}"
    mkdir -p "$TP_DIR"

    # Reuse split model from 04_agent
    PRUNED=$(bash "$SCRIPT_DIR/get_model_path.sh" "$BASE_04/tp${TP}/model")
    if [ -z "$PRUNED" ]; then
        echo "  Split model not found, splitting..."
        python3 "$SCRIPT_DIR/split_and_prune.py" \
            --model-dir "$MODEL" --tp-size $TP \
            --gpu-memory-gb "$GPU_MEM" --max-seq-len $((INPUT_LEN + OUTPUT_LEN + 2048)) \
            --output-dir "$TP_DIR/model"
        PRUNED=$(bash "$SCRIPT_DIR/get_model_path.sh" "$TP_DIR/model")
    fi
    [ -z "$PRUNED" ] && echo "  ERROR: no model" && continue

    # Start server
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    vllm serve "$PRUNED" \
        --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 \
        --max-model-len $((INPUT_LEN + OUTPUT_LEN + 64)) \
        --trust-remote-code --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 > "$TP_DIR/server.log" 2>&1 &
    SRV_PID=$!

    echo "  Waiting for server (PID=$SRV_PID)..."
    while ! curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
        if ! kill -0 $SRV_PID 2>/dev/null; then
            echo "  ERROR: server died"; tail -10 "$TP_DIR/server.log"; break
        fi
        sleep 2
    done

    if ! curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
        continue
    fi
    echo "  Server ready"

    for B in $BATCH_LIST; do
        LOG="$TP_DIR/batch${B}.log"
        echo ""
        echo "  [TP=$TP batch=$B]"
        set +e
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
        vllm bench serve \
            --model "$PRUNED" \
            --max-concurrency "$B" \
            --base-url "http://127.0.0.1:${PORT}" \
            --dataset-name random \
            --random-input-len "$INPUT_LEN" \
            --random-output-len "$OUTPUT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --request-rate "$REQUEST_RATE" \
            --percentile-metrics ttft,tpot,throughput \
            --trust-remote-code \
            2>&1 | tee "$LOG"
        set -e
        sleep 3
    done

    kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null || true
    echo "  Server stopped"
done

echo ""
echo "Done: $OUT/"
