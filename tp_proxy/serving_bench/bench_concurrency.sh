#!/bin/bash
# Sweep concurrency levels against a running vllm server.
#
# Usage:
#   bash bench_concurrency.sh /path/to/model [base_url] [output_dir]
#   CONCURRENCY_LIST="1 2 4 8 16 32" bash bench_concurrency.sh /path/to/model

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [base_url] [output_dir]}"
BASE_URL="${2:-http://127.0.0.1:8000}"
OUTPUT_DIR="${3:-./serving_results/concurrency}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-4096}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-1500}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
REQUEST_RATE="${REQUEST_RATE:-10}"
CONCURRENCY_LIST="${CONCURRENCY_LIST:-1 2 4 8 16 32 64}"

mkdir -p "$OUTPUT_DIR"

# Wait for server
echo "Checking server at $BASE_URL ..."
for i in $(seq 1 60); do
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Server ready"
        break
    fi
    [ $i -eq 60 ] && echo "ERROR: server not ready" && exit 1
    sleep 2
done

echo ""
echo "============================================"
echo "  Concurrency Sweep"
echo "  Model: $MODEL"
echo "  Input=$RANDOM_INPUT_LEN, Output=$RANDOM_OUTPUT_LEN"
echo "  Concurrency: $CONCURRENCY_LIST"
echo "============================================"

for C in $CONCURRENCY_LIST; do
    LOG="$OUTPUT_DIR/c${C}.log"

    echo ""
    echo "----------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] max-concurrency = $C"
    echo "----------------------------------------------"

    set +e
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    vllm bench serve \
        --model "$MODEL" \
        --max-concurrency "$C" \
        --base-url "$BASE_URL" \
        --dataset-name random \
        --random-input-len "$RANDOM_INPUT_LEN" \
        --random-output-len "$RANDOM_OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --percentile-metrics ttft,tpot,throughput \
        --trust-remote-code \
        2>&1 | tee "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Failed for concurrency=$C (exit code: $EXIT_CODE)"
    else
        echo "[OK] concurrency=$C → $LOG"
    fi

    sleep 5
done

echo ""
echo "============================================"
echo "  Done! Results in $OUTPUT_DIR/"
echo "============================================"
