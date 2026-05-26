#!/bin/bash
# Sweep input lengths (2K to 128K) against a running vllm server.
# Measures TTFT and TPOT at each input length.
#
# Usage:
#   bash bench_input_sweep.sh /path/to/model [base_url] [output_dir]
#   INPUT_LENS="2048 4096 8192" bash bench_input_sweep.sh /path/to/model

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [base_url] [output_dir]}"
BASE_URL="${2:-http://127.0.0.1:8000}"
OUTPUT_DIR="${3:-./serving_results/input_sweep}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
INPUT_LENS="${INPUT_LENS:-2048 4096 8192 16384 32768 65536 131072}"

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
echo "  Input Length Sweep (TTFT & TPOT)"
echo "  Model: $MODEL"
echo "  Output=$OUTPUT_LEN, Prompts=$NUM_PROMPTS"
echo "  Input lengths: $INPUT_LENS"
echo "============================================"

# CSV header
SUMMARY="$OUTPUT_DIR/summary.csv"
echo "input_len,ttft_median_ms,ttft_p99_ms,tpot_median_ms,tpot_p99_ms" > "$SUMMARY"

for IL in $INPUT_LENS; do
    LOG="$OUTPUT_DIR/input_${IL}.log"

    echo ""
    echo "----------------------------------------------"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] input_len = $IL"
    echo "----------------------------------------------"

    set +e
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    vllm bench serve \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --dataset-name random \
        --random-input-len "$IL" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency 1 \
        --request-rate inf \
        --percentile-metrics ttft,tpot \
        --trust-remote-code \
        2>&1 | tee "$LOG"
    EXIT_CODE=${PIPESTATUS[0]}
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Failed for input_len=$IL"
        echo "$IL,,,," >> "$SUMMARY"
    else
        # Extract metrics from log
        TTFT_MED=$(grep -i "median ttft" "$LOG" | tail -1 | grep -oP '[\d.]+(?=\s*ms)' || echo "")
        TTFT_P99=$(grep -i "p99 ttft" "$LOG" | tail -1 | grep -oP '[\d.]+(?=\s*ms)' || echo "")
        TPOT_MED=$(grep -i "median.*inter-token" "$LOG" | tail -1 | grep -oP '[\d.]+(?=\s*ms)' || echo "")
        TPOT_P99=$(grep -i "p99.*inter-token" "$LOG" | tail -1 | grep -oP '[\d.]+(?=\s*ms)' || echo "")
        echo "$IL,$TTFT_MED,$TTFT_P99,$TPOT_MED,$TPOT_P99" >> "$SUMMARY"
        echo "[OK] input=$IL → TTFT=${TTFT_MED}ms, TPOT=${TPOT_MED}ms"
    fi

    sleep 3
done

echo ""
echo "============================================"
echo "  Summary: $SUMMARY"
cat "$SUMMARY"
echo "============================================"
