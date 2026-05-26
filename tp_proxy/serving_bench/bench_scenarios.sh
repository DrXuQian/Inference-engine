#!/bin/bash
# Benchmark 3 serving scenarios against a running vllm server.
#
# Scenarios (input = non-cached prefill length):
#   1. Mainstream:     13.4K input, 500 output
#   2. Heavy Prefill:  79.7K input, 200 output
#   3. Heavy Decode:   0.6K input, 10K output
#
# Usage:
#   bash bench_scenarios.sh /path/to/model [base_url] [output_dir]

set -euo pipefail

MODEL="${1:?Usage: $0 <model> [base_url] [output_dir]}"
BASE_URL="${2:-http://127.0.0.1:8000}"
OUTPUT_DIR="${3:-./serving_results}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"

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

# Scenario definitions: name, input_len, output_len
declare -a SCENARIOS=(
    "mainstream,13400,500"
    "heavy_prefill,79700,200"
    "heavy_decode,600,10000"
)

echo ""
echo "============================================"
echo "  Serving Benchmark: 3 Scenarios"
echo "  Model: $MODEL"
echo "  Prompts per scenario: $NUM_PROMPTS"
echo "============================================"

for scenario in "${SCENARIOS[@]}"; do
    IFS=',' read -r NAME INPUT_LEN OUTPUT_LEN <<< "$scenario"
    LOG="$OUTPUT_DIR/${NAME}.log"

    echo ""
    echo "========================================"
    echo "  $NAME: input=$INPUT_LEN, output=$OUTPUT_LEN"
    echo "========================================"

    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    vllm bench serve \
        --model "$MODEL" \
        --base-url "$BASE_URL" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --max-concurrency 1 \
        --request-rate inf \
        --percentile-metrics ttft,tpot,throughput \
        --trust-remote-code \
        2>&1 | tee "$LOG"

    echo "[OK] $NAME → $LOG"
    sleep 3
done

echo ""
echo "============================================"
echo "  Done! Results in $OUTPUT_DIR/"
echo "============================================"
