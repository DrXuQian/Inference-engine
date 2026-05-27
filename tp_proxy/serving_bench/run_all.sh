#!/bin/bash
# Run all serving benchmarks against a RUNNING server.
# Start the server first with serve.sh in another terminal.
#
# Usage:
#   # Terminal 1: start server
#   bash serve.sh /path/to/model 2
#
#   # Terminal 2: run benchmarks
#   bash run_all.sh /path/to/model http://127.0.0.1:8000 [output_dir]

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${1:?Usage: $0 <model> <base_url> [output_dir]}"
BASE_URL="${2:?Usage: $0 <model> <base_url> [output_dir]}"
OUTPUT_DIR="${3:-./serving_results/$(basename "$MODEL")}"

mkdir -p "$OUTPUT_DIR"

# Wait for server
echo "Checking server at $BASE_URL ..."
for i in $(seq 1 300); do
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Server ready"
        break
    fi
    [ $i -eq 300 ] && echo "ERROR: server not ready" && exit 1
    sleep 2
done

echo "============================================"
echo "  Serving Benchmark Suite"
echo "  Model: $MODEL"
echo "  Server: $BASE_URL"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

echo ""
echo "=== [1/3] Scenario Benchmarks ==="
bash "$DIR/bench_scenarios.sh" "$MODEL" "$BASE_URL" "$OUTPUT_DIR/scenarios"

echo ""
echo "=== [2/3] Concurrency Sweep ==="
bash "$DIR/bench_concurrency.sh" "$MODEL" "$BASE_URL" "$OUTPUT_DIR/concurrency"

echo ""
echo "=== [3/3] Input Length Sweep ==="
bash "$DIR/bench_input_sweep.sh" "$MODEL" "$BASE_URL" "$OUTPUT_DIR/input_sweep"

echo ""
echo "============================================"
echo "  All done! Results: $OUTPUT_DIR"
echo "============================================"
