#!/bin/bash
# Run all serving benchmarks: scenarios + concurrency sweep + input sweep
#
# Prerequisites: vllm serve is already running (use start_server.sh)
#
# Usage:
#   bash run_all.sh /path/to/model [base_url] [output_dir]

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${1:?Usage: $0 <model> [base_url] [output_dir]}"
BASE_URL="${2:-http://127.0.0.1:8000}"
OUTPUT_DIR="${3:-./serving_results}"

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
echo "  All done!"
echo ""
echo "  Generate plots:"
echo "    python $DIR/plot_input_sweep.py --csv $OUTPUT_DIR/input_sweep/summary.csv --label '$(basename $MODEL)'"
echo "    python $DIR/plot_concurrency.py --log-dir $OUTPUT_DIR/concurrency/"
echo "============================================"
