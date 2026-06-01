#!/bin/bash
# Start vllm serve + run all serving benchmarks + stop server.
#
# Usage:
#   bash run_all.sh /path/to/model 2                    # TP=2
#   bash run_all.sh /path/to/model 4 ./my_results       # TP=4, custom output
#   GPU_IDS=2,3 bash run_all.sh /path/to/model 2        # custom GPUs

set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${1:?Usage: $0 <model> <tp_size> [output_dir]}"
TP="${2:?Usage: $0 <model> <tp_size> [output_dir]}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${3:-./serving_results/$(basename "$MODEL")_tp${TP}_${TIMESTAMP}}"
PORT="${PORT:-8000}"
GPU_IDS="${GPU_IDS:-$(seq -s, 0 $((TP-1)))}"
BASE_URL="http://127.0.0.1:${PORT}"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  Serving Benchmark Suite"
echo "  Model: $MODEL"
echo "  TP=$TP, GPUs=$GPU_IDS, Port=$PORT"
echo "  Output: $OUTPUT_DIR"
echo "============================================"

# === Start server ===
echo ""
echo "=== Starting vllm serve ==="
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
vllm serve "$MODEL" \
    --host 127.0.0.1 --port $PORT \
    --tensor-parallel-size $TP \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9 > "$OUTPUT_DIR/server.log" 2>&1 &
SRV_PID=$!
echo "Server PID=$SRV_PID, log: $OUTPUT_DIR/server.log"

cleanup() {
    echo ""
    echo "Stopping server (PID=$SRV_PID)..."
    kill $SRV_PID 2>/dev/null
    wait $SRV_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server (up to 600s for large models)
echo "Waiting for server..."
SERVER_READY=0
for i in $(seq 1 300); do
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Server ready (${i}×2s = $((i*2))s)"
        SERVER_READY=1
        break
    fi
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "ERROR: server died. Last 30 lines of log:"
        tail -30 "$OUTPUT_DIR/server.log"
        exit 1
    fi
    sleep 2
done
if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: server not ready after 600s. Last 30 lines of log:"
    tail -30 "$OUTPUT_DIR/server.log"
    exit 1
fi

# === Run benchmarks ===
echo ""
echo "=== [1/3] Scenario Benchmarks ==="
bash "$DIR/bench_scenarios.sh" "$MODEL" "$BASE_URL" "$OUTPUT_DIR/scenarios"

echo ""
echo "=== [2/3] Concurrency Sweep ==="
bash "$DIR/bench_concurrency.sh" "$MODEL" "$BASE_URL" "$OUTPUT_DIR/concurrency"

echo ""
echo "=== [3/3] Input Length Sweep ==="
bash "$DIR/bench_input_sweep.sh" "$MODEL" "$BASE_URL" "$OUTPUT_DIR/input_sweep"

# Server stopped by trap
echo ""
echo "============================================"
echo "  All done! Results: $OUTPUT_DIR"
echo ""
echo "  Plots:"
echo "    python $DIR/plot_input_sweep.py --csv $OUTPUT_DIR/input_sweep/summary.csv --label 'TP=$TP'"
echo "    python $DIR/plot_concurrency.py --log-dir $OUTPUT_DIR/concurrency/ --label 'TP=$TP'"
echo ""
echo "  Scale to target:"
echo "    python $DIR/scale_report.py --input-csv $OUTPUT_DIR/input_sweep/summary.csv \\"
echo "        --scenario-dir $OUTPUT_DIR/scenarios/ \\"
echo "        --src-flops <SRC_TFLOPS> --src-bw <SRC_BW> --src-link-bw <SRC_LINK> \\"
echo "        --tgt-flops <TGT_TFLOPS> --tgt-bw <TGT_BW> --tgt-link-bw <TGT_LINK> \\"
echo "        --tp-size $TP"
echo "============================================"
