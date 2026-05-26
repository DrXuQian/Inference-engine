#!/bin/bash
# Profile vLLM serving with nsys + vllm bench serve.
#
# Usage:
#   bash nsys_serve_profile.sh /path/to/rank_0 [nsys_output_name]
#
# This starts the server under nsys, runs vllm bench serve,
# then kills the server to finalize the nsys trace.

set -euo pipefail

RANK_DIR="${1:?Usage: $0 <model_dir> [nsys_output]}"
NSYS_OUT="${2:-/tmp/nsys_serve_profile}"

export TRITON_BACKENDS_IN_TREE=1

# Clean up any old server
pkill -f "api_server" 2>/dev/null || true
pkill -f "EngineCore" 2>/dev/null || true
sleep 2

# Start server under nsys
echo "[1/4] Starting vllm serve under nsys..."
nsys profile -t cuda --cuda-trace-scope=system-wide \
    --cuda-graph-trace=node \
    --force-overwrite=true -o "${NSYS_OUT}" \
    vllm serve "${RANK_DIR}" \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.9 \
    > /tmp/nsys_serve_stdout.log 2>&1 &
SERVER_PID=$!

cleanup() {
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server
echo "[2/4] Waiting for server..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "       Server ready (${i}s)"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server died. Check /tmp/nsys_serve_stdout.log"
        exit 1
    fi
    sleep 2
done

# Run bench serve
echo "[3/4] Running vllm bench serve..."
vllm bench serve \
    --model "${RANK_DIR}" \
    --max-concurrency 1 \
    --base-url http://127.0.0.1:8000 \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 64 \
    --num-prompts 50 \
    --request-rate 10 \
    --percentile-metrics ttft,tpot,itl \
    --trust-remote-code \
    2>&1 | tee /tmp/bench_serve_result.txt

# Kill server → nsys finalizes trace
echo "[4/4] Stopping server, finalizing nsys trace..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
trap - EXIT

echo ""
echo "nsys report: ${NSYS_OUT}.nsys-rep"
echo "Run: python3 nsys_kernel_classify.py --nsys-rep ${NSYS_OUT}.nsys-rep --num-layers 10 --original-layers 40"
