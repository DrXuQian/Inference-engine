#!/bin/bash
# Capture asys/nsys trace from vllm serve + bench serve.
#
# Usage:
#   # PPU
#   bash capture_trace.sh /path/to/model 25600 1024 ./results/trace
#
#   # NVIDIA
#   PLATFORM=nvidia bash capture_trace.sh /path/to/model 25600 1024 ./results/trace

set -euo pipefail

MODEL="$(readlink -f "${1:?Usage: $0 <model_dir> <input_len> <output_len> <output_dir> [num_prompts]}")"
INPUT_LEN="${2:?}"
OUTPUT_LEN="${3:?}"
OUT_DIR="${4:?}"
NUM_PROMPTS="${5:-10}"
PORT=8200
MAX_MODEL_LEN=$((INPUT_LEN + OUTPUT_LEN + 64))

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

PLATFORM="${PLATFORM:-ppu}"

# Start server under profiler
echo "[1/5] Starting server under profiler (max_model_len=$MAX_MODEL_LEN)..."
if [ "$PLATFORM" = "ppu" ]; then
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    asys profile -o "$OUT_DIR/trace.report" -f true \
        -t hggc,acdnn,acblas \
        vllm serve "$MODEL" \
        --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 \
        --max-model-len $MAX_MODEL_LEN \
        --trust-remote-code --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 &
else
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    TRITON_BACKENDS_IN_TREE=1 \
    nsys profile -t cuda --cuda-trace-scope=system-wide --cuda-graph-trace=node \
        --force-overwrite=true -o "$OUT_DIR/trace" \
        vllm serve "$MODEL" \
        --host 127.0.0.1 --port $PORT --tensor-parallel-size 1 \
        --max-model-len $MAX_MODEL_LEN \
        --trust-remote-code --no-enable-prefix-caching \
        --gpu-memory-utilization 0.9 &
fi
SRV_PID=$!
cleanup() { kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; }
trap cleanup EXIT

# Wait for server
echo "[2/5] Waiting for server..."
while ! curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 $SRV_PID 2>/dev/null; then
        echo "ERROR: server died"; exit 1
    fi
    sleep 2
done
echo "       Server ready"

# Run bench serve
echo "[3/5] Running bench serve (${NUM_PROMPTS} prompts, input=${INPUT_LEN}, output=${OUTPUT_LEN})..."
MAX_MODEL_LEN=$((INPUT_LEN + OUTPUT_LEN + 64))
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm bench serve \
    --model "$MODEL" \
    --max-concurrency 1 --base-url "http://127.0.0.1:${PORT}" \
    --dataset-name random \
    --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --request-rate 5 \
    --trust-remote-code 2>&1 | tee "$OUT_DIR/bench_serve.txt"

# Kill server → profiler saves trace
echo "[4/5] Stopping server..."
kill $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null || true
trap - EXIT

# Wait for profiler to finish writing
echo "[5/5] Waiting for trace file to be written..."
sleep 5

echo "       Exporting sqlite..."
if [ "$PLATFORM" = "ppu" ]; then
    # asys saves as trace.report.asysrep
    ASYS_REP=""
    for i in 1 2 3 4 5; do
        if [ -f "$OUT_DIR/trace.report.asysrep" ]; then
            ASYS_REP="$OUT_DIR/trace.report.asysrep"
        elif [ -f "$OUT_DIR/trace.report" ]; then
            ASYS_REP="$OUT_DIR/trace.report"
        fi
        if [ -n "$ASYS_REP" ]; then break; fi
        echo "       Waiting for asys report file... (attempt $i)"
        sleep 3
    done
    if [ -z "$ASYS_REP" ]; then
        echo "ERROR: asys report file not found in $OUT_DIR"
        ls -la "$OUT_DIR/" || true
        exit 1
    fi
    echo "       Found: $ASYS_REP"
    asys export --force-overwrite true -o "$OUT_DIR/trace.sqlite" "$ASYS_REP" || {
        echo "ERROR: asys export failed (exit code $?)"
        exit 1
    }
else
    nsys stats -r cuda_gpu_kern_sum --format csv --force-export=true \
        "$OUT_DIR/trace.nsys-rep" > /dev/null 2>&1
    # nsys auto-creates .sqlite alongside .nsys-rep
    SQLITE=$(ls "$OUT_DIR/trace.sqlite" 2>/dev/null || ls "$OUT_DIR/"*.sqlite 2>/dev/null | head -1)
    if [ -n "$SQLITE" ]; then
        echo "       SQLite: $SQLITE"
    fi
fi

echo ""
echo "Done. Output:"
echo "  Trace: $OUT_DIR/trace.sqlite"
echo "  Bench: $OUT_DIR/bench_serve.txt"
echo ""
echo "Next: python compensate_ppu.py --asys-sqlite $OUT_DIR/trace.sqlite ..."
