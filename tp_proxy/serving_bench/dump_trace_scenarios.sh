#!/bin/bash
# Dump profiler traces for the same 3 scenarios used by bench_scenarios.sh.
#
# This is a debug helper. It uses offline vllm.LLM through scripts/capture_trace.sh
# instead of attaching to a running server, so the trace has clean prefill/decode
# NVTX ranges and can be consumed by the existing trace/compensate tooling.
#
# Scenarios:
#   mainstream:     input=13400, output=500
#   heavy_prefill:  input=79700, output=200
#   heavy_decode:   input=600,   output=10000
#
# Usage:
#   PLATFORM=nvidia bash dump_trace_scenarios.sh /path/to/model 4 [output_dir]
#   SCENARIOS=mainstream bash dump_trace_scenarios.sh /path/to/model 4
#   GPU_IDS=2,3 CAPTURE_ITERS=3 BATCH_SIZE=1 bash dump_trace_scenarios.sh /path/to/model 2
#
# Environment:
#   PLATFORM        nvidia or ppu. Defaults to capture_trace.sh default: ppu.
#   GPU_IDS         CUDA_VISIBLE_DEVICES value. Defaults to 0..TP-1.
#   SCENARIOS       Space-separated subset: mainstream heavy_prefill heavy_decode.
#   CAPTURE_ITERS   Number of prompts passed to capture_trace.sh. Default: 5.
#   BATCH_SIZE      Offline batch size. Default: 1.
#   Extra vLLM debug envs are inherited, including VLLM_FORCE_CUSTOM_AR_*.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_DIR="$(cd "$DIR/../scripts" && pwd)"

MODEL="${1:?Usage: $0 <model> <tp_size> [output_dir]}"
TP="${2:?Usage: $0 <model> <tp_size> [output_dir]}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${3:-./serving_results/trace_$(basename "$MODEL")_tp${TP}_${TIMESTAMP}}"

CAPTURE_ITERS="${CAPTURE_ITERS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SELECTED_SCENARIOS="${SCENARIOS:-mainstream heavy_prefill heavy_decode}"

if [ -n "${GPU_IDS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((TP - 1)))"
    export CUDA_VISIBLE_DEVICES
fi

export TP_SIZE="$TP"

mkdir -p "$OUTPUT_DIR"

declare -a SCENARIO_DEFS=(
    "mainstream,13400,500"
    "heavy_prefill,79700,200"
    "heavy_decode,600,10000"
)

should_run() {
    local name="$1"
    for selected in $SELECTED_SCENARIOS; do
        if [ "$selected" = "$name" ]; then
            return 0
        fi
    done
    return 1
}

echo "============================================"
echo "  Serving Scenario Trace Dump"
echo "  Model: $MODEL"
echo "  TP=$TP, GPUs=${CUDA_VISIBLE_DEVICES:-unset}, Platform=${PLATFORM:-ppu}"
echo "  Output: $OUTPUT_DIR"
echo "  Scenarios: $SELECTED_SCENARIOS"
echo "  Capture iters=$CAPTURE_ITERS, batch=$BATCH_SIZE"
echo "============================================"

for scenario in "${SCENARIO_DEFS[@]}"; do
    IFS=',' read -r NAME INPUT_LEN OUTPUT_LEN <<< "$scenario"
    if ! should_run "$NAME"; then
        continue
    fi

    TRACE_DIR="$OUTPUT_DIR/$NAME"

    echo ""
    echo "========================================"
    echo "  Trace $NAME: input=$INPUT_LEN, output=$OUTPUT_LEN"
    echo "========================================"

    bash "$SCRIPT_DIR/capture_trace.sh" \
        "$MODEL" \
        "$INPUT_LEN" \
        "$OUTPUT_LEN" \
        "$TRACE_DIR" \
        "$CAPTURE_ITERS" \
        "$BATCH_SIZE"

    echo "[OK] $NAME -> $TRACE_DIR"
done

echo ""
echo "============================================"
echo "  Done. Trace outputs:"
echo "    $OUTPUT_DIR/<scenario>/trace.nsys-rep"
echo "    $OUTPUT_DIR/<scenario>/trace.sqlite"
echo "    $OUTPUT_DIR/<scenario>/bench_trace.txt"
echo "============================================"
