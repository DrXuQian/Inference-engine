#!/usr/bin/env bash
# Vision benchmark wrapper.
#
# Usage:
#   bash kernel_bench/vision/run_vision_bench.sh [output_dir] [component]
#
# Components:
#   vit       4x 480x480 input, synthetic ViT-L-like trunk
#   clipdino 1x 480x480 input, CLIP ViT-B/16 config + DINO-style head
#   all       run both components serially
#
# Environment:
#   PLATFORM=ppu|cuda   default: ppu
#   DTYPE=bf16          fp32|fp16|bf16
#   WARMUP=10
#   ITERS=30
#   CUDA_GRAPH=0       set to 1 to match vla_bench.py --cuda-graph
#   TORCH_TRACE=0      optional torch.jit.trace, off by default

set -euo pipefail

OUT_ROOT="${1:-./results/vision_bench}"
COMPONENT="${2:-all}"
PLATFORM="${PLATFORM:-ppu}"
DTYPE="${DTYPE:-bf16}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-30}"
CUDA_GRAPH="${CUDA_GRAPH:-0}"
TORCH_TRACE="${TORCH_TRACE:-0}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_ROOT}_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

PY_ARGS=(--dtype "$DTYPE" --warmup "$WARMUP" --iters "$ITERS")
if [ "$CUDA_GRAPH" = "1" ]; then
  PY_ARGS+=(--cuda-graph)
fi
if [ "$TORCH_TRACE" = "1" ]; then
  PY_ARGS+=(--torch-trace)
fi

echo "============================================================"
echo "Vision Benchmark"
echo "  output:      $OUT_DIR"
echo "  component:   $COMPONENT"
echo "  platform:    $PLATFORM"
echo "  dtype:       $DTYPE"
echo "  torch_trace: $TORCH_TRACE"
echo "  cuda_graph:  $CUDA_GRAPH"
echo "============================================================"

run_component() {
  local component="$1"
  local comp_dir="$OUT_DIR/$component"
  mkdir -p "$comp_dir"

  local component_args=()
  if [ "$component" = "vit" ]; then
    component_args=(--component vit --vit-batch 4 --vit-image-size 480)
  elif [ "$component" = "clipdino" ]; then
    component_args=(--component clipdino --clipdino-batch 1 --clipdino-image-size 480)
  else
    echo "Unknown component: $component" >&2
    exit 1
  fi

  echo ""
  echo "=== [$component] trace ==="
  if [ "$PLATFORM" = "ppu" ]; then
    asys profile -o "$comp_dir/trace.report" -f true \
      -t hggc,acdnn,acblas,hgtx \
      python3 "$SCRIPT_DIR/vision_bench.py" \
        "${component_args[@]}" "${PY_ARGS[@]}" \
        --output-json "$comp_dir/results.json" \
      2>&1 | tee "$comp_dir/bench.log"

    sleep 2
    local asys_rep
    asys_rep="$(find "$comp_dir" -name "*.asysrep" -o -name "trace.report" | head -1 || true)"
    if [ -n "$asys_rep" ]; then
      asys export --force-overwrite true \
        -o "$comp_dir/trace.sqlite" "$asys_rep" 2>/dev/null || true
    fi
  else
    nsys profile --force-overwrite=true \
      --sample=none --cpuctxsw=none --trace=cuda,nvtx \
      --cuda-graph-trace=node \
      -o "$comp_dir/trace" \
      python3 "$SCRIPT_DIR/vision_bench.py" \
        "${component_args[@]}" "${PY_ARGS[@]}" \
        --output-json "$comp_dir/results.json" \
      2>&1 | tee "$comp_dir/bench.log"

    if [ -f "$comp_dir/trace.nsys-rep" ]; then
      nsys export --type sqlite --force-overwrite=true \
        -o "$comp_dir/trace.sqlite" "$comp_dir/trace.nsys-rep" >/dev/null
    fi
  fi

  echo "[$component] done: $comp_dir/results.json"
}

case "$COMPONENT" in
  vit)
    run_component vit
    ;;
  clipdino)
    run_component clipdino
    ;;
  all)
    run_component vit
    run_component clipdino

    # Auto-generate report, matching vla/run_vla_bench.sh behavior:
    # prefer trace.sqlite, fall back to results.json.
    echo ""
    echo "=== [3/2] Vision Report ==="
    REPORT_ARGS=""
    VIT_SQLITE="$OUT_DIR/vit/trace.sqlite"
    CLIPDINO_SQLITE="$OUT_DIR/clipdino/trace.sqlite"

    [ -f "$VIT_SQLITE" ] && REPORT_ARGS="$REPORT_ARGS --vit-trace $VIT_SQLITE" || \
      { [ -f "$OUT_DIR/vit/results.json" ] && REPORT_ARGS="$REPORT_ARGS --vit-json $OUT_DIR/vit/results.json"; }
    [ -f "$CLIPDINO_SQLITE" ] && REPORT_ARGS="$REPORT_ARGS --clipdino-trace $CLIPDINO_SQLITE" || \
      { [ -f "$OUT_DIR/clipdino/results.json" ] && REPORT_ARGS="$REPORT_ARGS --clipdino-json $OUT_DIR/clipdino/results.json"; }

    if [ -n "$REPORT_ARGS" ]; then
      python3 "$SCRIPT_DIR/vision_report.py" $REPORT_ARGS \
        --output-json "$OUT_DIR/report.json" \
        2>&1 | tee "$OUT_DIR/report.txt"
    fi
    ;;
  *)
    echo "Unknown component: $COMPONENT (use vit|clipdino|all)" >&2
    exit 1
    ;;
esac

echo ""
echo "Vision benchmark complete: $OUT_DIR"
