#!/bin/bash
# VLA (Vision-Language-Action) full pipeline benchmark.
#
# Three independent components:
#   1. VIT:  PyTorch spatio-temporal vision transformer (24L×1024)
#   2. LLM:  vLLM prefill with Qwen3-VL-4B-Instruct (36L×2560, 1550 tokens)
#   3. DiT:  PyTorch diffusion action transformer (18L×1024, 50 steps)
#
# Usage:
#   bash run_vla_bench.sh [output_dir] [component]
#   bash run_vla_bench.sh ./results/vla
#
# Run single component:
#   bash run_vla_bench.sh ./results/vla vit
#   bash run_vla_bench.sh ./results/vla llm
#   bash run_vla_bench.sh ./results/vla dit

set -euo pipefail

LLM_MODEL="${LLM_MODEL:-/sim/eec/shared/models/Qwen/Qwen3-VL-4B-Instruct}"
OUT_DIR="${1:-./results/vla_bench}"
COMPONENT="${2:-all}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR}_${TIMESTAMP}"

VLA_DIR="$(cd "$(dirname "$0")" && pwd)"
TP_PROXY_DIR="$(cd "$VLA_DIR/../tp_proxy" && pwd)"
PLATFORM="${PLATFORM:-ppu}"

# VLA architecture params
VIT_INPUT_TOKENS=5400     # 6 cam × 225 pos × 4 patches
VIT_OUTPUT_TOKENS=1350    # 6 cam × 225 pos (aggregated)
VIT_HISTORY_TOKENS=11475  # 17 frames × 3 cam × 225 pos
LLM_TOKENS=1550           # 1350 visual + 200 task
DIT_TOKENS=51             # 50 action steps + 1 robot state
DIT_DENOISE_STEPS=50

mkdir -p "$OUT_DIR"

echo "============================================"
echo "  VLA Pipeline Benchmark"
echo "  LLM model: $LLM_MODEL"
echo "  Output: $OUT_DIR"
echo "  Component: $COMPONENT"
echo "  Platform: $PLATFORM"
echo ""
echo "  VIT: ${VIT_INPUT_TOKENS} → ${VIT_OUTPUT_TOKENS} tokens (24L×1024)"
echo "  LLM: ${LLM_TOKENS} tokens prefill (Qwen3-4B, 36L×2560)"
echo "  DiT: ${DIT_TOKENS} tokens × ${DIT_DENOISE_STEPS} steps (18L×1024)"
echo "============================================"


# ============================================================
# 1. VIT — PyTorch benchmark under profiler
# ============================================================
run_vit() {
    echo ""
    echo "=== [1/3] VIT: Spatio-Temporal Vision Transformer ==="
    local VIT_DIR="$OUT_DIR/vit"
    mkdir -p "$VIT_DIR"

    if [ "$PLATFORM" = "ppu" ]; then
        asys profile -o "$VIT_DIR/trace.report" -f true \
            -t hggc,acdnn,acblas,hgtx \
            python3 "$VLA_DIR/vla_bench.py" \
                --component vit --dtype bf16 \
                --warmup 10 --iters 30 \
                --output-json "$VIT_DIR/results.json" \
            2>&1 | tee "$VIT_DIR/bench.log"

        # Export sqlite
        sleep 2
        ASYS_REP=$(find "$VIT_DIR" -name "*.asysrep" -o -name "trace.report" | head -1)
        [ -n "$ASYS_REP" ] && asys export --force-overwrite true \
            -o "$VIT_DIR/trace.sqlite" "$ASYS_REP" 2>/dev/null || true
    else
        nsys profile -t cuda --cuda-graph-trace=node \
            --force-overwrite=true -o "$VIT_DIR/trace" \
            python3 "$VLA_DIR/vla_bench.py" \
                --component vit --dtype bf16 \
                --warmup 10 --iters 30 \
                --output-json "$VIT_DIR/results.json" \
            2>&1 | tee "$VIT_DIR/bench.log"
    fi

    echo "[VIT] Done → $VIT_DIR/results.json"
}


# ============================================================
# 2. LLM — vLLM prefill with Qwen3-4B
# ============================================================
run_llm() {
    echo ""
    echo "=== [2/3] LLM: Qwen3-4B Prefill (vLLM) ==="
    local LLM_DIR="$OUT_DIR/llm"
    mkdir -p "$LLM_DIR"

    # Use capture_trace.sh for LLM prefill
    # max_tokens=1 → pure prefill, no autoregressive decode
    bash "$TP_PROXY_DIR/scripts/capture_trace.sh" \
        "$LLM_MODEL" "$LLM_TOKENS" 1 "$LLM_DIR" 10 1 \
        2>&1 | tee "$LLM_DIR/bench.log"

    echo "[LLM] Done → $LLM_DIR/"
}


# ============================================================
# 3. DiT — PyTorch benchmark under profiler
# ============================================================
run_dit() {
    echo ""
    echo "=== [3/3] DiT: Diffusion Action Transformer ==="
    local DIT_DIR="$OUT_DIR/dit"
    mkdir -p "$DIT_DIR"

    if [ "$PLATFORM" = "ppu" ]; then
        asys profile -o "$DIT_DIR/trace.report" -f true \
            -t hggc,acdnn,acblas,hgtx \
            python3 "$VLA_DIR/vla_bench.py" \
                --component dit --dtype bf16 \
                --warmup 5 --iters 20 \
                --output-json "$DIT_DIR/results.json" \
            2>&1 | tee "$DIT_DIR/bench.log"

        sleep 2
        ASYS_REP=$(find "$DIT_DIR" -name "*.asysrep" -o -name "trace.report" | head -1)
        [ -n "$ASYS_REP" ] && asys export --force-overwrite true \
            -o "$DIT_DIR/trace.sqlite" "$ASYS_REP" 2>/dev/null || true
    else
        nsys profile -t cuda --cuda-graph-trace=node \
            --force-overwrite=true -o "$DIT_DIR/trace" \
            python3 "$VLA_DIR/vla_bench.py" \
                --component dit --dtype bf16 \
                --warmup 5 --iters 20 \
                --output-json "$DIT_DIR/results.json" \
            2>&1 | tee "$DIT_DIR/bench.log"
    fi

    echo "[DiT] Done → $DIT_DIR/results.json"
}


# ============================================================
# Run selected components
# ============================================================
case "$COMPONENT" in
    vit) run_vit ;;
    llm) run_llm ;;
    dit) run_dit ;;
    all)
        run_vit
        run_llm
        run_dit

        # Auto-generate report
        echo ""
        echo "=== [4/3] Pipeline Report ==="
        REPORT_ARGS=""
        VIT_SQLITE="$OUT_DIR/vit/trace.sqlite"
        LLM_SQLITE="$OUT_DIR/llm/trace.sqlite"
        DIT_SQLITE="$OUT_DIR/dit/trace.sqlite"

        [ -f "$VIT_SQLITE" ] && REPORT_ARGS="$REPORT_ARGS --vit-trace $VIT_SQLITE" || \
            { [ -f "$OUT_DIR/vit/results.json" ] && REPORT_ARGS="$REPORT_ARGS --vit-json $OUT_DIR/vit/results.json"; }
        [ -f "$LLM_SQLITE" ] && REPORT_ARGS="$REPORT_ARGS --llm-trace $LLM_SQLITE"
        [ -f "$DIT_SQLITE" ] && REPORT_ARGS="$REPORT_ARGS --dit-trace $DIT_SQLITE" || \
            { [ -f "$OUT_DIR/dit/results.json" ] && REPORT_ARGS="$REPORT_ARGS --dit-json $OUT_DIR/dit/results.json"; }

        if [ -n "$REPORT_ARGS" ]; then
            python3 "$VLA_DIR/vla_report.py" $REPORT_ARGS \
                --show-all-combos \
                --output-json "$OUT_DIR/report.json" \
                2>&1 | tee "$OUT_DIR/report.txt"
        fi

        echo ""
        echo "============================================"
        echo "  VLA Benchmark Complete"
        echo "  Output: $OUT_DIR"
        echo "  Report: $OUT_DIR/report.json"
        echo "============================================"
        ;;
    *)
        echo "Unknown component: $COMPONENT (use vit|llm|dit|all)"
        exit 1
        ;;
esac
