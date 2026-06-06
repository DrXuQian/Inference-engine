#!/bin/bash
# Test vLLM custom all-reduce: verify it's used and check accuracy.
#
# Usage:
#   bash test_custom_ar.sh [model] [tp]
#   bash test_custom_ar.sh Qwen/Qwen3-0.6B 2
#
# What it does:
#   1) WITH custom AR  → run vLLM, capture outputs
#   2) WITHOUT custom AR (NCCL only) → baseline outputs
#   3) Compare for accuracy
#   4) Optionally profile with nsys/asys to extract comm kernel names
#
# The call-counting hook lives inside the worker process (via a temp
# sitecustomize.py on PYTHONPATH) so it survives vLLM v1 multiprocess.
set -uo pipefail   # no -e: we want both runs even if one fails

MODEL="${1:-Qwen/Qwen3-0.6B}"
TP="${2:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_TOKENS="${MAX_TOKENS:-32}"
OUTDIR="${OUTDIR:-/tmp/test_custom_ar}"
PROFILE="${PROFILE:-0}"       # set PROFILE=1 to also capture nsys/asys trace

mkdir -p "$OUTDIR"

echo "============================================"
echo " Test: custom AR vs NCCL on ${TP}-GPU PCIe"
echo " Model: $MODEL"
echo "============================================"

# ── Create a sitecustomize.py that hooks into worker processes ───────
HOOK_DIR="$OUTDIR/_hook"
mkdir -p "$HOOK_DIR"
cat > "$HOOK_DIR/sitecustomize.py" << 'HOOKEOF'
"""Auto-installed hook: count custom AR calls inside vLLM worker processes."""
import atexit, os, json

_LOG_FILE = os.environ.get("_CUSTOM_AR_LOG", "")
if not _LOG_FILE:
    pass  # not our run, skip
else:
    _counts = {"custom_ar": 0, "nccl_fallback": 0, "custom_ar_bytes": 0}
    _hooked = False

    def _try_hook():
        global _hooked
        if _hooked:
            return
        try:
            from vllm.distributed.device_communicators import custom_all_reduce as car
            _orig = car.CustomAllreduce.custom_all_reduce

            def _counted(self, inp):
                out = _orig(self, inp)
                if out is not None:
                    _counts["custom_ar"] += 1
                    _counts["custom_ar_bytes"] += inp.numel() * inp.element_size()
                else:
                    _counts["nccl_fallback"] += 1
                return out

            car.CustomAllreduce.custom_all_reduce = _counted
            _hooked = True
        except Exception:
            pass

    def _dump():
        if _counts["custom_ar"] + _counts["nccl_fallback"] > 0:
            pid = os.getpid()
            path = f"{_LOG_FILE}.{pid}"
            with open(path, "w") as f:
                json.dump({"pid": pid, **_counts}, f)

    atexit.register(_dump)

    # Hook as early as possible; also hook lazily via import hook
    _try_hook()

    import importlib
    _orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def _lazy_import(*args, **kwargs):
        mod = _orig_import(*args, **kwargs)
        _try_hook()
        return mod

    try:
        __builtins__.__import__ = _lazy_import
    except (AttributeError, TypeError):
        import builtins
        builtins.__import__ = _lazy_import
HOOKEOF

# ── Run vLLM inference ──────────────────────────────────────────────
run_vllm() {
    local label="$1"
    local disable_ar="$2"       # True or False
    local log_file="$OUTDIR/${label}"
    local ar_log="$OUTDIR/${label}_ar_counts"

    # Clean up old count files
    rm -f "${ar_log}".* 2>/dev/null

    echo "[${label}] Starting vLLM (TP=$TP, disable_custom_all_reduce=$disable_ar) ..."

    _CUSTOM_AR_LOG="$ar_log" \
    PYTHONPATH="$HOOK_DIR:${PYTHONPATH:-}" \
    python3 -c "
import time
from vllm import LLM, SamplingParams

prompts = [
    'The capital of France is',
    '1+1=',
    'Explain GPU all-reduce in one sentence:',
    'Write a haiku about parallel computing:',
]
sp = SamplingParams(max_tokens=${MAX_TOKENS}, temperature=0, seed=42)

llm = LLM(
    model='${MODEL}',
    tensor_parallel_size=${TP},
    max_model_len=${MAX_MODEL_LEN},
    enforce_eager=True,
    disable_custom_all_reduce=${disable_ar},
)

t0 = time.time()
outs = llm.generate(prompts, sp)
elapsed = time.time() - t0

for i, o in enumerate(outs):
    print(f'[OUTPUT-{i}] {o.outputs[0].text}')
print(f'[TIME] {elapsed:.2f}s')
" 2>&1 | tee "${log_file}.txt"

    # Aggregate AR counts from all worker processes
    echo ""
    echo "[${label}] Custom AR call counts (per worker):"
    local total_ar=0 total_nccl=0 total_bytes=0
    for f in "${ar_log}".*; do
        [ -f "$f" ] || continue
        echo "  $(cat "$f")"
        ar=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('custom_ar',0))")
        nccl=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('nccl_fallback',0))")
        bytes=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('custom_ar_bytes',0))")
        total_ar=$((total_ar + ar))
        total_nccl=$((total_nccl + nccl))
        total_bytes=$((total_bytes + bytes))
    done

    if [ "$total_ar" -gt 0 ] 2>/dev/null; then
        echo "  [RESULT] ✓ Custom AR ACTIVE: ${total_ar} calls, $((total_bytes / 1024)) KB"
    elif [ "$total_nccl" -gt 0 ] 2>/dev/null; then
        echo "  [RESULT] All ${total_nccl} calls went to NCCL (custom AR not used)"
    else
        echo "  [RESULT] No AR call counts recorded (hook may not have triggered)"
    fi
    echo ""
}

# ── Run 1: Custom AR ────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [Run 1] Custom AR enabled"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
VLLM_FORCE_CUSTOM_AR_PCIE=$TP \
VLLM_CUSTOM_AR_MAX_BYTES=524288 \
VLLM_SKIP_P2P_CHECK=1 \
VLLM_LOGGING_LEVEL=WARNING \
    run_vllm "custom_ar" "False"

# ── Run 2: NCCL only ────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [Run 2] NCCL only (disable_custom_all_reduce=True)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
VLLM_LOGGING_LEVEL=WARNING \
    run_vllm "nccl_only" "True"

# ── Compare outputs ─────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Accuracy comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ON=$(grep '^\[OUTPUT-' "$OUTDIR/custom_ar.txt" 2>/dev/null | sort)
OFF=$(grep '^\[OUTPUT-' "$OUTDIR/nccl_only.txt" 2>/dev/null | sort)

if [ -z "$ON" ] || [ -z "$OFF" ]; then
    echo "⚠ Cannot compare: one or both runs produced no output"
elif [ "$ON" = "$OFF" ]; then
    echo "✓ PASS: Outputs IDENTICAL between custom AR and NCCL"
else
    echo "⚠ DIFF: Outputs differ (may be acceptable — floating point reduction order)"
    diff <(echo "$ON") <(echo "$OFF") || true
fi

# ── Optional: nsys/asys trace for kernel names ──────────────────────
if [ "$PROFILE" = "1" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Profiling: extracting comm kernel names"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    NSYS_CMD="nsys"
    command -v asys &>/dev/null && NSYS_CMD="asys"
    TRACE_FILE="$OUTDIR/profile_tp${TP}"

    VLLM_FORCE_CUSTOM_AR_PCIE=$TP \
    VLLM_CUSTOM_AR_MAX_BYTES=524288 \
    VLLM_SKIP_P2P_CHECK=1 \
    VLLM_LOGGING_LEVEL=WARNING \
    $NSYS_CMD profile --force-overwrite true \
        -o "$TRACE_FILE" -t cuda --duration 120 \
        python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='${MODEL}', tensor_parallel_size=${TP}, max_model_len=${MAX_MODEL_LEN}, enforce_eager=True)
llm.generate(['Hello'], SamplingParams(max_tokens=8, temperature=0))
" 2>&1

    # Export + extract
    SQLITE="${TRACE_FILE}.sqlite"
    [ -f "$SQLITE" ] || $NSYS_CMD export --type sqlite -o "$SQLITE" "${TRACE_FILE}.nsys-rep" 2>/dev/null

    if [ -f "$SQLITE" ]; then
        python3 << PYEOF
import sqlite3
conn = sqlite3.connect("$SQLITE")
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
kt = 'CUPTI_ACTIVITY_KIND_KERNEL' if 'CUPTI_ACTIVITY_KIND_KERNEL' in tables else None
if not kt:
    for t in tables:
        if 'KERNEL' in t.upper():
            kt = t; break
if kt:
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({kt})").fetchall()]
    ncol = 'demangledName' if 'demangledName' in cols else 'shortName' if 'shortName' in cols else cols[0]
    rows = conn.execute(f"SELECT {ncol}, COUNT(*), SUM(end-start)/1e6 FROM {kt} GROUP BY {ncol}").fetchall()
    # Filter comm-related
    comm_kws = ['cross_device_reduce', 'nccl', 'allreduce', 'broadcast', 'all_reduce']
    print("Comm kernels:")
    for name, cnt, ms in sorted(rows, key=lambda r: -r[2]):
        n = str(name)
        if any(k in n.lower() for k in comm_kws):
            print(f"  {cnt:5d}x {ms:8.2f}ms  {n[:100]}")
    print("\nTop-10 compute kernels:")
    for name, cnt, ms in sorted(rows, key=lambda r: -r[2])[:10]:
        n = str(name)
        if not any(k in n.lower() for k in comm_kws):
            print(f"  {cnt:5d}x {ms:8.2f}ms  {n[:100]}")
conn.close()
PYEOF
    else
        echo "  (no sqlite trace generated)"
    fi
fi

echo ""
echo "Done. Results in $OUTDIR/"
