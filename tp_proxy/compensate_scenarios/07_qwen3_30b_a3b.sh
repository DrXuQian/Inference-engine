#!/bin/bash
# Qwen3-30B-A3B BF16, TP=1 and TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/07_qwen3_30b_a3b
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B}

if [ ! -d "$MODEL" ]; then
    echo "ERROR: MODEL not found: $MODEL"
    exit 1
fi

for TP in 2; do
    echo "--- TP=$TP ---"
    DIR="$BASE/tp${TP}"

    COMM="$DIR/comm.json"
    [ -f "$COMM" ] && COMM_ARG="--comm-json $COMM" || COMM_ARG=""

    for OUTLEN in 200 500; do
        echo "--- TP=$TP, Output=${OUTLEN} ---"
        TRACE="$DIR/trace_${OUTLEN}/trace.sqlite"
        if [ ! -f "$TRACE" ]; then
            echo "  SKIP: $TRACE not found (run trace_scenarios/07 first)"
            continue
        fi

        python3 "$SCRIPT_DIR/compensate_ppu.py" \
            --model-dir "$MODEL" \
            --asys-sqlite "$TRACE" \
            --output-len "$OUTLEN" \
            $COMM_ARG \
            --output-json "$DIR/compensated_${OUTLEN}.json"

        # Decode BW utilization via model_weight_utils
        python3 -c "
import sys, json
sys.path.insert(0, '$SCRIPT_DIR')
from model_weight_utils import decode_weight_bytes, print_decode_bw

d = json.load(open('$DIR/compensated_${OUTLEN}.json'))
t = d.get('tail', {})
print(f'  TTFT: kernel={t.get(\"ttft_kernel_ms\",0):.2f}ms  wall={t.get(\"ttft_wall_ms\",0):.2f}ms')
print(f'  TPOT: {t.get(\"tpot_ms\",0):.4f}ms')

comp_tpot = 0
for r in d.get('results', []):
    if 'comp_tpot_ms' in r:
        comp_tpot = r['comp_tpot_ms']; break
if comp_tpot <= 0:
    comp_tpot = t.get('tpot_ms', 0)

info = decode_weight_bytes('qwen3-30b-a3b', tp=$TP)
print_decode_bw(info, comp_tpot_ms=comp_tpot, peak_bw=680)
" 2>/dev/null || true
    done
    echo ""
done

echo "Done: $BASE/"
