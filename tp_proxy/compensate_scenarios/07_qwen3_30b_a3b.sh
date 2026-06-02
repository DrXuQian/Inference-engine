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

        # Compute INT4 projection and save to JSON
        python3 -c "
import sys, json
sys.path.insert(0, '$SCRIPT_DIR')
from model_weight_utils import decode_weight_bytes, print_decode_bw

f = '$DIR/compensated_${OUTLEN}.json'
d = json.load(open(f))
t = d.get('tail', {})
ls = d.get('layer_scale', 1.0)
tp = d.get('tp_size', $TP)

comp_ttft = comp_tpot = 0
for r in d.get('results', []):
    if 'comp_tpot_ms' in r:
        comp_tpot = r['comp_tpot_ms']
        comp_ttft = r.get('comp_ttft_ms', 0)
        break
if comp_tpot <= 0:
    comp_tpot = t.get('tpot_ms', 0)

info = decode_weight_bytes('qwen3-30b-a3b', tp=tp)
print_decode_bw(info, comp_tpot_ms=comp_tpot, peak_bw=680)

# INT4 projection: TPOT × int4_scale
int4_tpot = comp_tpot * info['int4_scale']
int4_tps = 1000 / int4_tpot if int4_tpot > 0 else 0
output_tokens = d.get('results', [{}])[0].get('output_tokens', $OUTLEN)
int4_total = comp_ttft + (output_tokens - 1) * int4_tpot

# Save INT4 results back to JSON
d['int4'] = {
    'comp_ttft_ms': round(comp_ttft, 3),
    'comp_tpot_ms': round(int4_tpot, 4),
    'tps': round(int4_tps, 1),
    'total_ms': round(int4_total, 2),
    'int4_scale': round(info['int4_scale'], 4),
    'bf16_gb': round(info['bf16_gb'], 3),
    'int4_gb': round(info['int4_gb'], 3),
}
d['decode_bw'] = {
    'weight_gb': round(info['total_gb'], 3),
    'bf16_gb': round(info['bf16_gb'], 3),
    'int4_gb': round(info['int4_gb'], 3),
    'bw_floor_ms': round(info['total_gb'] / 680 * 1000, 3),
}
json.dump(d, open(f, 'w'), indent=2)
print(f'  INT4: comp_TPOT={int4_tpot:.2f}ms  TPS={int4_tps:.1f}  scale={info[\"int4_scale\"]:.3f}')
print(f'  Saved INT4 projection to {f}')
" 2>/dev/null || true
    done
    echo ""
done

echo "Done: $BASE/"
