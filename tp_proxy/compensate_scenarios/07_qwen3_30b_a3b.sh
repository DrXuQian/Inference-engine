#!/bin/bash
# Qwen3-30B-A3B BF16, TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/07_qwen3_30b_a3b

for TP in 2; do
    echo "--- TP=$TP ---"
    DIR="$BASE/tp${TP}"

    # Use split/pruned model dir (contains split_meta.json with tp_size, pruned_layers, original_layers)
    MODEL_TP=${MODEL:-$(bash "$SCRIPT_DIR/get_model_path.sh" "$DIR/model" 2>/dev/null || echo "")}
    if [ -z "$MODEL_TP" ]; then
        echo "ERROR: Split model not found at $DIR/model. Run bench_scenarios/07 first"
        continue
    fi
    echo "Using model: $MODEL_TP"

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
            --model-dir "$MODEL_TP" \
            --asys-sqlite "$TRACE" \
            --output-len "$OUTLEN" \
            $COMM_ARG \
            --output-json "$DIR/compensated_${OUTLEN}.json"

        # Compute BW util + INT4 projection from kernel breakdown, save to JSON
        python3 -c "
import sys, json
sys.path.insert(0, '$SCRIPT_DIR')
from model_weight_utils import decode_weight_bytes, print_decode_bw

f = '$DIR/compensated_${OUTLEN}.json'
d = json.load(open(f))
t = d.get('tail', {})
kb = t.get('kernel_breakdown', {})

comp_ttft = comp_tpot = 0
for r in d.get('results', []):
    if 'comp_tpot_ms' in r:
        comp_tpot = r['comp_tpot_ms']
        comp_ttft = r.get('comp_ttft_ms', 0)
        break
if comp_tpot <= 0:
    comp_tpot = t.get('tpot_ms', 0)

# BW utilization (TP=$TP, forced — not from JSON which may be tp=1)
info = decode_weight_bytes('qwen3-30b-a3b', tp=$TP)
print_decode_bw(info, comp_tpot_ms=comp_tpot, peak_bw=680)

# INT4 projection from kernel breakdown (no magic scale)
# gemm_int4 kernels → ×0.25 (BF16 2B → INT4 0.5B weight reduction)
# other kernels unchanged
enc_gemm = kb.get('enc_gemm_ms', 0)
gemm_reduction = enc_gemm * 0.75  # time saved by INT4
# Apply layer_scale to the reduction (gemm is in encoder, gets scaled)
ls = d.get('layer_scale', 1.0)
int4_tpot = comp_tpot - gemm_reduction * ls

output_tokens = d.get('results', [{}])[0].get('output_tokens', $OUTLEN)
int4_tps = 1000 / int4_tpot if int4_tpot > 0 else 0
int4_total = comp_ttft + (output_tokens - 1) * int4_tpot

print(f'  INT4 from kernel breakdown:')
print(f'    enc_gemm={enc_gemm:.3f}ms × 0.75 × layer_scale({ls:.1f}) = {gemm_reduction*ls:.3f}ms saved')
print(f'    BF16 comp_TPOT={comp_tpot:.3f}ms → INT4={int4_tpot:.3f}ms')

# Save to JSON
d['int4'] = {
    'comp_ttft_ms': round(comp_ttft, 3),
    'comp_tpot_ms': round(int4_tpot, 4),
    'tps': round(int4_tps, 1),
    'total_ms': round(int4_total, 2),
    'enc_gemm_ms': round(enc_gemm, 4),
    'gemm_reduction_ms': round(gemm_reduction * ls, 4),
}
d['decode_bw'] = {
    'weight_per_gpu_gb': round(info['total_gb'], 3),
    'bw_floor_ms': round(info['total_gb'] / 680 * 1000, 3),
    'tp': $TP,
}
json.dump(d, open(f, 'w'), indent=2)
print(f'  Saved to {f}')
" || echo "  WARNING: INT4 projection failed"
    done
    echo ""
done

echo "Done: $BASE/"
