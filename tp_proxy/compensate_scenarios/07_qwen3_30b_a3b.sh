#!/bin/bash
# Qwen3-30B-A3B GPTQ-Int4, TP=1 and TP=2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/07_qwen3_30b_a3b
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B-GPTQ-Int4}

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

        # Decode BW utilization (MoE INT4)
        python3 -c "
import json
d = json.load(open('$DIR/compensated_${OUTLEN}.json'))
t = d.get('tail', {})
tpot = t.get('tpot_ms', 0)
ttft_k = t.get('ttft_kernel_ms', 0)
ttft_w = t.get('ttft_wall_ms', 0)

print(f'  TTFT: kernel={ttft_k:.2f}ms  wall={ttft_w:.2f}ms')
print(f'  TPOT: {tpot:.4f}ms')

# BW util: use compensated TPOT (lm_head already /tp in compensate)
# Weight per GPU: attn+shared+MoE (INT4 split) + lm_head (BF16 split) + router
H=2048; qd=32*128; kvd=4*128; moe_ffn=768; shared_ffn=6144
top_k=8; layers=48; tp=$TP; vocab=151936
# Transformer layers (INT4, TP-split)
attn = H*qd + H*kvd + H*kvd + qd*H
shared = 3 * H * shared_ffn
moe = top_k * 3 * H * moe_ffn
layer_bytes = (attn + shared + moe) * layers * 0.5 / tp  # INT4
# lm_head (BF16, TP-split)
lm_bytes = vocab * H * 2 / tp
# Router (BF16, replicated, small)
router_bytes = H * 128 * layers * 2
weight_gb = (layer_bytes + lm_bytes + router_bytes) / 1e9

# Use comp_TPOT (compensated, lm_head already /tp)
comp_tpot = 0
for r in d.get('results', []):
    if 'comp_tpot_ms' in r:
        comp_tpot = r['comp_tpot_ms']; break
if comp_tpot <= 0:
    comp_tpot = tpot

if comp_tpot > 0:
    bw_floor = weight_gb / 680 * 1000
    print(f'  Decode BW: {weight_gb:.3f}GB/GPU (TP={tp})')
    print(f'    layers(INT4): {layer_bytes/1e9:.3f}GB  lm_head(BF16): {lm_bytes/1e9:.3f}GB')
    print(f'  BW floor={bw_floor:.2f}ms  comp_TPOT={comp_tpot:.2f}ms  BW_util={bw_floor/comp_tpot*100:.0f}%')
" 2>/dev/null || true
    done
    echo ""
done

echo "Done: $BASE/"
