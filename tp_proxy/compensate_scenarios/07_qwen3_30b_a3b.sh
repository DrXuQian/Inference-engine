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

# BW analysis: measured in BF16, then scale to INT4 for MoE+attn weights
H=2048; qd=32*128; kvd=4*128; moe_ffn=768; shared_ffn=6144
top_k=8; layers=48; tp=$TP; vocab=151936

attn_params = (H*qd + H*kvd + H*kvd + qd*H) * layers / tp
shared_params = (3 * H * shared_ffn) * layers / tp
moe_params = (top_k * 3 * H * moe_ffn) * layers / tp
lm_params = vocab * H / tp
router_params = H * 128 * layers

# BF16 (measured)
bf16_gb = (attn_params + shared_params + moe_params + lm_params + router_params) * 2 / 1e9
# INT4 projection: MoE+attn+shared → INT4(0.5B), lm_head+router → BF16(2B)
int4_layer_gb = (attn_params + shared_params + moe_params) * 0.5 / 1e9
int4_other_gb = (lm_params + router_params) * 2 / 1e9
int4_gb = int4_layer_gb + int4_other_gb

comp_tpot = 0
for r in d.get('results', []):
    if 'comp_tpot_ms' in r:
        comp_tpot = r['comp_tpot_ms']; break
if comp_tpot <= 0:
    comp_tpot = tpot

if comp_tpot > 0:
    # Scale comp_TPOT to INT4: weight-bound kernels scale by BF16→INT4 ratio
    scale_ratio = int4_gb / bf16_gb
    int4_tpot = comp_tpot * scale_ratio
    bf16_floor = bf16_gb / 680 * 1000
    int4_floor = int4_gb / 680 * 1000
    print(f'  Weight/GPU: BF16={bf16_gb:.3f}GB → INT4={int4_gb:.3f}GB (scale={scale_ratio:.2f}x)')
    print(f'    layers(INT4): {int4_layer_gb:.3f}GB  lm_head+router(BF16): {int4_other_gb:.3f}GB')
    print(f'  BF16: comp_TPOT={comp_tpot:.2f}ms  BW_floor={bf16_floor:.2f}ms  BW_util={bf16_floor/comp_tpot*100:.0f}%')
    print(f'  INT4: est_TPOT={int4_tpot:.2f}ms  BW_floor={int4_floor:.2f}ms  BW_util={int4_floor/int4_tpot*100:.0f}%')
" 2>/dev/null || true
    done
    echo ""
done

echo "Done: $BASE/"
