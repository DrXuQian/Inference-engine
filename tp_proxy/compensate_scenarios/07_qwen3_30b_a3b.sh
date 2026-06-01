#!/bin/bash
# Qwen3-30B-A3B GPTQ-Int4, TP=1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/../scripts" && pwd)"
BASE=./results/07_qwen3_30b_a3b
MODEL=${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-30B-A3B-GPTQ-Int4}

if [ ! -d "$MODEL" ]; then
    echo "ERROR: MODEL not found: $MODEL"
    exit 1
fi

for OUTLEN in 200 500; do
    echo "--- Output=${OUTLEN} ---"
    TRACE="$BASE/trace_${OUTLEN}/trace.sqlite"
    if [ ! -f "$TRACE" ]; then
        echo "  SKIP: $TRACE not found (run trace_scenarios/07 first)"
        continue
    fi

    COMM="$BASE/tp1/comm.json"
    [ -f "$COMM" ] && COMM_ARG="--comm-json $COMM" || COMM_ARG=""

    python3 "$SCRIPT_DIR/compensate_ppu.py" \
        --model-dir "$MODEL" \
        --asys-sqlite "$TRACE" \
        --output-len "$OUTLEN" \
        $COMM_ARG \
        --output-json "$BASE/compensated_${OUTLEN}.json"

    # Decode BW utilization (MoE INT4)
    python3 -c "
import json
d = json.load(open('$BASE/compensated_${OUTLEN}.json'))
t = d.get('tail', {})
tpot = t.get('tpot_ms', 0)
ttft_k = t.get('ttft_kernel_ms', 0)
ttft_w = t.get('ttft_wall_ms', 0)

print(f'  TTFT: kernel={ttft_k:.2f}ms  wall={ttft_w:.2f}ms')
print(f'  TPOT: {tpot:.4f}ms')

# BW util: Qwen3-30B-A3B active=2.72B, INT4=0.5B/param
H=2048; qd=32*128; kvd=4*128; moe_ffn=768; top_k=8; layers=48
active = (H*qd+H*kvd+H*kvd+qd*H + top_k*3*H*moe_ffn) * layers
weight_gb = active * 0.5 / 1e9
if tpot > 0:
    bw_floor = weight_gb / 680 * 1000
    print(f'  Decode BW: {active/1e9:.2f}B active × INT4 = {weight_gb:.2f}GB')
    print(f'  BW floor={bw_floor:.2f}ms @ 680GB/s  BW_util={bw_floor/tpot*100:.0f}%')
" 2>/dev/null || true
done

echo "Done: $BASE/"
