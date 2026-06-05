#!/bin/bash
# Benchmark all-reduce and all-gather using nccl-tests.
# Reads model config to determine message sizes, outputs comm.json.
#
# Usage:
#   bash comm_bench_nccl.sh /path/to/model 2 102400 ./results/comm.json
#   bash comm_bench_nccl.sh /path/to/model 4 13400 ./results/comm.json
#   AR_TOOL=/path/to/all_reduce_perf AG_TOOL=/path/to/all_gather_perf bash ...
#
# For PPU (pccl_tools):
#   AR_TOOL=/usr/local/PPU_SDK/pccl_tools/all_reduce_perf \
#   AG_TOOL=/usr/local/PPU_SDK/pccl_tools/all_gather_perf \
#   bash comm_bench_nccl.sh /path/to/model 2 102400 ./results/comm.json

set -euo pipefail

MODEL_DIR="${1:?Usage: $0 <model_dir> <tp_size> <input_len> <output_json>}"
TP_SIZE="${2:?}"
INPUT_LEN="${3:?}"
OUTPUT_JSON="${4:?}"

# Tool paths (override via env)
# Default: nccl-tests standard install paths
AR_TOOL=${AR_TOOL:-/usr/local/bin/all_reduce_perf}
AG_TOOL=${AG_TOOL:-/usr/local/bin/all_gather_perf}

# Fallback: try common nccl-tests locations
if [ ! -f "$AR_TOOL" ]; then
    for p in /usr/local/nccl-tests/build/all_reduce_perf \
             /opt/nccl-tests/build/all_reduce_perf \
             $(which all_reduce_perf 2>/dev/null); do
        if [ -f "$p" ]; then AR_TOOL="$p"; break; fi
    done
fi
if [ ! -f "$AG_TOOL" ]; then
    for p in /usr/local/nccl-tests/build/all_gather_perf \
             /opt/nccl-tests/build/all_gather_perf \
             $(which all_gather_perf 2>/dev/null); do
        if [ -f "$p" ]; then AG_TOOL="$p"; break; fi
    done
fi

echo "AR tool: $AR_TOOL"
echo "AG tool: $AG_TOOL"

# Read config
read HIDDEN VOCAB NUM_LAYERS <<< $(python3 -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('text_config', cfg)
print(tc['hidden_size'], tc['vocab_size'], tc['num_hidden_layers'])
")

echo "Model: hidden=$HIDDEN, vocab=$VOCAB, layers=$NUM_LAYERS, TP=$TP_SIZE"

# Message sizes
# Decode AR: batch=1 × hidden × bf16
AR_DECODE_SIZE=$((HIDDEN * 2))
# Prefill AR: input_len × hidden × bf16
AR_PREFILL_SIZE=$((INPUT_LEN * HIDDEN * 2))
# AG: lm_head = (vocab / tp) × bf16
AG_SIZE=$(python3 -c "print(($VOCAB // $TP_SIZE) * 2)")

echo ""
echo "=== All-Reduce DECODE (${AR_DECODE_SIZE} bytes) ==="
AR_DEC_OUTPUT=$($AR_TOOL -b $AR_DECODE_SIZE -e $AR_DECODE_SIZE -f 2 \
    -n 500 -w 100 -g $TP_SIZE 2>&1)
echo "$AR_DEC_OUTPUT" | grep -v "^#" | grep -v "^$" | head -3
AR_DECODE_US=$(echo "$AR_DEC_OUTPUT" | grep -v "^#" | grep -v "^$" | head -1 | awk '{print $6}')

echo ""
echo "=== All-Reduce PREFILL (${AR_PREFILL_SIZE} bytes, input_len=$INPUT_LEN) ==="
AR_PRE_OUTPUT=$($AR_TOOL -b $AR_PREFILL_SIZE -e $AR_PREFILL_SIZE -f 2 \
    -n 100 -w 20 -g $TP_SIZE 2>&1)
echo "$AR_PRE_OUTPUT" | grep -v "^#" | grep -v "^$" | head -3
AR_PREFILL_US=$(echo "$AR_PRE_OUTPUT" | grep -v "^#" | grep -v "^$" | head -1 | awk '{print $6}')

echo ""
echo "=== All-Gather (lm_head, ${AG_SIZE} bytes) ==="
AG_OUTPUT=$($AG_TOOL -b $AG_SIZE -e $AG_SIZE -f 2 \
    -n 300 -w 50 -g $TP_SIZE 2>&1)
echo "$AG_OUTPUT" | grep -v "^#" | grep -v "^$" | head -3
AG_US=$(echo "$AG_OUTPUT" | grep -v "^#" | grep -v "^$" | head -1 | awk '{print $6}')

# Compute per decode step
N_AR=$((NUM_LAYERS * 2))
N_AG=2

mkdir -p "$(dirname "$OUTPUT_JSON")"

python3 -c "
import json
ar_decode_us = float('${AR_DECODE_US}') if '${AR_DECODE_US}' else 0
ar_prefill_us = float('${AR_PREFILL_US}') if '${AR_PREFILL_US}' else 0
ag_us = float('${AG_US}') if '${AG_US}' else 0
n_ar = $N_AR
n_ag = $N_AG
decode_total = (n_ar * ar_decode_us + n_ag * ag_us) / 1000
prefill_total = (n_ar * ar_prefill_us + n_ag * ag_us) / 1000

print()
print(f'DECODE:')
print(f'  AR: {ar_decode_us:.1f} us/call × {n_ar} = {n_ar*ar_decode_us/1000:.3f} ms')
print(f'  AG: {ag_us:.1f} us/call × {n_ag} = {n_ag*ag_us/1000:.3f} ms')
print(f'  Total/step: {decode_total:.3f} ms')
print(f'PREFILL:')
print(f'  AR: {ar_prefill_us:.1f} us/call × {n_ar} = {n_ar*ar_prefill_us/1000:.3f} ms')
print(f'  AG: {ag_us:.1f} us/call × {n_ag} = {n_ag*ag_us/1000:.3f} ms')
print(f'  Total/step: {prefill_total:.3f} ms')

result = {
    'method': 'nccl_standalone',
    'decode_ar_us': round(ar_decode_us, 1),
    'prefill_ar_us': round(ar_prefill_us, 1),
    'ag_us': round(ag_us, 1),
    'decode_ar_size_bytes': $AR_DECODE_SIZE,
    'prefill_ar_size_bytes': $AR_PREFILL_SIZE,
    'input_len': $INPUT_LEN,
    'ag_size_bytes': $AG_SIZE,
    'n_ar_per_step': n_ar,
    'n_ag_per_step': n_ag,
    'decode_total_per_step_ms': round(decode_total, 3),
    'prefill_total_per_step_ms': round(prefill_total, 3),
    'total_per_step_ms': round(decode_total, 3),
    'tp_size': $TP_SIZE,
    'hidden_size': $HIDDEN,
    'vocab_size': $VOCAB,
    'num_layers': $NUM_LAYERS,
}
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved to ${OUTPUT_JSON}')
"
