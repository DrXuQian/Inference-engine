#!/bin/bash
# Benchmark all-reduce and all-gather using pccl_tools / nccl-tests.
# Reads hidden_size, vocab_size, num_hidden_layers from model config.
# Saves results to comm.json.
#
# Usage:
#   bash comm_bench.sh /path/to/model 2 ./results/comm.json
#   bash comm_bench.sh /path/to/model 4 ./results/comm.json

set -euo pipefail

MODEL_DIR="${1:?Usage: $0 <model_dir> <tp_size> <input_len> <output_json> [--trace]}"
TP_SIZE="${2:?}"
INPUT_LEN="${3:?}"
OUTPUT_JSON="${4:?}"
USE_TRACE="${5:-}"

# Tool paths (override via env)
AR=${PCCL_AR:-/usr/local/PPU_SDK/pccl_tools/all_reduce_perf}
AG=${PCCL_AG:-/usr/local/PPU_SDK/pccl_tools/all_gather_perf}
PLATFORM="${PLATFORM:-ppu}"

# Read config
HIDDEN=$(python3 -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('text_config', cfg)
print(tc['hidden_size'])
")
VOCAB=$(python3 -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('text_config', cfg)
print(tc['vocab_size'])
")
NUM_LAYERS=$(python3 -c "
import json
with open('${MODEL_DIR}/config.json') as f:
    cfg = json.load(f)
tc = cfg.get('text_config', cfg)
print(tc['num_hidden_layers'])
")

echo "Model: hidden=$HIDDEN, vocab=$VOCAB, layers=$NUM_LAYERS, TP=$TP_SIZE"

# Sizes
# Decode AR: batch=1 × hidden × bf16
AR_DECODE_SIZE=$((HIDDEN * 2))
# Prefill AR: input_len × hidden × bf16
AR_PREFILL_SIZE=$((INPUT_LEN * HIDDEN * 2))
# AG: lm_head = (vocab / tp) × bf16
AG_SIZE=$(python3 -c "print(($VOCAB // $TP_SIZE) * 2)")

TRACE_DIR="$(dirname "$OUTPUT_JSON")/comm_trace"

# Profiler prefix for --trace mode
if [ "$USE_TRACE" = "--trace" ]; then
    mkdir -p "$TRACE_DIR"
    if [ "$PLATFORM" = "ppu" ]; then
        PROF_PREFIX="asys profile -f true -t hggc,acdnn,acblas,hgtx"
    else
        PROF_PREFIX="nsys profile -t cuda --cuda-graph-trace=node --force-overwrite=true"
    fi
    echo "Trace mode: profiling AR/AG kernels"
else
    PROF_PREFIX=""
fi

echo ""
AR_DEC_CMD="$AR -b $AR_DECODE_SIZE -e $AR_DECODE_SIZE -f 2 -d bf16 -n 50 -w 10 -g $TP_SIZE -c 1 -G 100"
echo "=== All-Reduce DECODE (${AR_DECODE_SIZE} bytes) ==="
echo "CMD: $AR_DEC_CMD"
if [ -n "$PROF_PREFIX" ]; then
    $PROF_PREFIX -o "$TRACE_DIR/ar_decode" $AR_DEC_CMD 2>&1 | tee "$TRACE_DIR/ar_decode.log"
    AR_DEC_OUTPUT=$(cat "$TRACE_DIR/ar_decode.log")
else
    AR_DEC_OUTPUT=$($AR_DEC_CMD 2>&1)
fi
echo "$AR_DEC_OUTPUT" | grep -v "^#" | grep -v "^$" | head -3
AR_DECODE_US=$(echo "$AR_DEC_OUTPUT" | grep -v "^#" | grep -v "^$" | head -1 | awk '{print $11}')

echo ""
AR_PRE_CMD="$AR -b $AR_PREFILL_SIZE -e $AR_PREFILL_SIZE -f 2 -d bf16 -n 50 -w 10 -g $TP_SIZE -c 1 -G 100"
echo "=== All-Reduce PREFILL (${AR_PREFILL_SIZE} bytes, input_len=$INPUT_LEN) ==="
echo "CMD: $AR_PRE_CMD"
if [ -n "$PROF_PREFIX" ]; then
    $PROF_PREFIX -o "$TRACE_DIR/ar_prefill" $AR_PRE_CMD 2>&1 | tee "$TRACE_DIR/ar_prefill.log"
    AR_PRE_OUTPUT=$(cat "$TRACE_DIR/ar_prefill.log")
else
    AR_PRE_OUTPUT=$($AR_PRE_CMD 2>&1)
fi
echo "$AR_PRE_OUTPUT" | grep -v "^#" | grep -v "^$" | head -3
AR_PREFILL_US=$(echo "$AR_PRE_OUTPUT" | grep -v "^#" | grep -v "^$" | head -1 | awk '{print $11}')

echo ""
AG_CMD="$AG -b $AG_SIZE -e $AG_SIZE -f 2 -d bf16 -n 50 -w 10 -g $TP_SIZE -c 1 -G 100"
echo "=== All-Gather (lm_head, ${AG_SIZE} bytes) ==="
echo "CMD: $AG_CMD"
if [ -n "$PROF_PREFIX" ]; then
    $PROF_PREFIX -o "$TRACE_DIR/ag" $AG_CMD 2>&1 | tee "$TRACE_DIR/ag.log"
    AG_OUTPUT=$(cat "$TRACE_DIR/ag.log")
else
    AG_OUTPUT=$($AG_CMD 2>&1)
fi
echo "$AG_OUTPUT" | grep -v "^#" | grep -v "^$" | head -3
AG_US=$(echo "$AG_OUTPUT" | grep -v "^#" | grep -v "^$" | head -1 | awk '{print $11}')

# Compute per decode step
N_AR=$((NUM_LAYERS * 2))
N_AG=2

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
    'method': 'pccl_standalone',
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

# If --trace, extract kernel median from sqlite and update comm.json
if [ "$USE_TRACE" = "--trace" ]; then
    echo ""
    echo "=== Extracting kernel times from traces ==="
    python3 -c "
import sqlite3, json, os, glob

def get_kernel_median(trace_dir, prefix):
    # Find sqlite file
    patterns = [f'{trace_dir}/{prefix}*.sqlite', f'{trace_dir}/{prefix}*.nsys-rep']
    sqlite_file = None
    for pat in patterns:
        files = glob.glob(pat)
        if files:
            # If nsys-rep, need to export
            if files[0].endswith('.nsys-rep'):
                import subprocess
                sqlite_file = files[0].replace('.nsys-rep', '.sqlite')
                subprocess.run(['nsys', 'stats', '-r', 'cuda_gpu_kern_sum', '--format', 'csv',
                               '--force-export=true', files[0]], capture_output=True)
            else:
                sqlite_file = files[0]
            break
    # Try asys export
    if not sqlite_file:
        for ext in ['.asysrep', '.report']:
            reps = glob.glob(f'{trace_dir}/{prefix}*{ext}')
            if reps:
                sqlite_file = f'{trace_dir}/{prefix}.sqlite'
                os.system(f'asys export --force-overwrite true -o {sqlite_file} {reps[0]} 2>/dev/null')
                break
    if not sqlite_file or not os.path.exists(sqlite_file):
        return None

    conn = sqlite3.connect(sqlite_file)
    c = conn.cursor()
    c.execute('SELECT name FROM sqlite_master WHERE type=\"table\"')
    kt = [r[0] for r in c.fetchall() if 'KERNEL' in r[0].upper() and 'ACTIVITY' in r[0].upper()]
    if not kt:
        conn.close(); return None
    c.execute(f'SELECT (k.\"end\" - k.start) / 1000.0 FROM \"{kt[0]}\" k ORDER BY k.start')
    durs = [r[0] for r in c.fetchall()]  # us
    conn.close()
    if not durs:
        return None
    durs.sort()
    return durs[len(durs)//2]  # median us

td = '${TRACE_DIR}'
ar_dec_kernel = get_kernel_median(td, 'ar_decode')
ar_pre_kernel = get_kernel_median(td, 'ar_prefill')
ag_kernel = get_kernel_median(td, 'ag')

print(f'  AR decode kernel median: {ar_dec_kernel:.1f} us' if ar_dec_kernel else '  AR decode: N/A')
print(f'  AR prefill kernel median: {ar_pre_kernel:.1f} us' if ar_pre_kernel else '  AR prefill: N/A')
print(f'  AG kernel median: {ag_kernel:.1f} us' if ag_kernel else '  AG: N/A')

# Update comm.json with kernel times
if ar_dec_kernel or ar_pre_kernel:
    f = '${OUTPUT_JSON}'
    d = json.load(open(f))
    if ar_dec_kernel:
        d['decode_ar_kernel_us'] = round(ar_dec_kernel, 1)
        d['decode_total_kernel_ms'] = round((ar_dec_kernel * d['n_ar_per_step'] + (ag_kernel or 0) * d['n_ag_per_step']) / 1000, 3)
    if ar_pre_kernel:
        d['prefill_ar_kernel_us'] = round(ar_pre_kernel, 1)
        d['prefill_total_kernel_ms'] = round((ar_pre_kernel * d['n_ar_per_step'] + (ag_kernel or 0) * d['n_ag_per_step']) / 1000, 3)
    if ag_kernel:
        d['ag_kernel_us'] = round(ag_kernel, 1)
    d['method'] = d.get('method', 'pccl') + '+trace'
    json.dump(d, open(f, 'w'), indent=2)
    print(f'  Updated {f} with kernel times')
" 2>/dev/null || echo "  WARNING: trace extraction failed"
fi
