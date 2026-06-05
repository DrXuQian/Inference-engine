#!/bin/bash
# Benchmark Qwen3-VL-30B-A3B: 1.5K input, 200-500 output
#
# Usage:
#   bash bench_qwen3vl_30b.sh /path/to/Qwen3-VL-30B-A3B [output_dir]
#   MODEL=/path/to/model bash bench_qwen3vl_30b.sh
#
# Captures:
#   1. Prefill trace (max_tokens=1, pure TTFT)
#   2. Decode trace (max_tokens=500, TTFT + TPOT)
#   3. Report with kernel breakdown

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-${MODEL:-/sim/eec/shared/models/Qwen/Qwen3-VL-30B-A3B}}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${2:-./results/qwen3vl_30b_${TIMESTAMP}}"
PLATFORM="${PLATFORM:-ppu}"

INPUT_LEN=1536
OUTPUT_LENS="200 500"
NUM_PROMPTS=5
BATCH_SIZE=1

if [ ! -d "$MODEL" ]; then
    echo "ERROR: Model not found: $MODEL"
    echo "Usage: bash $0 /path/to/Qwen3-VL-30B-A3B [output_dir]"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "============================================"
echo "  Qwen3-VL-30B-A3B Benchmark"
echo "  Model: $MODEL"
echo "  Input: ${INPUT_LEN} tokens"
echo "  Output: ${OUTPUT_LENS} tokens"
echo "  Output: $OUT_DIR"
echo "  Platform: $PLATFORM"
echo "============================================"

# Generate bench script with NVTX markers
BENCH_SCRIPT="$OUT_DIR/_bench.py"
cat > "$BENCH_SCRIPT" << 'PYEOF'
import sys, os, time, json, numpy as np
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

def main():
    # Apply NVTX patch
    patch_path = os.environ.get("PATCH_NVTX", "")
    if patch_path and os.path.exists(patch_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("patch_nvtx", patch_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.apply()

    import torch
    try:
        import nvtx
        has_nvtx = True
    except ImportError:
        has_nvtx = False

    from vllm import LLM, SamplingParams

    model = sys.argv[1]
    input_len = int(sys.argv[2])
    output_len = int(sys.argv[3])
    num_prompts = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    max_model_len = input_len + output_len + 64

    print(f"Loading model: {model}")
    print(f"Config: input={input_len}, output={output_len}, prompts={num_prompts}, batch={batch_size}")

    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True)

    sp = SamplingParams(max_tokens=output_len, temperature=0, ignore_eos=True)
    sp_prefill = SamplingParams(max_tokens=1, temperature=0)

    prompts = [{"prompt_token_ids": np.random.randint(0, 10000, size=input_len).tolist()}
               for _ in range(num_prompts + batch_size + 5)]

    # Warmup
    if has_nvtx:
        rng = nvtx.start_range("warmup", color="red")
    print("Warmup...")
    llm.generate(prompts[:batch_size], sampling_params=sp)
    if has_nvtx:
        nvtx.end_range(rng)
    print("Warmup done")

    # Prefill-only round (max_tokens=1)
    prefill_prompts = prompts[batch_size:batch_size + batch_size]
    if has_nvtx:
        rng = nvtx.start_range("prefill", color="blue")
    t0 = time.perf_counter()
    llm.generate(prefill_prompts, sampling_params=sp_prefill)
    ttft_ms = (time.perf_counter() - t0) * 1000 / len(prefill_prompts)
    if has_nvtx:
        nvtx.end_range(rng)
    print(f"Prefill: TTFT={ttft_ms:.2f}ms")

    # Decode rounds
    results = []
    idx = batch_size * 2
    n_rounds = max(num_prompts // batch_size, 3)
    for r in range(n_rounds):
        batch = prompts[idx:idx + batch_size]
        if has_nvtx:
            rng = nvtx.start_range(f"decode_{r}", color="green")
        t0 = time.perf_counter()
        outputs = llm.generate(batch, sampling_params=sp)
        total_ms = (time.perf_counter() - t0) * 1000
        if has_nvtx:
            nvtx.end_range(rng)

        per_req = total_ms / len(batch)
        toks = [len(o.outputs[0].token_ids) for o in outputs]
        avg_tok = sum(toks) / len(toks)
        tpot = (per_req - ttft_ms) / max(avg_tok - 1, 1)
        results.append({"total_ms": per_req, "ttft_ms": ttft_ms, "tpot_ms": tpot, "tokens": avg_tok})
        print(f"Round {r}: total={per_req:.1f}ms, TTFT={ttft_ms:.1f}ms, TPOT={tpot:.2f}ms, tokens={toks}")
        idx += batch_size

    # Summary
    tpots = sorted(r["tpot_ms"] for r in results)
    totals = sorted(r["total_ms"] for r in results)
    mid = len(results) // 2
    summary = {
        "model": model,
        "input_len": input_len,
        "output_len": output_len,
        "ttft_median_ms": round(ttft_ms, 2),
        "tpot_median_ms": round(tpots[mid], 3),
        "total_median_ms": round(totals[mid], 2),
        "output_tokens": round(sum(r["tokens"] for r in results) / len(results)),
    }
    print(f"\n{'='*50}")
    print(f"TTFT:  {summary['ttft_median_ms']:.2f} ms")
    print(f"TPOT:  {summary['tpot_median_ms']:.3f} ms")
    print(f"Total: {summary['total_median_ms']:.2f} ms")
    print(f"Tokens: {summary['output_tokens']}")
    print(f"{'='*50}")

    out_file = os.environ.get("BENCH_OUTPUT", "bench_result.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_file}")

    del llm

if __name__ == "__main__":
    main()
PYEOF

echo ""
PATCH_NVTX="$SCRIPT_DIR/patch_vllm_batch_nvtx.py"
export PATCH_NVTX

for OUTLEN in $OUTPUT_LENS; do
    RUN_DIR="$OUT_DIR/out${OUTLEN}"
    mkdir -p "$RUN_DIR"

    echo "=== Input=${INPUT_LEN}, Output=${OUTLEN} ==="

    if [ "$PLATFORM" = "ppu" ]; then
        BENCH_OUTPUT="$RUN_DIR/result.json" \
        asys profile -o "$RUN_DIR/trace.report" -f true \
            -t hggc,acdnn,acblas,hgtx \
            python3 "$BENCH_SCRIPT" "$MODEL" "$INPUT_LEN" "$OUTLEN" \
                "$NUM_PROMPTS" "$BATCH_SIZE" \
            2>&1 | tee "$RUN_DIR/bench.log"

        # Export sqlite
        sleep 2
        ASYS_REP=$(find "$RUN_DIR" -name "*.asysrep" -o -name "trace.report" 2>/dev/null | head -1)
        [ -n "$ASYS_REP" ] && asys export --force-overwrite true \
            -o "$RUN_DIR/trace.sqlite" "$ASYS_REP" 2>/dev/null || true
    else
        BENCH_OUTPUT="$RUN_DIR/result.json" \
        nsys profile -t cuda,nvtx --cuda-graph-trace=node --cuda-trace-scope=system-wide \
            --force-overwrite=true -o "$RUN_DIR/trace" \
            python3 "$BENCH_SCRIPT" "$MODEL" "$INPUT_LEN" "$OUTLEN" \
                "$NUM_PROMPTS" "$BATCH_SIZE" \
            2>&1 | tee "$RUN_DIR/bench.log"

        nsys stats -r cuda_gpu_kern_sum --format csv --force-export=true \
            "$RUN_DIR/trace.nsys-rep" > /dev/null 2>&1 || true
    fi

    # Extract TTFT/TPOT from trace (kernel time, not wall clock)
    TRACE_SQLITE="$RUN_DIR/trace.sqlite"
    if [ -f "$TRACE_SQLITE" ]; then
        echo "  Extracting kernel times from trace..."
        python3 "$SCRIPT_DIR/compensate_ppu.py" \
            --model-dir "$MODEL" \
            --asys-sqlite "$TRACE_SQLITE" \
            --output-len "$OUTLEN" \
            --output-json "$RUN_DIR/trace_result.json" \
            2>&1 | tee -a "$RUN_DIR/bench.log" || true
    fi

    echo "[Done] Output=${OUTLEN} → $RUN_DIR/"
    echo ""
done

# Summary with decode BW utilization
echo "============================================"
echo "  Results: $OUT_DIR"
echo ""
for OUTLEN in $OUTPUT_LENS; do
    RUN_DIR="$OUT_DIR/out${OUTLEN}"
    echo "  Output=${OUTLEN}:"

    # Wall clock (from Python timer)
    if [ -f "$RUN_DIR/result.json" ]; then
        python3 -c "
import json
d = json.load(open('$RUN_DIR/result.json'))
print(f'    [wall]  TTFT={d[\"ttft_median_ms\"]:.2f}ms  TPOT={d[\"tpot_median_ms\"]:.3f}ms')
" 2>/dev/null || true
    fi

    # Kernel time (from trace) + decode BW utilization
    if [ -f "$RUN_DIR/trace_result.json" ]; then
        python3 -c "
import json, sys

d = json.load(open('$RUN_DIR/trace_result.json'))
t = d.get('tail', {})
tpot = t.get('tpot_ms', 0)
ttft_k = t.get('ttft_kernel_ms', 0)
ttft_w = t.get('ttft_wall_ms', 0)

print(f'    [trace] TTFT_kernel={ttft_k:.2f}ms  TTFT_wall={ttft_w:.2f}ms  TPOT={tpot:.4f}ms')

# Decode BW utilization for Qwen3-30B-A3B-GPTQ-Int4 (single GPU, TP=1)
# attn + shared expert (6144) + MoE top-8 (768/expert)
H=2048; qd=32*128; kvd=4*128
moe_ffn=768; shared_ffn=6144; top_k=8; layers=48
attn = H*qd + H*kvd + H*kvd + qd*H
shared = 3 * H * shared_ffn
moe = top_k * 3 * H * moe_ffn
active_params = (attn + shared + moe) * layers
weight_gb = active_params * 0.5 / 1e9  # INT4
peak_bw = 680  # GB/s
bw_floor_ms = weight_gb / peak_bw * 1000

if tpot > 0:
    bw_util = bw_floor_ms / tpot * 100
    print(f'    [decode BW] active={active_params/1e9:.2f}B × INT4 = {weight_gb:.2f}GB/step')
    print(f'    [decode BW] BW_floor={bw_floor_ms:.2f}ms @ {peak_bw}GB/s  TPOT={tpot:.2f}ms  BW_util={bw_util:.0f}%')
" 2>/dev/null || true
    fi
done
echo "============================================"
