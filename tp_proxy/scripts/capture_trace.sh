#!/bin/bash
# Capture trace using offline mode (vllm.LLM) under asys/nsys profiler.
# Guarantees proper batching via llm.generate([batch prompts]).
#
# Usage:
#   bash capture_trace.sh /path/to/model 25600 1024 ./results/trace [num_prompts] [batch_size]
#   PLATFORM=nvidia bash capture_trace.sh /path/to/model 512 100 ./results/trace 20 4

set -euo pipefail

MODEL="$(readlink -f "${1:?Usage: $0 <model_dir> <input_len> <output_len> <output_dir> [num_prompts] [batch_size]}")"
INPUT_LEN="${2:?}"
OUTPUT_LEN="${3:?}"
OUT_DIR="${4:?}"
NUM_PROMPTS="${5:-10}"
BATCH_SIZE="${6:-1}"
MAX_MODEL_LEN=$((INPUT_LEN + OUTPUT_LEN + 64))
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

PLATFORM="${PLATFORM:-ppu}"

# Generate the offline bench script
BENCH_SCRIPT="$OUT_DIR/_bench_offline.py"
cat > "$BENCH_SCRIPT" << 'PYEOF'
import sys, os, numpy as np
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

def patch_batch_nvtx():
    """Patch ModelRunner.execute_model to add NVTX marker with batch size.
    nvtx.range_push/pop is CPU-only (<1us), does not affect GPU timing."""
    import torch
    try:
        from vllm.worker.model_runner import ModelRunner
        _orig = ModelRunner.execute_model
        def _patched(self, *args, **kwargs):
            model_input = args[0] if args else kwargs.get('model_input')
            bs = 0
            if hasattr(model_input, 'input_tokens'):
                bs = model_input.input_tokens.shape[0]
            elif hasattr(model_input, 'seq_lens'):
                bs = len(model_input.seq_lens)
            torch.cuda.nvtx.range_push(f"bs={bs}")
            result = _orig(self, *args, **kwargs)
            torch.cuda.nvtx.range_pop()
            return result
        ModelRunner.execute_model = _patched
        print("Patched ModelRunner with batch NVTX markers")
    except Exception as e:
        print(f"NVTX patch failed (non-fatal): {e}")

def main():
    patch_batch_nvtx()
    from vllm import LLM, SamplingParams

    model = sys.argv[1]
    input_len = int(sys.argv[2])
    output_len = int(sys.argv[3])
    num_prompts = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    max_model_len = int(sys.argv[6])

    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=0.9, trust_remote_code=True)
    sp = SamplingParams(max_tokens=output_len, temperature=0, ignore_eos=True)

    # Warmup
    warmup = [{"prompt_token_ids": np.random.randint(0, 10000, size=input_len).tolist()}
              for _ in range(batch_size)]
    llm.generate(warmup, sampling_params=sp)
    print(f"Warmup done (batch={batch_size})")

    # Bench rounds
    n_rounds = max(num_prompts // batch_size, 3)
    for r in range(n_rounds):
        prompts = [{"prompt_token_ids": np.random.randint(0, 10000, size=input_len).tolist()}
                   for _ in range(batch_size)]
        outputs = llm.generate(prompts, sampling_params=sp)
        toks = [len(o.outputs[0].token_ids) for o in outputs]
        print(f"Round {r}: batch={len(outputs)}, tokens={toks}")

    del llm
    print("Done")

if __name__ == "__main__":
    main()
PYEOF

echo "============================================"
echo "  Capture Trace (offline mode)"
echo "  Model: $MODEL"
echo "  Input=$INPUT_LEN, Output=$OUTPUT_LEN"
echo "  Batch=$BATCH_SIZE, Prompts=$NUM_PROMPTS"
echo "============================================"

# Run under profiler
echo "[1/3] Running offline bench under profiler..."
if [ "$PLATFORM" = "ppu" ]; then
    asys profile -o "$OUT_DIR/trace.report" -f true \
        -t hggc,acdnn,acblas \
        python3 "$BENCH_SCRIPT" "$MODEL" "$INPUT_LEN" "$OUTPUT_LEN" \
            "$NUM_PROMPTS" "$BATCH_SIZE" "$MAX_MODEL_LEN" \
        2>&1 | tee "$OUT_DIR/bench_trace.txt"
else
    TRITON_BACKENDS_IN_TREE=1 \
    nsys profile -t cuda --cuda-trace-scope=system-wide --cuda-graph-trace=node \
        --force-overwrite=true -o "$OUT_DIR/trace" \
        python3 "$BENCH_SCRIPT" "$MODEL" "$INPUT_LEN" "$OUTPUT_LEN" \
            "$NUM_PROMPTS" "$BATCH_SIZE" "$MAX_MODEL_LEN" \
        2>&1 | tee "$OUT_DIR/bench_trace.txt"
fi

# Export sqlite
echo ""
echo "[2/3] Waiting for trace file..."
sleep 3

echo "[3/3] Exporting sqlite..."
if [ "$PLATFORM" = "ppu" ]; then
    ASYS_REP=""
    for i in 1 2 3 4 5; do
        if [ -f "$OUT_DIR/trace.report.asysrep" ]; then
            ASYS_REP="$OUT_DIR/trace.report.asysrep"
        elif [ -f "$OUT_DIR/trace.report" ]; then
            ASYS_REP="$OUT_DIR/trace.report"
        fi
        if [ -n "$ASYS_REP" ]; then break; fi
        echo "  Waiting... (attempt $i)"
        sleep 3
    done
    if [ -z "$ASYS_REP" ]; then
        echo "ERROR: asys report not found"
        ls -la "$OUT_DIR/" || true
        exit 1
    fi
    echo "  Found: $ASYS_REP"
    asys export --force-overwrite true -o "$OUT_DIR/trace.sqlite" "$ASYS_REP" || {
        echo "ERROR: asys export failed"; exit 1
    }
else
    nsys stats -r cuda_gpu_kern_sum --format csv --force-export=true \
        "$OUT_DIR/trace.nsys-rep" > /dev/null 2>&1
fi

echo ""
echo "Done: $OUT_DIR/trace.sqlite"
