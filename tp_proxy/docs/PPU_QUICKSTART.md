# PPU Platform Quick Start

## Prerequisites

- PPU SDK installed (`/usr/local/PPU_SDK/pccl_tools/`)
- asys profiler available
- vLLM installed with PPU backend
- Models at `/sim/eec/shared/models/Qwen/`

## Model Matrix

| Model | TP configs | GPU memory / card |
|---|---|---|
| Qwen3.5-35B-A3B-GPTQ-Int4 | TP=1 | 24 GB |
| Qwen 27B | TP=1 | 24 GB |
| Qwen3.5-122B-A10B-GPTQ-Int4 | TP=1, TP=2 | 24 GB |
| Qwen 397B-A17B | TP=2, TP=4 | 24 GB |

## Step 1: Split & Prune

```bash
# TP=1 (single card, only prune to fit 24GB)
python split_and_prune.py \
    --model-dir /sim/eec/shared/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 \
    --tp-size 1 \
    --gpu-memory-gb 24 \
    --output-dir /path/to/output

# TP=2 (split + prune)
python split_and_prune.py \
    --model-dir /sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --tp-size 2 \
    --gpu-memory-gb 24 \
    --output-dir /path/to/output

# TP=4
python split_and_prune.py \
    --model-dir /sim/eec/shared/models/Qwen/Qwen3.5-397B-A27B-GPTQ-Int4 \
    --tp-size 4 \
    --gpu-memory-gb 24 \
    --output-dir /path/to/output
```

## Step 2: Benchmark

```bash
python auto_bench.py \
    --model-dir /path/to/output/rank_0_NL \
    --input-lens 512,1024,2048,4096,8192,16384,32768,65536,131072 \
    --output-len 256 \
    --num-prompts 10 \
    --output-json bench_results.json
```

## Step 3: Compensate

### Option A: pccl_tools standalone (no TP hardware needed)

```bash
python compensate_ppu.py \
    --bench-results bench_results.json \
    --model-dir /sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --pruned-layers 10 --original-layers 40 \
    --tp-size 2 \
    --pccl-ar /usr/local/PPU_SDK/pccl_tools/all_reduce_perf \
    --pccl-ag /usr/local/PPU_SDK/pccl_tools/all_gather_perf \
    --output-json compensated.json
```

### Option B: asys trace (accurate, requires TP run)

```bash
# 1. Profile TP=2 serving with asys
CUDA_VISIBLE_DEVICES=0,1 asys profile \
    -o vllm.report -f true \
    -t hggc,acdnn,acblas \
    --hggc-memory-usage true \
    vllm serve /sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 &

# 2. Wait for server ready, then send requests
vllm bench serve \
    --model /sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --max-concurrency 1 \
    --base-url http://127.0.0.1:8000 \
    --dataset-name random \
    --random-input-len 512 --random-output-len 256 \
    --num-prompts 20 --request-rate 5 \
    --trust-remote-code

# 3. Kill server, export sqlite
kill %1
asys export -o result.sqlite vllm.report

# 4. Compensate with real comm time
python compensate_ppu.py \
    --bench-results bench_results.json \
    --model-dir /sim/eec/shared/models/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 \
    --pruned-layers 10 --original-layers 40 \
    --tp-size 2 \
    --asys-sqlite result.sqlite \
    --output-json compensated.json
```

## Optional: Analyze Sync Overhead

```bash
# Export trace and analyze all-reduce synchronization
asys export -o result.sqlite vllm.report

python analyze_sync_overhead.py \
    --sqlite result.sqlite \
    --platform ppu

# Filter to specific time range
python analyze_sync_overhead.py \
    --sqlite result.sqlite \
    --platform ppu \
    --start-ns <start> --end-ns <end>
```

## Communication Benchmark (standalone)

```bash
AR=/usr/local/PPU_SDK/pccl_tools/all_reduce_perf
AG=/usr/local/PPU_SDK/pccl_tools/all_gather_perf

# Decode: hidden=2048, bf16 = 4KB
$AR -b 4096 -e 4096 -f 2 -d bf16 -o sum -n 500 -w 100 -g 2 -c 0 -a 1

# Prefill chunk: 2048 × 2048 × bf16 = 8MB
$AR -b 8388608 -e 8388608 -f 2 -d bf16 -o sum -n 100 -w 20 -g 2 -c 0 -a 1

# lm_head all-gather: vocab/tp × bf16
$AG -b 496640 -e 496640 -f 2 -d bf16 -g 2 -n 300 -w 50 -c 0 -a 1
```

Note: Standalone pccl_tools measures PCCL protocol stack latency.
If vLLM uses a custom P2P reduce (like NVIDIA's cross_device_reduce),
actual comm time may be lower. Use asys trace (Option B) for accurate numbers.

## Key Differences from NVIDIA

| | NVIDIA | PPU |
|---|---|---|
| Profiler | nsys | asys |
| Trace export | `nsys stats -r cuda_gpu_trace --format csv` | `asys export -o result.sqlite` |
| Kernel table | `CUPTI_ACTIVITY_KIND_KERNEL` | `HGPTI_ACTIVITY_KIND_KERNEL` |
| NVTX table | `NVTX_EVENTS` | `HGTX_EVENTS` |
| Comm bench | nccl-tests / nccl_bench.py | pccl_tools |
| CUDA Graph trace | `--cuda-graph-trace=node` | equivalent asys flag |
| Compensate script | `compensate.py` | `compensate_ppu.py` |

## Output

Both compensate scripts output JSON:
```json
{
  "platform": "ppu",
  "tp_size": 2,
  "lm_head": {"decode_delta_ms": 0.34, ...},
  "communication": {"total_per_step_ms": 1.85, ...},
  "encoder_block": {"per_layer_tpot_ms": 0.08, ...},
  "results": [
    {
      "input_len": 512,
      "ttft_median_ms": 33.47,
      "tpot_median_ms": 4.58,
      "comp_ttft_ms": 32.83,
      "comp_tpot_ms": 4.71,
      "comp_total_ms": 329.4
    }
  ]
}
```
