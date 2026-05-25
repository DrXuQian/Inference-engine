# TP Proxy Benchmark Pipeline

## Overview

Single-card proxy for multi-GPU TP inference performance estimation.
Split model by TP, optionally prune layers to fit GPU memory, benchmark, then compensate.

## Pipeline Steps

| Step | Script | Input | Output | Platform |
|---|---|---|---|---|
| 1. Split & Prune | `split_and_prune.py` | model_dir, tp_size, gpu_memory_gb | rank_0 model dir | Both |
| 2. Benchmark | `auto_bench.py` | rank_0 dir, input_len sweep | bench_results.json | Both |
| 3. Compensate | `compensate.py` | bench_results.json, model config | compensated_results.json | Both |

## Step 1: Split & Prune

```bash
python split_and_prune.py \
    --model-dir /path/to/original \
    --tp-size 2 \
    --gpu-memory-gb 80 \
    --output-dir /path/to/output
```

- Auto-computes per-layer weight size from safetensors
- Calculates max layers that fit in single GPU
- Calls split_tp2.py + prune_layers.py internally

## Step 2: Benchmark

```bash
python auto_bench.py \
    --model-dir /path/to/output/rank_0 \
    --input-lens 512,1024,2048,4096 \
    --output-len 256 \
    --num-prompts 10 \
    --output-json bench_results.json
```

- Uses generate_bench.py with NVTX markers (warmup vs bench)
- Sweeps input_lens sequentially
- Records median TTFT and TPOT per input_len

## Step 3: Compensate

```bash
# NVIDIA platform
python compensate.py \
    --bench-results bench_results.json \
    --model-dir /path/to/original \
    --pruned-layers 10 \
    --original-layers 40 \
    --tp-size 2 \
    --platform nvidia \
    --output-json compensated_results.json

# PPU platform
python compensate.py \
    --bench-results bench_results.json \
    --model-dir /path/to/original \
    --pruned-layers 10 \
    --original-layers 40 \
    --tp-size 2 \
    --platform ppu \
    --pccl-ar /usr/local/PPU_SDK/pccl_tools/all_reduce_perf \
    --pccl-ag /usr/local/PPU_SDK/pccl_tools/all_gather_perf \
    --output-json compensated_results.json
```

### Compensation Components

| Component | TPOT | TTFT | Method |
|---|---|---|---|
| LM head | -delta_decode | -delta_prefill | torch.mm benchmark |
| Communication | +comm_per_step | ≈0 (overlap) | nccl-tests / pccl_tools |
| Encoder block | +(orig-pruned)×per_layer | +(orig-pruned)×per_layer | Differential: bench 2 layer counts |

### Communication Bench

**NVIDIA**: `nccl_bench.py` with torch.distributed
**PPU**: pccl_tools (auto-extracted tensor sizes from config)

```
AR size (decode):  hidden_size × 2 bytes (bf16) = 4KB for hidden=2048
AR size (prefill): chunk_size × hidden_size × 2 bytes = 8MB
AG size (lm_head): batch × (vocab_size/tp) × 2 bytes
```

### Encoder Block Compensation (Differential)

Run generate_bench.py at two layer counts (e.g., N/2 and N):
```
per_layer = (TPOT_high - TPOT_low) / (N_high - N_low)
```
No kernel name matching needed. Works across NVIDIA/PPU.

## Validated Accuracy

| Metric | Method | Error |
|---|---|---|
| TPOT | proxy + lm_head + nsys comm | +0.0% |
| TPOT | proxy + lm_head + nccl standalone | +0.1% (with CUDA Graph, 50-prompt median) |
| TTFT | proxy - lm_head | +8.9% (non-kernel overhead, cannot compensate) |
| Encoder block (10L→40L) | nsys encoder-span | +0.5% |
| Encoder block (10L→40L) | differential (5L+10L regression) | +2.8% |
