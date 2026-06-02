# Vision Benchmarks

This directory contains PyTorch vision-model benchmarks. Timing follows
`vla/vla_bench.py`: eager warmup, optional CUDA Graph capture, per-iteration
`torch.cuda.synchronize()` around `time.perf_counter()`, sorted samples, and
middle-element median. Profiler traces are collected by `run_vision_bench.sh`,
not by `vision_bench.py` itself.

## Workloads

| Component | Input | Model |
|---|---:|---|
| `vit` | 4 x 3 x 480 x 480 | synthetic ViT-L-like trunk, default 24L x 1024 |
| `clipdino` | 1 x 3 x 480 x 480 | CLIP ViT-B/16 config plus DINO-style MLP head |

The CLIP trunk uses the public `openai/clip-vit-base-patch16` config:

```text
patch_size=16
hidden_size=768
intermediate_size=3072
num_hidden_layers=12
num_attention_heads=12
projection_dim=512
hidden_act=quick_gelu
```

The original config image size is 224. This benchmark intentionally runs
480 x 480 inputs, so the CLIP-DINO path has `30 x 30 + CLS = 901` tokens.

## Direct PyTorch Runs

```bash
cd /root/autodl-tmp/Inference-engine

python3 kernel_bench/vision/vision_bench.py \
  --component vit \
  --dtype bf16 \
  --warmup 10 \
  --iters 30

python3 kernel_bench/vision/vision_bench.py \
  --component clipdino \
  --dtype bf16 \
  --warmup 10 \
  --iters 30
```

Optional CUDA Graph replay, matching `vla_bench.py --cuda-graph`:

```bash
python3 kernel_bench/vision/vision_bench.py \
  --component vit \
  --dtype bf16 \
  --cuda-graph \
  --warmup 10 \
  --iters 30
```

## Profiled Runs

Run components serially:

```bash
bash kernel_bench/vision/run_vision_bench.sh ./results/vision vit

bash kernel_bench/vision/run_vision_bench.sh ./results/vision clipdino

bash kernel_bench/vision/run_vision_bench.sh ./results/vision all
```

PPU systems use `asys` by default:

```bash
PLATFORM=ppu bash kernel_bench/vision/run_vision_bench.sh ./results/vision all
```

CUDA systems use `nsys` with CUDA Graph node tracing enabled in the profiler:

```bash
PLATFORM=cuda bash kernel_bench/vision/run_vision_bench.sh ./results/vision all
```

Useful environment overrides:

```bash
DTYPE=fp16
WARMUP=5
ITERS=20
CUDA_GRAPH=1
TORCH_TRACE=1
```
