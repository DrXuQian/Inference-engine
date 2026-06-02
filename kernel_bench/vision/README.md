# Vision Benchmarks

This directory follows the same split as `vla/`:

- `vision_bench.py` runs the model and emits NVTX ranges plus `results.json`.
- `run_vision_bench.sh` is the default entry point and runs `asys`/`nsys`.
- `vision_report.py` reads `trace.sqlite` first, filters one measured NVTX range,
  and computes GEMM / FlashAttention / Other kernel time. `results.json` is only
  a fallback when sqlite is unavailable.

## Workloads

| Component | Input | Model |
|---|---:|---|
| `vit` | 4 x 3 x 480 x 480 | synthetic ViT-L-like trunk, default 24L x 1024 |
| `clipdino` | 1 x 3 x 480 x 480 | `openai/clip-vit-base-patch16` vision tower |

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

The original CLIP config image size is 224. This benchmark intentionally runs
480 x 480 inputs, so the vision path has `30 x 30 + CLS = 901` tokens. The
`text_config` in the CLIPModel config is not executed for this image-only
benchmark.

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

`all` automatically writes:

```text
./results/vision_YYYYmmdd_HHMMSS/
  vit/results.json
  vit/trace.sqlite
  clipdino/results.json
  clipdino/trace.sqlite
  report.txt
  report.json
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

## Report From Existing Trace

```bash
python3 kernel_bench/vision/vision_report.py \
  --vit-trace ./results/vision_YYYYmmdd_HHMMSS/vit/trace.sqlite \
  --clipdino-trace ./results/vision_YYYYmmdd_HHMMSS/clipdino/trace.sqlite \
  --output-json ./results/vision_YYYYmmdd_HHMMSS/report.json
```
