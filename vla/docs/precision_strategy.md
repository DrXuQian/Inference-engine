# VLA Pipeline Precision Strategy

## Recommended Configuration (Aligned with DreamZero)

Reference: [DreamZero: World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922)

DreamZero uses NVIDIA Model Optimizer on Blackwell architecture, achieving **38x inference speedup** at 7Hz real-time robotic control with negligible quality loss.

### Precision Mapping

| Component | Operation | Precision | Speedup | DreamZero Reference |
|-----------|-----------|-----------|---------|---------------------|
| **VIT Spatial** | GEMM (Linear) | NVFP4 (E2M1) | 4x | Weight+Activation → NVFP4 |
| | FlashAttention QKV | FP8 (E4M3) | 2x | QKV projections → FP8 |
| | Softmax | FP8 (E4M3) | — | Softmax → FP8 |
| | LayerNorm, GELU | FP16 | — | Non-linear ops → FP16 |
| **VIT Temporal** | GEMM (Q/K/V/Out proj) | NVFP4 (E2M1) | 4x | Same as above |
| | Cross-Attention | FP8 (E4M3) | 2x | QKV → FP8 |
| **LLM Prefill** | GEMM (QKV, FFN) | NVFP4 (E2M1) | 4x | Weight+Activation → NVFP4 |
| | FlashAttention | FP8 (E4M3) | 2x | QKV → FP8 |
| | SwiGLU FFN | NVFP4 (E2M1) | 4x | Linear layers → NVFP4 |
| | LayerNorm, RoPE | FP16 | — | Non-linear → FP16 |
| **DiT** | GEMM (Self-Attn, Cross-Attn, FFN) | NVFP4 (E2M1) | 4x | Video diffusion → NVFP4 |
| | FlashAttention | FP8 (E4M3) | 2x | QKV → FP8 |
| | LayerNorm, GELU | FP16 | — | Non-linear → FP16 |

### Summary per Component

| Component | GEMM | FA/Attention | Non-linear | Overall |
|-----------|------|-------------|------------|---------|
| VIT | NVFP4 | FP8 | FP16 | Mixed FP4/FP8/FP16 |
| LLM | NVFP4 | FP8 | FP16 | Mixed FP4/FP8/FP16 |
| DiT | NVFP4 | FP8 | FP16 | Mixed FP4/FP8/FP16 |

## Projected Performance

Baseline: FP16 on target platform.

| Component | FP16 (ms) | Optimized (ms) | Speedup | Method |
|-----------|-----------|----------------|---------|--------|
| VIT | 46.0 | 17.9 | 2.57x | GEMM 4x, FA 2x |
| LLM | 98.0 | 36.9 | 2.66x | GEMM 4x, FA 2x |
| DiT | 43.0 | 16.7 | 2.57x | GEMM 4x, FA 2x |
| **Total** | **187.0** | **71.5** | **2.62x** | |
| **FPS** | **5.3** | **14.0** | | |

## Research Backup

### NVFP4 for LLM/VIT (GEMM)

- **DreamZero** (2025): 14B video diffusion model, NVFP4 weights+activations with FP8 QKV, FP16 accumulation. 38x speedup, negligible quality loss.
- **FP4 All the Way** (ICML 2025): Fully FP4 training of LLMs achieves accuracy comparable to BF16/FP8. Extended to BERT and ViT (DeiT-S on ImageNet).
- **LLM-FP4** (NeurIPS 2024): 4-bit floating-point quantized transformers, validated on LLaMA, BERT, ViT.
- **RaZeR** (2025): Pushes NVFP4 further with redundant zero remapping, improving quality at FP4.

### FP8 for FlashAttention

- **vLLM FP8 W8A8**: Production-ready FP8 inference, effectively lossless across all model scales.
- **DreamZero**: QKV projections and Softmax maintained at FP8 (E4M3) for stability.
- FP8 attention is the de-facto standard for Blackwell/Hopper inference.

### NVFP4/FP8 for DiT (Diffusion Transformer)

- **DreamZero**: Video diffusion backbone quantized to NVFP4 with FP8 attention, validated in real robot deployment at 7Hz.
- **PTQ4DiT** (2024): W8A8 comparable to FP baseline, W4A8 still generates high-quality images.
- **Q-DiT** (2024): W4A8 on DiT-XL/2 ImageNet, maintains high fidelity.
- **HQ-DiT** (2024): FP4 hybrid quantization specifically designed for DiT.
- **TaQ-DiT** (2024): Time-aware quantization — earlier denoising steps tolerate more aggressive quantization.
- **LRQ-DiT** (2025): Low-bit quantization on PixArt/FLUX, outperforms existing PTQ baselines.

## Key Insight from DreamZero

> "Quantizing model weights and activations to NVFP4 (E2M1) while maintaining sensitive QKV projections and Softmax operations in FP8 (E4M3), and employing FP16 accumulation for non-linear operations including LayerNorm and RoPE. This configuration improves latency with negligible impact on generated video and action quality."

This validates the mixed-precision strategy: **NVFP4 for compute-heavy linear layers, FP8 for attention-sensitive operations, FP16 for numerical-sensitive non-linear operations**.

## References

1. [DreamZero: World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) — NVIDIA, 2025
2. [FP4 All the Way: Fully Quantized Training of LLMs](https://arxiv.org/abs/2505.19115) — ICML 2025
3. [PTQ4DiT: Post-training Quantization for Diffusion Transformers](https://arxiv.org/abs/2405.16005)
4. [Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](https://arxiv.org/abs/2406.17343)
5. [HQ-DiT: Efficient DiT with FP4 Hybrid Quantization](https://arxiv.org/abs/2405.19751)
6. [TaQ-DiT: Time-aware Quantization for Diffusion Transformers](https://arxiv.org/abs/2411.14172)
7. [LRQ-DiT: Log-Rotation PTQ for DiT](https://arxiv.org/abs/2508.03485)
8. [RaZeR: Pushing the Limits of NVFP4 Quantization](https://arxiv.org/abs/2501.04052)
9. [NVFP4 Quantization | DGX Spark](https://build.nvidia.com/spark/nvfp4-quantization)
10. [vLLM FP8 W8A8](https://docs.vllm.ai/en/latest/features/quantization/fp8/)
