#!/usr/bin/env python3
"""
Profile CUDA kernels during vLLM inference using torch.profiler.
Groups kernels into encoder-layer vs non-layer, computes per-layer time.

Usage:
    python profile_kernels.py --model /path/to/rank_0_10L --num-layers 10 --original-layers 40
"""

import argparse
import json
import re
import time
from collections import defaultdict

import torch
from torch.profiler import profile, ProfilerActivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--original-layers", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    args = ap.parse_args()

    import os
    os.environ["TRITON_BACKENDS_IN_TREE"] = "1"

    from vllm import LLM, SamplingParams
    import numpy as np

    print(f"Loading model ({args.num_layers}L)...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_mem,
    )

    prompts = [{"prompt_token_ids": np.random.randint(0, 10000, size=args.input_len).tolist()}
               for _ in range(args.batch_size)]
    sp = SamplingParams(max_tokens=args.output_len, temperature=0, ignore_eos=True)

    # Warmup
    print("Warmup...")
    llm.generate(prompts[:1], sampling_params=sp)

    # Profile
    print("Profiling...")
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        llm.generate(prompts, sampling_params=sp)

    # Parse kernel events
    events = prof.key_averages()

    # Classify kernels by name patterns
    layer_keywords = [
        "marlin", "gptq", "gemm", "moe", "expert",    # MoE expert compute
        "flash_attn", "attention", "fmha",              # attention
        "mamba", "ssm", "conv1d",                       # linear attention
        "silu", "gelu",                                  # activations in FFN
        "topk", "softmax",                               # routing
    ]
    non_layer_keywords = [
        "embedding", "embed",                            # embedding lookup
        "vocab", "lm_head",                              # output projection
    ]
    norm_keywords = [
        "layernorm", "rmsnorm", "rms_norm", "layer_norm",
    ]

    layer_time_us = 0
    non_layer_time_us = 0
    norm_time_us = 0
    other_time_us = 0

    layer_kernels = []
    non_layer_kernels = []
    norm_kernels = []
    other_kernels = []

    for evt in events:
        if evt.device_type != torch.device("cuda").type:
            continue
        name = evt.key.lower()
        total_us = evt.cuda_time_total  # microseconds

        is_layer = any(kw in name for kw in layer_keywords)
        is_non_layer = any(kw in name for kw in non_layer_keywords)
        is_norm = any(kw in name for kw in norm_keywords)

        if is_non_layer:
            non_layer_time_us += total_us
            non_layer_kernels.append((evt.key, total_us, evt.count))
        elif is_norm:
            # Norms appear in both layer and non-layer contexts
            # Approximate: most norms are per-layer (2 per layer: input + post_attn)
            norm_time_us += total_us
            norm_kernels.append((evt.key, total_us, evt.count))
        elif is_layer:
            layer_time_us += total_us
            layer_kernels.append((evt.key, total_us, evt.count))
        else:
            other_time_us += total_us
            other_kernels.append((evt.key, total_us, evt.count))

    total_us = layer_time_us + non_layer_time_us + norm_time_us + other_time_us
    N = args.num_layers

    # Norms: ~2 per layer (input_layernorm + post_attn_layernorm) + 1 final
    # Assign proportionally to layer time
    layer_norm_us = norm_time_us * (2 * N) / (2 * N + 1) if N > 0 else 0
    non_layer_norm_us = norm_time_us - layer_norm_us

    encoder_total_us = layer_time_us + layer_norm_us
    fixed_total_us = non_layer_time_us + non_layer_norm_us + other_time_us
    per_layer_us = encoder_total_us / N if N > 0 else 0
    per_layer_ms = per_layer_us / 1000

    print(f"\n{'='*60}")
    print(f"CUDA Kernel Time Breakdown ({N} layers)")
    print(f"{'='*60}")
    print(f"  Encoder layers:  {encoder_total_us/1000:10.2f} ms  "
          f"({encoder_total_us/total_us*100:.1f}%)")
    print(f"  Non-layer fixed: {fixed_total_us/1000:10.2f} ms  "
          f"({fixed_total_us/total_us*100:.1f}%)")
    print(f"  Total CUDA:      {total_us/1000:10.2f} ms")

    print(f"\n  Per encoder block: {per_layer_ms:.2f} ms")

    # Extrapolate
    orig = args.original_layers
    extrap_us = fixed_total_us + orig * per_layer_us
    print(f"\n  Extrapolated {orig}L:")
    print(f"    = {fixed_total_us/1000:.2f} + {orig} × {per_layer_ms:.2f}")
    print(f"    = {extrap_us/1000:.2f} ms")

    # Top kernels
    print(f"\n{'='*60}")
    print("Top 15 encoder-layer kernels:")
    for name, t, cnt in sorted(layer_kernels, key=lambda x: -x[1])[:15]:
        print(f"  {t/1000:8.2f} ms  ×{cnt:5d}  {name[:80]}")

    print(f"\nNon-layer kernels:")
    for name, t, cnt in sorted(non_layer_kernels, key=lambda x: -x[1])[:5]:
        print(f"  {t/1000:8.2f} ms  ×{cnt:5d}  {name[:80]}")

    print(f"\nNorm kernels: {norm_time_us/1000:.2f} ms total")
    print(f"Other/unclassified: {other_time_us/1000:.2f} ms total")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
