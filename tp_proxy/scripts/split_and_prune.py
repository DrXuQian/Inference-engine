#!/usr/bin/env python3
"""
Split model by TP and prune layers to fit single GPU memory.

Auto-computes:
  1. Per-layer weight size from safetensors
  2. Max layers that fit in gpu_memory_gb
  3. Splits by TP, then prunes to computed layer count

Usage:
    python split_and_prune.py \
        --model-dir /path/to/original \
        --tp-size 2 \
        --gpu-memory-gb 80 \
        --output-dir /path/to/output
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg), cfg


def estimate_layer_size(model_dir: str, num_layers: int) -> tuple[float, float]:
    """Estimate per-layer and base (non-layer) weight size in bytes."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        # Single safetensors file - estimate from config
        return _estimate_from_config(model_dir, num_layers)

    with open(index_path) as f:
        index = json.load(f)

    layer_bytes = 0
    base_bytes = 0
    layer_pattern = re.compile(r"model\.language_model\.layers\.(\d+)\.")

    # Use weight_map to categorize without loading tensors
    from safetensors import safe_open

    shard_sizes = {}  # cache per-shard measurements
    for tensor_name, shard_file in index["weight_map"].items():
        if shard_file not in shard_sizes:
            shard_sizes[shard_file] = {}

        m = layer_pattern.match(tensor_name)
        if m:
            layer_id = int(m.group(1))
            shard_sizes[shard_file].setdefault("layer", set()).add(layer_id)
        else:
            shard_sizes[shard_file].setdefault("base", []).append(tensor_name)

    # Measure actual tensor sizes from a sample shard
    total_layer = 0
    total_base = 0
    counted_layers = set()

    for shard_file in sorted(set(index["weight_map"].values())):
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            continue
        with safe_open(shard_path, framework="numpy") as f:
            for key in f.keys():
                nbytes = f.get_tensor(key).nbytes
                m = layer_pattern.match(key)
                if m:
                    total_layer += nbytes
                    counted_layers.add(int(m.group(1)))
                else:
                    total_base += nbytes
        # Only need to scan enough shards to see all layers
        if len(counted_layers) >= num_layers:
            break

    if not counted_layers:
        # Fallback: divide total evenly
        total = index.get("metadata", {}).get("total_size", 0)
        return total / num_layers, 0

    per_layer = total_layer / len(counted_layers)
    return per_layer, total_base


def _estimate_from_config(model_dir: str, num_layers: int) -> tuple[float, float]:
    """Rough estimate from config when safetensors aren't available."""
    tc, _ = load_config(model_dir)
    hidden = tc.get("hidden_size", 2048)
    vocab = tc.get("vocab_size", 248320)
    n_experts = tc.get("num_experts", 256)
    moe_inter = tc.get("moe_intermediate_size", 512)

    # Per-layer: attention (4 projections) + MoE experts + shared expert + norms
    attn_bytes = 4 * hidden * hidden * 2  # bf16
    expert_bytes = n_experts * 3 * hidden * moe_inter * 0.5  # GPTQ int4 ≈ 0.5 bytes
    shared_bytes = 3 * hidden * moe_inter * 2  # bf16
    norm_bytes = 2 * hidden * 2
    per_layer = attn_bytes + expert_bytes + shared_bytes + norm_bytes

    # Base: embed + lm_head + visual
    base = 2 * vocab * hidden * 2  # embed + lm_head, bf16
    base += 50 * 1024 * 1024  # visual encoder ~50MB

    return per_layer, base


def estimate_kv_cache(model_dir: str, num_layers: int, tp_size: int,
                      max_seq_len: int) -> float:
    """Estimate KV cache size in bytes for a given sequence length."""
    tc, _ = load_config(model_dir)

    # Only full_attention layers have KV cache
    # Linear attention (Mamba/SSM) has state cache but much smaller
    layer_types = tc.get("layer_types", ["full_attention"] * num_layers)
    n_full_attn = sum(1 for lt in layer_types[:num_layers] if lt == "full_attention")
    n_linear_attn = num_layers - n_full_attn

    # Full attention KV cache: 2 (K+V) × n_kv_heads/tp × head_dim × 2 bytes × seq_len
    n_kv_heads = tc.get("num_key_value_heads", 2)
    head_dim = tc.get("head_dim", 256)
    kv_per_token_per_layer = 2 * (n_kv_heads // max(tp_size, 1)) * head_dim * 2  # bf16

    # Linear attention state cache: much smaller (fixed per layer, not per token)
    # Mamba state: d_model × d_state × 2 bytes ≈ hidden × 16 × 2
    hidden = tc.get("hidden_size", 2048)
    lin_state_per_layer = hidden * 16 * 2  # approximate

    kv_total = (n_full_attn * kv_per_token_per_layer * max_seq_len +
                n_linear_attn * lin_state_per_layer)
    return kv_total


def compute_max_layers(per_layer_bytes: float, base_bytes: float,
                       gpu_memory_gb: float, num_layers: int,
                       tp_size: int, model_dir: str,
                       max_seq_len: int = 4096) -> int:
    """Compute max layers after TP split that fit in GPU memory.

    Accounts for: weights + KV cache + activation overhead.
    """
    available = gpu_memory_gb * 1e9

    # After TP split: per-layer shrinks proportionally, base stays same (replicated)
    per_layer_tp = per_layer_bytes / tp_size if tp_size > 1 else per_layer_bytes
    base_tp = base_bytes  # replicated (embed, lm_head, visual, norms)

    # Activation overhead: ~500MB fixed
    activation_overhead = 500 * 1024 * 1024

    # Binary search for max layers that fit (weights + KV cache)
    best = 1
    for n in range(1, num_layers + 1):
        weight_bytes = base_tp + n * per_layer_tp
        kv_bytes = estimate_kv_cache(model_dir, n, tp_size, max_seq_len)
        total = weight_bytes + kv_bytes + activation_overhead
        if total <= available:
            best = n
        else:
            break

    return best


def main():
    ap = argparse.ArgumentParser(description="Split + prune model for single-GPU proxy")
    ap.add_argument("--model-dir", required=True, help="Original model directory")
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--gpu-memory-gb", type=float, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="Max sequence length for KV cache estimation")
    ap.add_argument("--max-layers", type=int, default=None,
                    help="Override auto-computed layer count")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tc, full_cfg = load_config(args.model_dir)
    num_layers = tc["num_hidden_layers"]

    print(f"Model: {args.model_dir}")
    print(f"  num_hidden_layers: {num_layers}")
    print(f"  hidden_size: {tc.get('hidden_size')}")
    print(f"  vocab_size: {tc.get('vocab_size')}")
    print(f"  TP size: {args.tp_size}")
    print(f"  GPU memory: {args.gpu_memory_gb} GB")

    # Step 1: Estimate sizes
    print("\nEstimating layer sizes...")
    try:
        import ml_dtypes  # noqa
    except ImportError:
        pass
    per_layer, base = estimate_layer_size(args.model_dir, num_layers)
    print(f"  Per-layer: {per_layer / 1e6:.1f} MB")
    print(f"  Base (non-layer): {base / 1e6:.1f} MB")

    # Step 2: Compute max layers (weights + KV cache + activations)
    kv_full = estimate_kv_cache(args.model_dir, num_layers, args.tp_size, args.max_seq_len)
    print(f"  KV cache ({num_layers}L, seq={args.max_seq_len}): {kv_full / 1e6:.1f} MB")

    if args.max_layers is not None:
        max_layers = args.max_layers
    else:
        max_layers = compute_max_layers(per_layer, base, args.gpu_memory_gb,
                                        num_layers, args.tp_size,
                                        args.model_dir, args.max_seq_len)
    kv_pruned = estimate_kv_cache(args.model_dir, max_layers, args.tp_size, args.max_seq_len)
    per_layer_tp = per_layer / args.tp_size if args.tp_size > 1 else per_layer
    weight_pruned = base + max_layers * per_layer_tp
    print(f"  Weights ({max_layers}L, TP={args.tp_size}): {weight_pruned / 1e9:.2f} GB")
    print(f"  KV cache ({max_layers}L): {kv_pruned / 1e6:.1f} MB")
    print(f"  Total estimated: {(weight_pruned + kv_pruned + 500*1024*1024) / 1e9:.2f} GB / {args.gpu_memory_gb} GB")
    print(f"  Max layers for {args.gpu_memory_gb}GB: {max_layers} / {num_layers}")

    # Step 3: Split (skip if TP=1)
    if args.tp_size > 1:
        split_dir = os.path.join(args.output_dir, "split")
        print(f"\nStep 1/2: Splitting by TP={args.tp_size}...")
        cmd = [sys.executable, os.path.join(script_dir, "split_tp2.py"),
               "--model-dir", args.model_dir,
               "--output-dir", split_dir]
        subprocess.run(cmd, check=True)
        source_dir = os.path.join(split_dir, "rank_0")
    else:
        print(f"\nStep 1/2: TP=1, no splitting needed")
        source_dir = args.model_dir

    # Step 4: Prune
    rank0_pruned = os.path.join(args.output_dir, f"rank_0_{max_layers}L")
    if max_layers < num_layers:
        print(f"\nStep 2/2: Pruning to {max_layers} layers...")
        cmd = [sys.executable, os.path.join(script_dir, "prune_layers.py"),
               "--rank-dir", source_dir,
               "--num-layers", str(max_layers),
               "--output-dir", rank0_pruned]
        subprocess.run(cmd, check=True)
    else:
        print(f"\nStep 2/2: No pruning needed ({max_layers} == {num_layers})")
        rank0_pruned = source_dir

    # Save metadata
    meta = {
        "original_model": args.model_dir,
        "tp_size": args.tp_size,
        "gpu_memory_gb": args.gpu_memory_gb,
        "original_layers": num_layers,
        "pruned_layers": min(max_layers, num_layers),
        "per_layer_bytes": per_layer,
        "base_bytes": base,
        "output_dir": rank0_pruned,
    }
    meta_path = os.path.join(args.output_dir, "split_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone!")
    print(f"  Model: {rank0_pruned}")
    print(f"  Layers: {min(max_layers, num_layers)} / {num_layers}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
