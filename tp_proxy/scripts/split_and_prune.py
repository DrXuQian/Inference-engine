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
import struct
import subprocess
import sys
from pathlib import Path


def _read_safetensors_header(path: str) -> dict:
    """Parse the safetensors header to get tensor metadata without loading data."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    header.pop("__metadata__", None)
    return header


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

    # Measure actual tensor sizes by parsing safetensors headers (no tensor loading)
    total_layer = 0
    total_base = 0
    counted_layers = set()

    for shard_file in sorted(set(index["weight_map"].values())):
        shard_path = os.path.join(model_dir, shard_file)
        if not os.path.exists(shard_path):
            continue
        header = _read_safetensors_header(shard_path)
        for key, info in header.items():
            offsets = info["data_offsets"]
            nbytes = offsets[1] - offsets[0]
            m = layer_pattern.match(key)
            if m:
                total_layer += nbytes
                counted_layers.add(int(m.group(1)))
            else:
                total_base += nbytes

    if not counted_layers:
        # Fallback: use metadata total_size
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


def _get_attn_cycle(model_dir: str) -> int:
    """Get attention type cycle length from config (e.g., 4 for [lin,lin,lin,full])."""
    tc, _ = load_config(model_dir)
    interval = tc.get("full_attention_interval")
    if interval:
        return interval
    layer_types = tc.get("layer_types", [])
    if not layer_types:
        return 1
    for cycle in range(1, len(layer_types) + 1):
        pattern = layer_types[:cycle]
        if all(layer_types[i] == pattern[i % cycle] for i in range(len(layer_types))):
            return cycle
    return 1


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
    # gpu_memory_gb is the USABLE budget (not total GPU memory).
    # Caller should pass actual usable memory (e.g., 16GB on a 24GB card,
    # accounting for CUDA context, CUDA Graph, vLLM overhead).
    available = gpu_memory_gb * 1e9

    # After TP split: per-layer shrinks proportionally, base stays same (replicated)
    per_layer_tp = per_layer_bytes / tp_size if tp_size > 1 else per_layer_bytes
    base_tp = base_bytes  # replicated (embed, lm_head, visual, norms)

    # Activation + CUDA Graph + vLLM overhead
    activation_overhead = 1 * 1024 * 1024 * 1024  # 1GB conservative

    # Find max layers that fit, aligned to attention cycle
    cycle = _get_attn_cycle(model_dir)

    best = cycle  # minimum = one full cycle
    for n in range(cycle, num_layers + 1, cycle):
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

    # Build expected metadata for this run
    meta = {
        "original_model": os.path.abspath(args.model_dir),
        "tp_size": args.tp_size,
        "gpu_memory_gb": args.gpu_memory_gb,
        "max_seq_len": args.max_seq_len,
        "original_layers": num_layers,
        "pruned_layers": min(max_layers, num_layers),
        "per_layer_bytes": per_layer,
        "base_bytes": base,
    }

    # Check if output already exists with matching metadata
    meta_path = os.path.join(args.output_dir, "split_meta.json")
    rank0_pruned = os.path.join(args.output_dir, f"rank_0_{max_layers}L")
    if max_layers >= num_layers and args.tp_size <= 1:
        rank0_pruned = args.model_dir

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing_meta = json.load(f)
        # Compare key fields
        match = all(
            existing_meta.get(k) == meta.get(k)
            for k in ["original_model", "tp_size", "original_layers", "pruned_layers", "max_seq_len"]
        )
        if match and os.path.exists(existing_meta.get("output_dir", "")):
            rank0_pruned = existing_meta["output_dir"]
            print(f"\nSkipping split & prune: output already exists with matching config")
            print(f"  Model: {rank0_pruned}")
            print(f"  Layers: {existing_meta['pruned_layers']} / {existing_meta['original_layers']}")
            return

    # Step 3: Split (skip if TP=1)
    if args.tp_size > 1:
        split_dir = os.path.join(args.output_dir, "split")
        print(f"\nStep 1/2: Splitting by TP={args.tp_size}...")
        cmd = [sys.executable, os.path.join(script_dir, "split_tp2.py"),
               "--model-dir", args.model_dir,
               "--output-dir", split_dir,
               "--tp-size", str(args.tp_size)]
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
    meta["output_dir"] = rank0_pruned
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone!")
    print(f"  Model: {rank0_pruned}")
    print(f"  Layers: {min(max_layers, num_layers)} / {num_layers}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
