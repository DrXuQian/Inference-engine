#!/usr/bin/env python3
"""
Verify that a TP-split rank directory has consistent tensor shapes
with respect to its config.json.

Usage:
    python verify_split.py --rank-dir /path/to/rank_0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

GPTQ_BITS = 4
GPTQ_GROUP_SIZE = 128
GPTQ_PACK_FACTOR = 32 // GPTQ_BITS


def verify(rank_dir: str) -> bool:
    rank_dir = Path(rank_dir)
    with open(rank_dir / "config.json") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)

    hidden = tc["hidden_size"]
    head_dim = tc["head_dim"]
    n_heads = tc["num_attention_heads"]
    n_kv = tc["num_key_value_heads"]
    moe_inter = tc["moe_intermediate_size"]
    shared_inter = tc["shared_expert_intermediate_size"]
    lin_k_heads = tc.get("linear_num_key_heads", 0)
    lin_v_heads = tc.get("linear_num_value_heads", 0)
    lin_k_dim = tc.get("linear_key_head_dim", 0)
    lin_v_dim = tc.get("linear_value_head_dim", 0)
    has_gate = tc.get("attn_output_gate", False)

    q_out = n_heads * head_dim * (2 if has_gate else 1)
    kv_out = n_kv * head_dim
    o_in = n_heads * head_dim

    lin_qkv = lin_k_heads * lin_k_dim + lin_k_heads * lin_k_dim + lin_v_heads * lin_v_dim
    lin_z = lin_v_heads * lin_v_dim
    lin_out_in = lin_v_heads * lin_v_dim

    expected = {
        # full attention
        "self_attn.q_proj.weight": (q_out, hidden),
        "self_attn.k_proj.weight": (kv_out, hidden),
        "self_attn.v_proj.weight": (kv_out, hidden),
        "self_attn.o_proj.weight": (hidden, o_in),
        "self_attn.q_norm.weight": (head_dim,),
        "self_attn.k_norm.weight": (head_dim,),
        # linear attention
        "linear_attn.in_proj_qkv.weight": (lin_qkv, hidden),
        "linear_attn.in_proj_z.weight": (lin_z, hidden),
        "linear_attn.out_proj.weight": (hidden, lin_out_in),
        "linear_attn.in_proj_a.weight": (lin_v_heads, hidden),
        "linear_attn.in_proj_b.weight": (lin_v_heads, hidden),
        "linear_attn.conv1d.weight": (lin_qkv, 1, tc.get("linear_conv_kernel_dim", 4)),
        "linear_attn.A_log": (lin_v_heads,),
        "linear_attn.dt_bias": (lin_v_heads,),
        "linear_attn.norm.weight": (lin_v_dim,),
        # shared expert
        "shared_expert.gate_proj.weight": (shared_inter, hidden),
        "shared_expert.up_proj.weight": (shared_inter, hidden),
        "shared_expert.down_proj.weight": (hidden, shared_inter),
        # GPTQ expert (gate_proj/up_proj column parallel)
        "experts.0.gate_proj.qweight": (hidden // GPTQ_PACK_FACTOR, moe_inter),
        "experts.0.gate_proj.scales": (hidden // GPTQ_GROUP_SIZE, moe_inter),
        "experts.0.gate_proj.qzeros": (hidden // GPTQ_GROUP_SIZE, moe_inter // GPTQ_PACK_FACTOR),
        "experts.0.gate_proj.g_idx": (hidden,),
        "experts.0.up_proj.qweight": (hidden // GPTQ_PACK_FACTOR, moe_inter),
        "experts.0.up_proj.scales": (hidden // GPTQ_GROUP_SIZE, moe_inter),
        "experts.0.up_proj.qzeros": (hidden // GPTQ_GROUP_SIZE, moe_inter // GPTQ_PACK_FACTOR),
        "experts.0.up_proj.g_idx": (hidden,),
        # GPTQ expert (down_proj row parallel)
        "experts.0.down_proj.qweight": (moe_inter // GPTQ_PACK_FACTOR, hidden),
        "experts.0.down_proj.scales": (moe_inter // GPTQ_GROUP_SIZE, hidden),
        "experts.0.down_proj.qzeros": (moe_inter // GPTQ_GROUP_SIZE, hidden // GPTQ_PACK_FACTOR),
        "experts.0.down_proj.g_idx": (moe_inter,),
        # norms
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
    }

    with open(rank_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    errors = 0
    checked = 0
    for sf in shard_files:
        path = rank_dir / sf
        if not path.exists():
            print(f"MISSING shard: {sf}")
            errors += 1
            continue
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                for suffix, exp_shape in expected.items():
                    if key.endswith(suffix):
                        if tuple(tensor.shape) != exp_shape:
                            print(f"SHAPE MISMATCH: {key}")
                            print(f"  expected: {exp_shape}")
                            print(f"  got:      {tuple(tensor.shape)}")
                            errors += 1
                        checked += 1
                        break

    if errors == 0:
        print(f"OK: verified {checked} tensors, no mismatches.")
    else:
        print(f"FAILED: {errors} errors in {checked} checked tensors.")
    return errors == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-dir", required=True)
    args = ap.parse_args()
    ok = verify(args.rank_dir)
    sys.exit(0 if ok else 1)
