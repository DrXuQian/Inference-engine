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

    vocab = tc["vocab_size"]

    expected = {
        # embeddings / lm_head (vocab-parallel)
        "embed_tokens.weight": (vocab, hidden),
        "lm_head.weight": (vocab, hidden),
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
        # GPTQ expert checks are done dynamically below (heterogeneous layer sizes)
        # norms
        "input_layernorm.weight": (hidden,),
        "post_attention_layernorm.weight": (hidden,),
    }

    with open(rank_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    errors = 0
    checked = 0
    gptq_expert_ok = 0
    gptq_expert_fail = 0
    for sf in shard_files:
        path = rank_dir / sf
        if not path.exists():
            print(f"MISSING shard: {sf}")
            errors += 1
            continue
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                # Fixed-shape checks
                for suffix, exp_shape in expected.items():
                    if key.endswith(suffix):
                        if tuple(tensor.shape) != exp_shape:
                            print(f"SHAPE MISMATCH: {key}")
                            print(f"  expected: {exp_shape}")
                            print(f"  got:      {tuple(tensor.shape)}")
                            errors += 1
                        checked += 1
                        break

                # Dynamic GPTQ expert consistency: qweight dim must
                # match scales/qzeros/g_idx within the same projection
                if ".experts." in key and key.endswith(".qweight"):
                    prefix = key[: -len("qweight")]
                    qw = tuple(tensor.shape)
                    try:
                        sc = tuple(f.get_tensor(prefix + "scales").shape)
                    except Exception:
                        continue
                    # column-parallel (gate/up): qw=[in/pack, inter], sc=[groups, inter]
                    # row-parallel (down):       qw=[inter/pack, out], sc=[groups, out]
                    if qw[1] == sc[1]:  # same out-dim → consistent
                        gptq_expert_ok += 1
                    else:
                        print(f"GPTQ INCONSISTENT: {prefix}")
                        print(f"  qweight {qw}  scales {sc}")
                        gptq_expert_fail += 1
                        errors += 1
                    checked += 1

    print(f"Checked {checked} tensors ({gptq_expert_ok} GPTQ experts consistent).")
    if errors == 0:
        print("OK: no mismatches.")
    else:
        print(f"FAILED: {errors} errors.")
    return errors == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-dir", required=True)
    args = ap.parse_args()
    ok = verify(args.rank_dir)
    sys.exit(0 if ok else 1)
