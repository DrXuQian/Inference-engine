#!/usr/bin/env python3
"""
Split model for arbitrary TP size.

Each rank produces a standalone model directory that can be loaded
by vLLM with tensor_parallel_size=1 for performance benchmarking.

Usage:
    python split_tp2.py --model-dir /path/to/model --output-dir /path/to/output
    python split_tp2.py --model-dir /path/to/model --output-dir /path/to/output --tp-size 4

Outputs:
    output-dir/rank_0/   # Complete model for rank 0
    output-dir/rank_1/   # Complete model for rank 1
    ...
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path

import ml_dtypes  # noqa: F401  — registers bfloat16 with numpy before safetensors uses it
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file

TP_SIZE = 2  # default, overridden by --tp-size
GPTQ_BITS = 4
GPTQ_GROUP_SIZE = 128
GPTQ_PACK_FACTOR = 32 // GPTQ_BITS  # 8 for int4


# ---------------------------------------------------------------------------
# Splitting strategy per tensor
# ---------------------------------------------------------------------------

def get_split_strategy(name: str) -> tuple[str, int | None]:
    """
    Determine how to split a tensor for TP=2.

    Returns (strategy, split_dim).
    Strategies:
      replicate   – copy unchanged
      col         – split dim-0 (column parallel, ≥2-D)
      col_1d      – split dim-0 (1-D tensor)
      row         – split dim-1 (row parallel)
      gptq_col    – GPTQ column parallel (split dim-1 of qweight/scales/qzeros)
      gptq_row    – GPTQ row parallel   (split dim-0 of qweight/scales/qzeros)
      gptq_gidx   – GPTQ row g_idx      (split + regenerate sequential)
    """
    n = name

    # ---- replicate: norms, embeddings, routing, visual, mtp helpers ----
    if "layernorm" in n:
        return "replicate", None
    if "q_norm." in n or "k_norm." in n:
        return "replicate", None
    if ".linear_attn.norm." in n:
        return "replicate", None
    if n.endswith(".norm.weight"):
        return "replicate", None
    if "embed_tokens" in n or n.startswith("lm_head."):
        return "replicate", None
    if ".mlp.gate.weight" in n:
        return "replicate", None
    if "shared_expert_gate" in n:
        return "replicate", None
    if "visual" in n:
        return "replicate", None
    if n.startswith("mtp.fc.") or n.startswith("mtp.norm.") or "pre_fc_norm" in n:
        return "replicate", None

    # ---- GPTQ-quantised expert weights (qweight / scales / qzeros / g_idx) ----
    if ".experts." in n:
        is_gptq = any(n.endswith(sfx) for sfx in (".qweight", ".qzeros", ".scales", ".g_idx"))
        if is_gptq:
            is_col = ".gate_proj." in n or ".up_proj." in n
            is_row = ".down_proj." in n
            if is_col:
                return ("replicate", None) if n.endswith(".g_idx") else ("gptq_col", None)
            if is_row:
                return ("gptq_gidx", None) if n.endswith(".g_idx") else ("gptq_row", None)
        else:
            # bf16 expert weights (MTP layer experts)
            if ".gate_proj." in n or ".up_proj." in n:
                return "col", 0
            if ".down_proj." in n:
                return "row", 1

    # ---- full attention (self_attn) ----
    if ".self_attn." in n:
        if ".q_proj." in n or ".k_proj." in n or ".v_proj." in n:
            return "col", 0
        if ".o_proj." in n:
            return "row", 1

    # ---- linear / Mamba2 attention ----
    if ".linear_attn." in n:
        if any(p in n for p in (".in_proj_qkv.", ".in_proj_z.", ".in_proj_a.", ".in_proj_b.")):
            return "col", 0
        if ".out_proj." in n:
            return "row", 1
        if ".conv1d." in n:
            return "col", 0
        if ".A_log" in n or ".dt_bias" in n:
            return "col_1d", 0

    # ---- shared expert (bf16, not GPTQ) ----
    if ".shared_expert." in n:
        if ".gate_proj." in n or ".up_proj." in n:
            return "col", 0
        if ".down_proj." in n:
            return "row", 1

    # ---- fallback ----
    print(f"  WARNING: unrecognised tensor '{n}', replicating")
    return "replicate", None


# ---------------------------------------------------------------------------
# Tensor splitting
# ---------------------------------------------------------------------------

def split_tensor(tensor: np.ndarray, strategy: str, rank: int) -> np.ndarray:
    """Return the shard of *tensor* belonging to *rank*."""
    if strategy == "replicate":
        return tensor  # no copy needed – safetensors.save_file reads it

    if strategy in ("col", "col_1d"):
        c = tensor.shape[0] // TP_SIZE
        return tensor[rank * c : (rank + 1) * c].copy()

    if strategy == "row":
        c = tensor.shape[1] // TP_SIZE
        return tensor[:, rank * c : (rank + 1) * c].copy()

    if strategy == "gptq_col":
        c = tensor.shape[1] // TP_SIZE
        return tensor[:, rank * c : (rank + 1) * c].copy()

    if strategy == "gptq_row":
        c = tensor.shape[0] // TP_SIZE
        return tensor[rank * c : (rank + 1) * c].copy()

    if strategy == "gptq_gidx":
        # desc_act=false → g_idx is sequential.  Regenerate for local range.
        c = tensor.shape[0] // TP_SIZE
        return (np.arange(c, dtype=np.int32) // GPTQ_GROUP_SIZE)

    raise ValueError(f"unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Config modification
# ---------------------------------------------------------------------------

def modify_config(config: dict) -> dict:
    """Return a config.json suitable for one TP=2 rank (served with TP=1)."""
    cfg = copy.deepcopy(config)
    tc = cfg.get("text_config", cfg)

    for key in (
        "num_attention_heads",
        "num_key_value_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "moe_intermediate_size",
        "shared_expert_intermediate_size",
    ):
        if key in tc:
            tc[key] = tc[key] // TP_SIZE

    return cfg


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------

def process_shard(shard_path: str, rank_dirs: list[str]) -> int:
    """Read one safetensors shard, split every tensor, write both ranks."""
    shard_name = os.path.basename(shard_path)

    # Read all tensors once
    originals: dict[str, np.ndarray] = {}
    with safe_open(shard_path, framework="numpy") as f:
        for key in f.keys():
            originals[key] = f.get_tensor(key)

    # Split + save per rank
    for rank in range(TP_SIZE):
        rank_tensors: dict[str, np.ndarray] = {}
        for key, tensor in originals.items():
            strategy, _ = get_split_strategy(key)
            rank_tensors[key] = split_tensor(tensor, strategy, rank)

        out = os.path.join(rank_dirs[rank], shard_name)
        np_save_file(rank_tensors, out)
        del rank_tensors

    n = len(originals)
    del originals
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TP_SIZE
    ap = argparse.ArgumentParser(description="Split model for TP")
    ap.add_argument("--model-dir", required=True, help="Original model directory")
    ap.add_argument("--output-dir", required=True, help="Output root (rank_0/, rank_1/, ... created inside)")
    ap.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size (default: 2)")
    ap.add_argument("--delete-shards", action="store_true",
                    help="Delete each original shard after processing (saves disk)")
    args = ap.parse_args()
    TP_SIZE = args.tp_size

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    # ---- rank dirs ----
    rank_dirs = []
    for r in range(TP_SIZE):
        d = output_dir / f"rank_{r}"
        d.mkdir(parents=True, exist_ok=True)
        rank_dirs.append(str(d))

    # ---- config ----
    with open(model_dir / "config.json") as f:
        orig_cfg = json.load(f)
    mod_cfg = modify_config(orig_cfg)
    for r in range(TP_SIZE):
        with open(os.path.join(rank_dirs[r], "config.json"), "w") as f:
            json.dump(mod_cfg, f, indent=4, ensure_ascii=False)

    # ---- copy auxiliary files ----
    aux = [
        "tokenizer.json", "tokenizer_config.json", "merges.txt",
        "vocab.json", "chat_template.jinja",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "configuration.json",
    ]
    for fname in aux:
        src = model_dir / fname
        if src.exists():
            for r in range(TP_SIZE):
                shutil.copy2(str(src), os.path.join(rank_dirs[r], fname))

    # ---- copy generation_config.json ----
    gc_src = model_dir / "generation_config.json"
    if gc_src.exists():
        for r in range(TP_SIZE):
            shutil.copy2(str(gc_src), os.path.join(rank_dirs[r], "generation_config.json"))

    # ---- load weight index ----
    with open(model_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    # ---- process shards ----
    total_tensors = 0
    t_start = time.time()
    for i, sf in enumerate(shard_files):
        path = str(model_dir / sf)
        print(f"[{i+1}/{len(shard_files)}] {sf} …", flush=True)
        t0 = time.time()
        n = process_shard(path, rank_dirs)
        total_tensors += n
        print(f"       {n} tensors  ({time.time()-t0:.1f}s)", flush=True)
        if args.delete_shards:
            os.remove(path)
            print(f"       (deleted original)", flush=True)

    # ---- write per-rank index.json ----
    for r in range(TP_SIZE):
        ri = copy.deepcopy(index)
        total_bytes = 0
        rd = Path(rank_dirs[r])
        for sf in shard_files:
            with safe_open(str(rd / sf), framework="numpy") as f:
                for key in f.keys():
                    total_bytes += f.get_tensor(key).nbytes
        ri["metadata"] = {"total_size": total_bytes}
        with open(rd / "model.safetensors.index.json", "w") as f:
            json.dump(ri, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s — {total_tensors} tensors × {TP_SIZE} ranks")
    for r in range(TP_SIZE):
        sz = sum(p.stat().st_size for p in Path(rank_dirs[r]).iterdir()) / (1024**3)
        print(f"  rank_{r}: {sz:.2f} GB")


if __name__ == "__main__":
    main()
