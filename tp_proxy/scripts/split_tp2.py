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

def _is_gptq_tensor(name: str) -> bool:
    return any(name.endswith(sfx) for sfx in (".qweight", ".qzeros", ".scales", ".g_idx"))


def _col_strategy(name: str) -> tuple[str, int | None]:
    """Column-parallel strategy, auto-detecting GPTQ vs bf16."""
    if _is_gptq_tensor(name):
        return ("replicate", None) if name.endswith(".g_idx") else ("gptq_col", None)
    return "col", 0


def _row_strategy(name: str) -> tuple[str, int | None]:
    """Row-parallel strategy, auto-detecting GPTQ vs bf16."""
    if _is_gptq_tensor(name):
        return ("gptq_gidx", None) if name.endswith(".g_idx") else ("gptq_row", None)
    return "row", 1


def get_split_strategy(name: str, num_kv_heads: int = 0) -> tuple[str, int | None]:
    """
    Determine how to split a tensor for TP.

    Args:
        name: fully qualified tensor name
        num_kv_heads: original (pre-split) num_key_value_heads from config.
            Used to replicate k/v projections when num_kv_heads < TP_SIZE.

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

    # ---- MoE routed expert weights ----
    # Per-expert col/row split (both GPTQ and BF16). vLLM loads per-expert
    # via weight_loader() and fuses internally. Config moe_intermediate_size
    # is already divided by TP in modify_config().
    if ".experts." in n:
        if ".gate_proj." in n or ".up_proj." in n:
            return _col_strategy(n)
        if ".down_proj." in n:
            return _row_strategy(n)

    # ---- full attention (self_attn) ----
    if ".self_attn." in n:
        if ".q_proj." in n:
            return _col_strategy(n)
        if ".k_proj." in n or ".v_proj." in n:
            if 0 < num_kv_heads < TP_SIZE:
                return "replicate", None
            return _col_strategy(n)
        if ".o_proj." in n:
            return _row_strategy(n)

    # ---- linear / Mamba2 attention ----
    if ".linear_attn." in n:
        if _is_gptq_tensor(n):
            if any(p in n for p in (".in_proj_qkv.", ".in_proj_z.", ".in_proj_a.", ".in_proj_b.")):
                return _col_strategy(n)
            if ".out_proj." in n:
                return _row_strategy(n)
        else:
            if any(p in n for p in (".in_proj_qkv.", ".in_proj_z.", ".in_proj_a.", ".in_proj_b.")):
                return "col", 0
            if ".out_proj." in n:
                return "row", 1
        if ".conv1d." in n:
            return "col", 0
        if ".A_log" in n or ".dt_bias" in n:
            return "col_1d", 0

    # ---- shared expert (GPTQ or bf16) ----
    if ".shared_expert." in n:
        if ".gate_proj." in n or ".up_proj." in n:
            return _col_strategy(n)
        if ".down_proj." in n:
            return _row_strategy(n)

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
        if tensor.shape[0] % TP_SIZE != 0:
            return tensor  # can't split evenly → replicate (e.g. kv_heads < tp)
        c = tensor.shape[0] // TP_SIZE
        return tensor[rank * c : (rank + 1) * c].copy()

    if strategy == "row":
        if tensor.shape[1] % TP_SIZE != 0:
            return tensor  # replicate
        c = tensor.shape[1] // TP_SIZE
        return tensor[:, rank * c : (rank + 1) * c].copy()

    if strategy == "gptq_col":
        if tensor.shape[1] % TP_SIZE != 0:
            return tensor  # replicate
        c = tensor.shape[1] // TP_SIZE
        return tensor[:, rank * c : (rank + 1) * c].copy()

    if strategy == "gptq_row":
        if tensor.shape[0] % TP_SIZE != 0:
            return tensor  # replicate
        c = tensor.shape[0] // TP_SIZE
        return tensor[rank * c : (rank + 1) * c].copy()

    if strategy == "gptq_gidx":
        if tensor.shape[0] % TP_SIZE != 0:
            return (np.arange(tensor.shape[0], dtype=np.int32) // GPTQ_GROUP_SIZE)
        c = tensor.shape[0] // TP_SIZE
        return (np.arange(c, dtype=np.int32) // GPTQ_GROUP_SIZE)

    raise ValueError(f"unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Config modification
# ---------------------------------------------------------------------------

def modify_config(config: dict) -> dict:
    """Return a config.json suitable for one TP rank (served with TP=1)."""
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
            # When TP > value, heads are replicated not split — keep original
            if tc[key] >= TP_SIZE:
                tc[key] = tc[key] // TP_SIZE
            # else: keep as-is (replicated)

    return cfg


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------

def process_shard(shard_path: str, rank_dirs: list[str],
                  num_kv_heads: int = 0) -> int:
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
            strategy, _ = get_split_strategy(key, num_kv_heads)
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
    orig_tc = orig_cfg.get("text_config", orig_cfg)
    num_kv_heads = orig_tc.get("num_key_value_heads", 0)
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
    index_path = model_dir / "model.safetensors.index.json"
    single_file = model_dir / "model.safetensors"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
    elif single_file.exists():
        # Single-file model: build index from tensor names
        with safe_open(str(single_file), framework="pt") as st:
            tensor_names = st.keys()
        weight_map = {name: "model.safetensors" for name in tensor_names}
        index = {"metadata": {}, "weight_map": weight_map}
        shard_files = ["model.safetensors"]
        print(f"Single-file model: {len(tensor_names)} tensors")
    else:
        print(f"ERROR: neither {index_path} nor {single_file} found")
        sys.exit(1)

    # ---- process shards ----
    total_tensors = 0
    t_start = time.time()
    for i, sf in enumerate(shard_files):
        path = str(model_dir / sf)
        print(f"[{i+1}/{len(shard_files)}] {sf} …", flush=True)
        t0 = time.time()
        n = process_shard(path, rank_dirs, num_kv_heads)
        total_tensors += n
        print(f"       {n} tensors  ({time.time()-t0:.1f}s)", flush=True)
        if args.delete_shards:
            os.remove(path)
            print(f"       (deleted original)", flush=True)

    # ---- write per-rank index.json (rebuild weight_map from actual saved files) ----
    for r in range(TP_SIZE):
        total_bytes = 0
        weight_map = {}
        rd = Path(rank_dirs[r])
        for sf in shard_files:
            sf_path = rd / sf
            if not sf_path.exists():
                continue
            with safe_open(str(sf_path), framework="numpy") as f:
                for key in f.keys():
                    total_bytes += f.get_tensor(key).nbytes
                    weight_map[key] = sf
        ri = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
        with open(rd / "model.safetensors.index.json", "w") as f:
            json.dump(ri, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s — {total_tensors} tensors × {TP_SIZE} ranks")
    for r in range(TP_SIZE):
        sz = sum(p.stat().st_size for p in Path(rank_dirs[r]).iterdir()) / (1024**3)
        print(f"  rank_{r}: {sz:.2f} GB")


if __name__ == "__main__":
    main()
