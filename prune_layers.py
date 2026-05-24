#!/usr/bin/env python3
"""
Prune transformer layers from a TP-split rank directory.

Keeps the first N layers (0..N-1) and drops the rest to fit in GPU memory.
Operates on an already TP-split rank_X directory.

Usage:
    python prune_layers.py --rank-dir /path/to/rank_0 --num-layers 10 --output-dir /path/to/rank_0_10L
    python prune_layers.py --rank-dir /path/to/rank_0 --num-layers 20 --output-dir /path/to/rank_0_20L
"""

import argparse
import copy
import json
import os
import re
import shutil
import time
from pathlib import Path

import ml_dtypes  # noqa: F401
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as np_save_file


def should_keep(name: str, num_layers: int) -> bool:
    """Return True if tensor should be kept for a model with num_layers."""
    # Match model.language_model.layers.{N}.xxx
    m = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
    if m:
        return int(m.group(1)) < num_layers

    # MTP layers are independent (only 1 layer), always keep
    # Visual encoder blocks are independent, always keep
    # All other tensors (embed, lm_head, norms, etc.) always keep
    return True


def main():
    ap = argparse.ArgumentParser(description="Prune layers from a TP-split rank")
    ap.add_argument("--rank-dir", required=True, help="Input rank directory")
    ap.add_argument("--num-layers", type=int, required=True,
                    help="Number of layers to keep (first N)")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    args = ap.parse_args()

    rank_dir = Path(args.rank_dir)
    output_dir = Path(args.output_dir)
    N = args.num_layers

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read and modify config ----
    with open(rank_dir / "config.json") as f:
        cfg = json.load(f)

    orig_cfg = copy.deepcopy(cfg)
    tc = cfg.get("text_config", cfg)
    orig_layers = tc["num_hidden_layers"]
    assert N <= orig_layers, f"num_layers {N} > original {orig_layers}"

    # Update num_hidden_layers
    tc["num_hidden_layers"] = N

    # Update layer_types list (trim to first N)
    if "layer_types" in tc:
        tc["layer_types"] = tc["layer_types"][:N]

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"Config: num_hidden_layers {orig_layers} -> {N}")
    if "layer_types" in tc:
        from collections import Counter
        lt = Counter(tc["layer_types"])
        print(f"  layer_types: {dict(lt)}")

    # ---- Copy auxiliary files ----
    aux = [
        "tokenizer.json", "tokenizer_config.json", "merges.txt",
        "vocab.json", "chat_template.jinja", "generation_config.json",
        "preprocessor_config.json", "video_preprocessor_config.json",
        "configuration.json",
    ]
    for fname in aux:
        src = rank_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(output_dir / fname))

    # ---- Process weight shards ----
    with open(rank_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    new_weight_map = {}
    total_kept = 0
    total_dropped = 0
    t_start = time.time()

    for i, sf in enumerate(shard_files):
        src_path = rank_dir / sf
        if not src_path.exists():
            continue

        kept = {}
        dropped = 0
        with safe_open(str(src_path), framework="numpy") as f:
            for key in f.keys():
                if should_keep(key, N):
                    kept[key] = f.get_tensor(key)
                else:
                    dropped += 1

        if kept:
            np_save_file(kept, str(output_dir / sf))
            for key in kept:
                new_weight_map[key] = sf
            total_kept += len(kept)
        total_dropped += dropped

        print(f"[{i+1}/{len(shard_files)}] {sf}: kept {len(kept)}, dropped {dropped}")

    # ---- Write new index ----
    new_index = {
        "metadata": {},
        "weight_map": new_weight_map,
    }
    # Compute total size
    total_bytes = 0
    for sf in set(new_weight_map.values()):
        with safe_open(str(output_dir / sf), framework="numpy") as f:
            for key in f.keys():
                total_bytes += f.get_tensor(key).nbytes
    new_index["metadata"]["total_size"] = total_bytes

    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    elapsed = time.time() - t_start
    disk_gb = sum(
        p.stat().st_size for p in output_dir.iterdir()
    ) / (1024**3)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Kept:    {total_kept} tensors")
    print(f"  Dropped: {total_dropped} tensors")
    print(f"  Layers:  {orig_layers} -> {N}")
    print(f"  Disk:    {disk_gb:.2f} GB")


if __name__ == "__main__":
    main()
