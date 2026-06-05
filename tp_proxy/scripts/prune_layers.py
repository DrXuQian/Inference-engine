#!/usr/bin/env python3
"""
Prune or replicate transformer layers from a TP-split rank directory.

Two modes:
  prune (default): Keep the first N layers, drop the rest.
  replicate:       Keep all original layers but copy layer-0 weights to every
                   layer. Only one layer's worth of unique weights on disk,
                   but vLLM sees the full layer count for accurate compute
                   benchmarking.

Operates on an already TP-split rank_X directory.

Usage:
    python prune_layers.py --rank-dir /path/to/rank_0 --num-layers 10 --output-dir /path/to/rank_0_10L
    python prune_layers.py --rank-dir /path/to/rank_0 --num-layers 20 --output-dir /path/to/rank_0_20L
    python prune_layers.py --rank-dir /path/to/rank_0 --replicate --output-dir /path/to/rank_0_rep
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

LAYER_PATTERN = re.compile(
    r"((?:model\.)?(?:language_model\.)?layers\.)(\d+)(\..+)"
)


def should_keep(name: str, num_layers: int) -> bool:
    """Return True if tensor should be kept for a model with num_layers."""
    m = LAYER_PATTERN.search(name)
    if m:
        return int(m.group(2)) < num_layers
    return True


def _copy_aux_files(rank_dir: Path, output_dir: Path) -> None:
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


def _write_index(output_dir: Path, weight_map: dict) -> None:
    total_bytes = 0
    for sf in set(weight_map.values()):
        with safe_open(str(output_dir / sf), framework="numpy") as f:
            for key in f.keys():
                total_bytes += f.get_tensor(key).nbytes
    index = {"metadata": {"total_size": total_bytes}, "weight_map": weight_map}
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)


# -----------------------------------------------------------------------
# Mode: prune
# -----------------------------------------------------------------------

def run_prune(rank_dir: Path, output_dir: Path, num_layers: int) -> None:
    with open(rank_dir / "config.json") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    orig_layers = tc["num_hidden_layers"]
    assert num_layers <= orig_layers, f"num_layers {num_layers} > original {orig_layers}"

    tc["num_hidden_layers"] = num_layers
    if "layer_types" in tc:
        tc["layer_types"] = tc["layer_types"][:num_layers]

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"Config: num_hidden_layers {orig_layers} -> {num_layers}")
    if "layer_types" in tc:
        from collections import Counter
        print(f"  layer_types: {dict(Counter(tc['layer_types']))}")

    _copy_aux_files(rank_dir, output_dir)

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
                if should_keep(key, num_layers):
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

    _write_index(output_dir, new_weight_map)

    elapsed = time.time() - t_start
    disk_gb = sum(p.stat().st_size for p in output_dir.iterdir()) / (1024**3)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Kept:    {total_kept} tensors")
    print(f"  Dropped: {total_dropped} tensors")
    print(f"  Layers:  {orig_layers} -> {num_layers}")
    print(f"  Disk:    {disk_gb:.2f} GB")


# -----------------------------------------------------------------------
# Mode: replicate
# -----------------------------------------------------------------------

def run_replicate(rank_dir: Path, output_dir: Path) -> None:
    """Create model where all layers share layer-0's weight data.

    The safetensors file stores layer-0 data once; all other layers'
    header entries point to the same byte offsets.  File size ≈ 1 layer.
    """
    import struct as _struct

    with open(rank_dir / "config.json") as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    num_layers = tc["num_hidden_layers"]

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"Replicate mode: {num_layers} layers, sharing layer-0 data")

    _copy_aux_files(rank_dir, output_dir)

    with open(rank_dir / "model.safetensors.index.json") as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))

    # Collect layer-0 tensors and non-layer tensors
    layer0_tensors: dict[str, tuple[str, np.ndarray]] = {}
    non_layer_tensors: dict[str, tuple[str, np.ndarray]] = {}

    print("Reading layer-0 and non-layer tensors ...")
    for sf in shard_files:
        src_path = rank_dir / sf
        if not src_path.exists():
            continue
        with safe_open(str(src_path), framework="numpy") as f:
            for key in f.keys():
                m = LAYER_PATTERN.search(key)
                if m:
                    if int(m.group(2)) == 0:
                        suffix = m.group(3)
                        layer0_tensors[suffix] = (key, f.get_tensor(key))
                else:
                    non_layer_tensors[key] = (sf, f.get_tensor(key))

    print(f"  layer-0 tensors: {len(layer0_tensors)}")
    print(f"  non-layer tensors: {len(non_layer_tensors)}")

    if not layer0_tensors:
        print("ERROR: no layer-0 tensors found")
        return

    sample_key = next(iter(layer0_tensors.values()))[0]
    m = LAYER_PATTERN.search(sample_key)
    prefix = m.group(1)

    t_start = time.time()
    new_weight_map = {}

    # Write non-layer tensors (unchanged)
    non_layer_by_shard: dict[str, dict[str, np.ndarray]] = {}
    for key, (sf, arr) in non_layer_tensors.items():
        non_layer_by_shard.setdefault(sf, {})[key] = arr
    for sf, tensors in non_layer_by_shard.items():
        np_save_file(tensors, str(output_dir / sf))
        for key in tensors:
            new_weight_map[key] = sf
    print(f"  wrote {len(non_layer_tensors)} non-layer tensors")

    # Build shared-offset safetensors for replicated layers.
    # Data section: layer-0 tensors written once.
    # Header: all N layers' entries point to the same byte offsets.
    DTYPE_MAP = {
        np.dtype("float16"): "F16",
        np.dtype("bfloat16"): "BF16",
        np.dtype("float32"): "F32",
        np.dtype("float64"): "F64",
        np.dtype("int8"): "I8",
        np.dtype("int16"): "I16",
        np.dtype("int32"): "I32",
        np.dtype("int64"): "I64",
        np.dtype("uint8"): "U8",
        np.dtype("bool"): "BOOL",
    }

    # Compute data layout for layer-0 tensors (sorted by suffix for determinism)
    sorted_suffixes = sorted(layer0_tensors.keys())
    data_chunks = []
    offset = 0
    # suffix -> (dtype_str, shape, begin, end)
    tensor_layout: dict[str, tuple[str, list[int], int, int]] = {}
    for suffix in sorted_suffixes:
        _, arr = layer0_tensors[suffix]
        raw = arr.tobytes()
        dtype_str = DTYPE_MAP.get(arr.dtype)
        if dtype_str is None:
            dtype_str = DTYPE_MAP.get(np.dtype(str(arr.dtype).replace("ml_dtypes.", "")), "U8")
        begin = offset
        end = offset + len(raw)
        tensor_layout[suffix] = (dtype_str, list(arr.shape), begin, end)
        data_chunks.append(raw)
        offset = end

    # Build header: layer-0..N-1 all share same offsets
    header_dict = {}
    for layer_id in range(num_layers):
        for suffix in sorted_suffixes:
            dtype_str, shape, begin, end = tensor_layout[suffix]
            key = f"{prefix}{layer_id}{suffix}"
            header_dict[key] = {"dtype": dtype_str, "shape": shape, "data_offsets": [begin, end]}

    header_json = json.dumps(header_dict, separators=(",", ":")).encode("utf-8")
    data_blob = b"".join(data_chunks)

    shard_name = "model-layers-replicated.safetensors"
    shard_path = output_dir / shard_name
    with open(shard_path, "wb") as f:
        f.write(_struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(data_blob)

    for layer_id in range(num_layers):
        for suffix in sorted_suffixes:
            key = f"{prefix}{layer_id}{suffix}"
            new_weight_map[key] = shard_name

    print(f"  wrote {shard_name}: {len(sorted_suffixes)} unique tensors × {num_layers} layers (shared offsets)")

    _write_index(output_dir, new_weight_map)

    elapsed = time.time() - t_start
    disk_gb = sum(p.stat().st_size for p in output_dir.iterdir()) / (1024**3)
    unique_bytes = sum(arr.nbytes for _, arr in layer0_tensors.values())

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Layers:           {num_layers} (all sharing layer-0 data)")
    print(f"  Unique weights:   {unique_bytes / 1e6:.1f} MB (1 layer)")
    print(f"  Disk total:       {disk_gb:.2f} GB")
    print(f"  Compression:      {num_layers}x vs full copy")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Prune or replicate layers from a TP-split rank")
    ap.add_argument("--rank-dir", required=True, help="Input rank directory")
    ap.add_argument("--output-dir", required=True, help="Output directory")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--num-layers", type=int, default=None,
                      help="Prune: keep first N layers, drop the rest")
    mode.add_argument("--replicate", action="store_true",
                      help="Replicate: copy layer-0 weights to all layers "
                           "(full layer count, minimal unique weights)")

    args = ap.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_dir = Path(args.rank_dir)

    if args.replicate:
        run_replicate(rank_dir, output_dir)
    else:
        run_prune(rank_dir, output_dir, args.num_layers)


if __name__ == "__main__":
    main()
