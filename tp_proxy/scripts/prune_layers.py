#!/usr/bin/env python3
"""
Prune transformer layers from a TP-split rank directory.

Keep the first N layers, drop the rest.

Usage:
    python prune_layers.py --rank-dir /path/to/rank_0 --num-layers 10 --output-dir /path/to/rank_0_10L
"""

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser(
        description="Prune layers from a TP-split rank directory")
    ap.add_argument("--rank-dir", required=True, help="Input rank directory")
    ap.add_argument("--output-dir", required=True, help="Output directory")
    ap.add_argument("--num-layers", type=int, required=True,
                    help="Keep first N layers, drop the rest")

    args = ap.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_prune(Path(args.rank_dir), output_dir, args.num_layers)


if __name__ == "__main__":
    main()
