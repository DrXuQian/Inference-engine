#!/usr/bin/env python3
"""
Compensate encoder block + lm_head via differential method.

Runs generate_bench at two layer counts (aligned to attention cycle),
diffs to get per-layer time. Also measures lm_head delta via torch.mm.

Usage:
    python compensate_encoder.py \
        --model-dir /path/to/rank_0 \
        --pruned-layers 10 --original-layers 40 \
        --tp-size 2 \
        --input-lens 512,1024 --output-len 256 \
        --output-json encoder_comp.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil


def get_attn_cycle(model_dir: str) -> int:
    """Get attention type cycle length from config."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
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


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    return {"hidden_size": tc["hidden_size"], "vocab_size": tc["vocab_size"],
            "num_hidden_layers": tc["num_hidden_layers"]}


def measure_lm_head(hidden: int, vocab: int, tp_size: int,
                    input_lens: list[int]) -> dict:
    """LM head delta: full vocab vs split vocab via torch.mm."""
    if tp_size <= 1:
        return {"decode_delta_ms": 0,
                "prefill_deltas_ms": {str(il): 0 for il in input_lens}}

    import torch
    device, dtype = "cuda", torch.bfloat16
    vocab_half = vocab // tp_size

    def bench_mm(M, K, N, warmup=50, iters=200):
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        for _ in range(warmup): torch.mm(A, B)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); torch.mm(A, B); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]

    decode_delta = bench_mm(1, hidden, vocab) - bench_mm(1, hidden, vocab_half)
    prefill_deltas = {}
    for il in input_lens:
        prefill_deltas[il] = bench_mm(il, hidden, vocab) - bench_mm(il, hidden, vocab_half)

    return {"decode_delta_ms": round(decode_delta, 4),
            "prefill_deltas_ms": {str(k): round(v, 4) for k, v in prefill_deltas.items()}}


def run_bench(model_dir: str, input_len: int, output_len: int,
              gpu_mem: float) -> dict:
    """Run generate_bench.py, return results dict."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_bench.py")
    tmp = tempfile.mktemp(suffix=".json")
    mml = input_len + output_len + 64
    env = os.environ.copy()
    env["TRITON_BACKENDS_IN_TREE"] = "1"

    subprocess.run([sys.executable, script,
                    "--model", model_dir,
                    "--input-len", str(input_len), "--output-len", str(output_len),
                    "--num-prompts", "5", "--num-warmup", "2",
                    "--max-model-len", str(mml), "--gpu-mem", str(gpu_mem),
                    "--output-json", tmp],
                   env=env, capture_output=True, timeout=600)

    with open(tmp) as f:
        data = json.load(f)
    os.remove(tmp)
    return data


def measure_encoder_diff(model_dir: str, pruned_layers: int,
                         input_len: int, output_len: int,
                         gpu_mem: float) -> dict:
    """Differential: bench at two cycle-aligned layer counts."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prune_script = os.path.join(script_dir, "prune_layers.py")

    cycle = get_attn_cycle(model_dir)
    n_hi = (pruned_layers // cycle) * cycle
    n_lo = max(n_hi // 2, cycle)
    n_lo = (n_lo // cycle) * cycle
    if n_lo == n_hi:
        n_lo = max(n_hi - cycle, cycle)

    print(f"  Attention cycle: {cycle}")
    print(f"  Differential: {n_lo}L vs {n_hi}L")

    tmp_dir = tempfile.mkdtemp(prefix="enc_diff_")
    results = {}

    for n in [n_lo, n_hi]:
        pruned_dir = os.path.join(tmp_dir, f"pruned_{n}L")
        subprocess.run([sys.executable, prune_script,
                        "--rank-dir", model_dir, "--num-layers", str(n),
                        "--output-dir", pruned_dir],
                       capture_output=True, check=True, timeout=300)

        results[n] = run_bench(pruned_dir, input_len, output_len, gpu_mem)
        print(f"  {n}L: TPOT={results[n]['tpot_median_ms']:.3f}ms "
              f"TTFT={results[n]['ttft_median_ms']:.2f}ms")
        shutil.rmtree(pruned_dir, ignore_errors=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    per_layer_tpot = (results[n_hi]["tpot_median_ms"] - results[n_lo]["tpot_median_ms"]) / (n_hi - n_lo)
    per_layer_ttft = (results[n_hi]["ttft_median_ms"] - results[n_lo]["ttft_median_ms"]) / (n_hi - n_lo)

    return {"cycle": cycle, "n_lo": n_lo, "n_hi": n_hi,
            "per_layer_tpot_ms": round(per_layer_tpot, 4),
            "per_layer_ttft_ms": round(per_layer_ttft, 4)}


def main():
    ap = argparse.ArgumentParser(description="Encoder + lm_head compensation (differential)")
    ap.add_argument("--model-dir", required=True, help="Split rank_0 model dir")
    ap.add_argument("--pruned-layers", type=int, required=True)
    ap.add_argument("--original-layers", type=int, required=True)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--input-lens", default="512", help="Comma-separated input lengths")
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--output-json", default="encoder_comp.json")
    args = ap.parse_args()

    input_lens = [int(x) for x in args.input_lens.split(",")]
    cfg = load_config(args.model_dir)

    print(f"Model: hidden={cfg['hidden_size']}, vocab={cfg['vocab_size']}")
    print(f"Layers: {args.pruned_layers} pruned / {args.original_layers} original")
    print(f"TP={args.tp_size}")
    print()

    # a) LM head
    print("=== LM head (torch.mm differential) ===")
    lm = measure_lm_head(cfg["hidden_size"], cfg["vocab_size"], args.tp_size, input_lens)
    print(f"  Decode delta: {lm['decode_delta_ms']:.3f} ms")
    for il, d in lm["prefill_deltas_ms"].items():
        print(f"  Prefill delta (input={il}): {d:.3f} ms")
    print()

    # b) Encoder block (differential)
    enc = None
    if args.pruned_layers < args.original_layers:
        print("=== Encoder block (layer count differential) ===")
        enc = measure_encoder_diff(args.model_dir, args.pruned_layers,
                                   input_lens[0], args.output_len, args.gpu_mem)
        removed = args.original_layers - args.pruned_layers
        print(f"  Per-layer TPOT: {enc['per_layer_tpot_ms']:.4f} ms")
        print(f"  Per-layer TTFT: {enc['per_layer_ttft_ms']:.4f} ms")
        print(f"  Compensation ({removed} layers): "
              f"TPOT +{removed * enc['per_layer_tpot_ms']:.3f}ms, "
              f"TTFT +{removed * enc['per_layer_ttft_ms']:.3f}ms")
    else:
        print("=== Encoder block: no pruning, skip ===")
    print()

    output = {"lm_head": lm, "encoder_block": enc,
              "tp_size": args.tp_size,
              "pruned_layers": args.pruned_layers,
              "original_layers": args.original_layers}
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {args.output_json}")


if __name__ == "__main__":
    main()
