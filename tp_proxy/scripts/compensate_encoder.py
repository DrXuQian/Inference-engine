#!/usr/bin/env python3
"""
Compensate encoder block + lm_head via differential method.

Reads split_meta.json from the model output directory to get:
  - original_layers, pruned_layers, tp_size, original_model path

What it does:
  1. LM head: torch.mm benchmark (full vocab vs half vocab per decode/prefill)
     → only for TP > 1
  2. Encoder block: prune model to two cycle-aligned layer counts,
     run generate_bench on each, diff to get per-layer TTFT/TPOT
     → only if pruned_layers < original_layers
  3. Apply to bench_results.json → output compensated results

Usage:
    # Auto-read meta from model dir
    python compensate_encoder.py \
        --model-dir ./results/03_chat_122B/tp2/model \
        --bench-results bench.json \
        --input-lens 25600 --output-len 1024

    # Manual override
    python compensate_encoder.py \
        --model-dir ./rank_0_28L \
        --pruned-layers 28 --original-layers 40 --tp-size 2 \
        --bench-results bench.json \
        --input-lens 25600 --output-len 1024
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil


def load_meta(model_dir: str) -> dict | None:
    """Try to load split_meta.json from model_dir or parent."""
    for d in [model_dir, os.path.dirname(model_dir)]:
        meta_path = os.path.join(d, "split_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
    return None


def load_config(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    return {"hidden_size": tc["hidden_size"], "vocab_size": tc["vocab_size"],
            "num_hidden_layers": tc["num_hidden_layers"]}


def get_attn_cycle(model_dir: str) -> int:
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


# ---------------------------------------------------------------------------
# a) LM head
# ---------------------------------------------------------------------------

def measure_lm_head(hidden: int, vocab: int, tp_size: int,
                    input_lens: list[int]) -> dict:
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


# ---------------------------------------------------------------------------
# b) Encoder block differential
# ---------------------------------------------------------------------------

def run_bench(model_dir: str, input_len: int, output_len: int,
              gpu_mem: float) -> dict:
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_bench.py")
    tmp = tempfile.mktemp(suffix=".json")
    env = os.environ.copy()
    env["TRITON_BACKENDS_IN_TREE"] = "1"
    subprocess.run([sys.executable, script,
                    "--model", model_dir,
                    "--input-len", str(input_len), "--output-len", str(output_len),
                    "--num-prompts", "5", "--num-warmup", "2",
                    "--max-model-len", str(input_len + output_len + 64),
                    "--gpu-mem", str(gpu_mem), "--output-json", tmp],
                   env=env, capture_output=True, timeout=600)
    with open(tmp) as f:
        data = json.load(f)
    os.remove(tmp)
    return data


def measure_encoder_diff(model_dir: str, pruned_layers: int,
                         input_len: int, output_len: int,
                         gpu_mem: float) -> dict:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prune_script = os.path.join(script_dir, "prune_layers.py")

    cycle = get_attn_cycle(model_dir)
    n_hi = (pruned_layers // cycle) * cycle
    n_lo = max(n_hi // 2, cycle)
    n_lo = (n_lo // cycle) * cycle
    if n_lo == n_hi:
        n_lo = max(n_hi - cycle, cycle)

    print(f"  Attention cycle: {cycle}")
    print(f"  Differential: {n_lo}L vs {n_hi}L (both multiples of {cycle})")

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

    return {
        "cycle": cycle, "n_lo": n_lo, "n_hi": n_hi,
        "per_layer_tpot_ms": round((results[n_hi]["tpot_median_ms"] - results[n_lo]["tpot_median_ms"]) / (n_hi - n_lo), 4),
        "per_layer_ttft_ms": round((results[n_hi]["ttft_median_ms"] - results[n_lo]["ttft_median_ms"]) / (n_hi - n_lo), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Encoder + lm_head compensation")
    ap.add_argument("--model-dir", required=True,
                    help="Pruned model dir (reads split_meta.json automatically)")
    ap.add_argument("--bench-results", default=None,
                    help="bench.json from auto_bench (optional, for applying compensation)")
    ap.add_argument("--pruned-layers", type=int, default=None,
                    help="Override: pruned layer count")
    ap.add_argument("--original-layers", type=int, default=None,
                    help="Override: original layer count")
    ap.add_argument("--tp-size", type=int, default=None,
                    help="Override: TP size")
    ap.add_argument("--input-lens", default="512",
                    help="Comma-separated input lengths for lm_head prefill measurement")
    ap.add_argument("--output-len", type=int, default=256)
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--output-json", default="encoder_comp.json")
    args = ap.parse_args()

    # Read meta
    meta = load_meta(args.model_dir)
    pruned_layers = args.pruned_layers or (meta and meta.get("pruned_layers"))
    original_layers = args.original_layers or (meta and meta.get("original_layers"))
    tp_size = args.tp_size or (meta and meta.get("tp_size")) or 1

    if pruned_layers is None or original_layers is None:
        cfg = load_config(args.model_dir)
        pruned_layers = pruned_layers or cfg["num_hidden_layers"]
        original_layers = original_layers or pruned_layers

    input_lens = [int(x) for x in args.input_lens.split(",")]
    cfg = load_config(args.model_dir)

    print(f"Model dir: {args.model_dir}")
    if meta:
        print(f"Meta: original={meta.get('original_model', '?')}")
    print(f"  hidden={cfg['hidden_size']}, vocab={cfg['vocab_size']}")
    print(f"  layers: {pruned_layers} pruned / {original_layers} original, TP={tp_size}")
    print()

    # a) LM head
    if tp_size > 1:
        print("=== LM head (torch.mm) ===")
        lm = measure_lm_head(cfg["hidden_size"], cfg["vocab_size"], tp_size, input_lens)
        print(f"  Decode delta: {lm['decode_delta_ms']:.3f} ms")
        for il, d in lm["prefill_deltas_ms"].items():
            print(f"  Prefill delta (input={il}): {d:.3f} ms")
    else:
        print("=== LM head: TP=1, skip ===")
        lm = {"decode_delta_ms": 0, "prefill_deltas_ms": {str(il): 0 for il in input_lens}}
    print()

    # b) Encoder block
    enc = None
    if pruned_layers < original_layers:
        print("=== Encoder block (differential) ===")
        enc = measure_encoder_diff(args.model_dir, pruned_layers,
                                   input_lens[0], args.output_len, args.gpu_mem)
        removed = original_layers - pruned_layers
        print(f"  Per-layer TPOT: {enc['per_layer_tpot_ms']:.4f} ms")
        print(f"  Per-layer TTFT: {enc['per_layer_ttft_ms']:.4f} ms")
        print(f"  Compensation ({removed} layers):")
        print(f"    TPOT: +{removed * enc['per_layer_tpot_ms']:.3f} ms")
        print(f"    TTFT: +{removed * enc['per_layer_ttft_ms']:.3f} ms")
    else:
        print("=== Encoder block: no pruning, skip ===")
    print()

    # c) Apply to bench results if provided
    if args.bench_results:
        with open(args.bench_results) as f:
            bench = json.load(f)

        print("=== Applied Compensation ===")
        print(f"{'input':>8} {'raw_ttft':>10} {'comp_ttft':>10} "
              f"{'raw_tpot':>10} {'comp_tpot':>10}")
        print("-" * 55)

        for r in bench.get("results", []):
            if "error" in r:
                continue
            il = r["input_len"]
            raw_ttft = r["ttft_median_ms"]
            raw_tpot = r["tpot_median_ms"]

            pf_delta = lm["prefill_deltas_ms"].get(str(il), lm["decode_delta_ms"])
            comp_tpot = raw_tpot - lm["decode_delta_ms"]
            comp_ttft = raw_ttft - pf_delta

            if enc and pruned_layers < original_layers:
                removed = original_layers - pruned_layers
                comp_tpot += removed * enc["per_layer_tpot_ms"]
                comp_ttft += removed * enc["per_layer_ttft_ms"]

            r["comp_ttft_ms"] = round(comp_ttft, 3)
            r["comp_tpot_ms"] = round(comp_tpot, 3)
            print(f"{il:>8} {raw_ttft:>10.2f} {comp_ttft:>10.2f} "
                  f"{raw_tpot:>10.3f} {comp_tpot:>10.3f}")

    # Save
    output = {
        "model_dir": args.model_dir,
        "tp_size": tp_size,
        "pruned_layers": pruned_layers,
        "original_layers": original_layers,
        "lm_head": lm,
        "encoder_block": enc,
    }
    if args.bench_results:
        output["compensated_results"] = bench.get("results", [])

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
