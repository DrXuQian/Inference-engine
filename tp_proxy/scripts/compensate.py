#!/usr/bin/env python3
"""
Compensate single-card TP proxy results to estimate real TP performance.

Three compensation components:
  a) LM head: torch.mm full vs half vocab
  b) Communication: nccl-tests (NVIDIA) or pccl_tools (PPU)
  c) Encoder block: differential method with two layer counts

Usage:
    # NVIDIA
    python compensate.py \
        --bench-results bench_results.json \
        --model-dir /path/to/original \
        --pruned-layers 10 --original-layers 40 \
        --tp-size 2 --platform nvidia

    # PPU
    python compensate.py \
        --bench-results bench_results.json \
        --model-dir /path/to/original \
        --pruned-layers 10 --original-layers 40 \
        --tp-size 2 --platform ppu \
        --pccl-ar /usr/local/PPU_SDK/pccl_tools/all_reduce_perf \
        --pccl-ag /usr/local/PPU_SDK/pccl_tools/all_gather_perf
"""

import argparse
import json
import os
import re
import subprocess
import sys


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def load_model_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    return {
        "hidden_size": tc["hidden_size"],
        "vocab_size": tc["vocab_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc.get("num_attention_heads", 16),
        "num_key_value_heads": tc.get("num_key_value_heads", 2),
        "moe_intermediate_size": tc.get("moe_intermediate_size", 512),
        "num_experts": tc.get("num_experts", 256),
        "num_experts_per_tok": tc.get("num_experts_per_tok", 8),
    }


# ---------------------------------------------------------------------------
# a) LM head compensation
# ---------------------------------------------------------------------------

def measure_lm_head(hidden: int, vocab: int, tp_size: int,
                    input_lens: list[int], batch_decode: int = 1) -> dict:
    """Measure lm_head delta: full vocab vs split vocab."""
    import torch

    device = "cuda"
    dtype = torch.bfloat16
    vocab_full = vocab
    vocab_half = vocab // tp_size

    def bench_mm(M, K, N, warmup=50, iters=200):
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        for _ in range(warmup):
            torch.mm(A, B)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            torch.mm(A, B)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        times.sort()
        return times[len(times) // 2]

    # Decode delta
    full_dec = bench_mm(batch_decode, hidden, vocab_full)
    half_dec = bench_mm(batch_decode, hidden, vocab_half)
    decode_delta = full_dec - half_dec

    # Prefill deltas per input_len
    prefill_deltas = {}
    for il in input_lens:
        full_pre = bench_mm(il, hidden, vocab_full)
        half_pre = bench_mm(il, hidden, vocab_half)
        prefill_deltas[il] = full_pre - half_pre

    return {
        "decode_delta_ms": round(decode_delta, 4),
        "prefill_deltas_ms": {str(k): round(v, 4) for k, v in prefill_deltas.items()},
    }


# ---------------------------------------------------------------------------
# b) Communication compensation
# ---------------------------------------------------------------------------

def measure_comm_nvidia(hidden: int, vocab: int, num_layers: int,
                        tp_size: int) -> dict:
    """Measure NCCL all-reduce/all-gather latency using nccl_bench.py subprocess."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nccl_bench.py")
    env = os.environ.copy()
    env["TRITON_BACKENDS_IN_TREE"] = "1"

    result = subprocess.run(
        [sys.executable, script,
         "--hidden-size", str(hidden), "--num-layers", str(num_layers),
         "--world-size", str(tp_size)],
        env=env, capture_output=True, text=True, timeout=120,
    )

    # Parse output
    ar_us = ag_us = 0.0
    for line in result.stdout.split("\n"):
        if "4KB" in line and "B=1" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                try:
                    v = float(p)
                    if i == 1: ar_us = v  # AR column
                    if i == 2: ag_us = v  # AG column
                except ValueError:
                    continue

    # Fallback: parse any line with numbers
    if ar_us == 0:
        for line in result.stdout.split("\n"):
            if "4KB" in line:
                nums = [float(x) for x in line.split() if x.replace(".", "").isdigit()]
                if len(nums) >= 2:
                    ar_us, ag_us = nums[0], nums[1]
                    break

    n_ar = num_layers * 2
    n_ag = 2
    total_ms = (n_ar * ar_us + n_ag * ag_us) / 1000

    return {
        "ar_per_call_us": round(ar_us, 1),
        "ag_per_call_us": round(ag_us, 1),
        "n_ar_per_step": n_ar,
        "n_ag_per_step": n_ag,
        "total_per_step_ms": round(total_ms, 3),
    }


def measure_comm_ppu(hidden: int, vocab: int, num_layers: int,
                     tp_size: int, ar_tool: str, ag_tool: str) -> dict:
    """Measure communication using pccl_tools."""
    ar_size = hidden * 2  # bf16
    ag_size = (vocab // tp_size) * 2

    def run_pccl(tool, size, iters=200, warmup=50):
        cmd = [tool, "-b", str(size), "-e", str(size), "-f", "2",
               "-d", "bf16", "-o", "sum", "-n", str(iters), "-w", str(warmup),
               "-g", str(tp_size), "-c", "0", "-a", "1"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # nccl-tests / pccl_tools output format:
        #   size  count  type  redop  root  time  algbw  busbw  #wrong  time  algbw  busbw  #wrong
        # The "time" column (index 5 for out-of-place) is in microseconds
        for line in result.stdout.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("="):
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    # Column 5 is out-of-place time (us)
                    return float(parts[5])
                except (ValueError, IndexError):
                    pass
        return 0.0

    ar_us = run_pccl(ar_tool, ar_size)
    ag_us = run_pccl(ag_tool, ag_size)

    n_ar = num_layers * 2
    n_ag = 2
    total_ms = (n_ar * ar_us + n_ag * ag_us) / 1000

    return {
        "ar_per_call_us": round(ar_us, 1),
        "ag_per_call_us": round(ag_us, 1),
        "n_ar_per_step": n_ar,
        "n_ag_per_step": n_ag,
        "total_per_step_ms": round(total_ms, 3),
    }


# ---------------------------------------------------------------------------
# c) Encoder block compensation (differential)
# ---------------------------------------------------------------------------

def _get_attn_cycle(model_dir: str) -> int:
    """Get attention type cycle length from config (e.g., 4 for [lin,lin,lin,full])."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)

    # Try full_attention_interval first
    interval = tc.get("full_attention_interval")
    if interval:
        return interval

    # Fall back to layer_types pattern detection
    layer_types = tc.get("layer_types", [])
    if not layer_types:
        return 1  # no mixed attention, every layer is the same

    # Find the repeating cycle length
    for cycle in range(1, len(layer_types) + 1):
        pattern = layer_types[:cycle]
        if all(layer_types[i] == pattern[i % cycle] for i in range(len(layer_types))):
            return cycle
    return 1


def measure_encoder_block(model_dir: str, pruned_layers: int,
                          input_len: int, output_len: int,
                          gpu_mem: float) -> dict:
    """Differential method: bench at two layer counts to get per-layer time.

    Uses layer counts that are multiples of the attention cycle
    (e.g., multiples of 4 for [lin,lin,lin,full]) so that the
    linear:full attention ratio is consistent across measurements.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prune_script = os.path.join(script_dir, "prune_layers.py")
    bench_script = os.path.join(script_dir, "generate_bench.py")

    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="encoder_comp_")

    # Align to attention cycle so lin:full ratio is consistent
    cycle = _get_attn_cycle(model_dir)
    n_hi = (pruned_layers // cycle) * cycle
    if n_hi >= 2 * cycle:
        n_lo = max(n_hi // 2, cycle)
        n_lo = (n_lo // cycle) * cycle
        if n_lo == n_hi:
            n_lo = n_hi - cycle
    elif n_hi > 1:
        n_lo = max(n_hi // 2, 1)
    else:
        return None
    print(f"  Attention cycle: {cycle} (using {n_lo}L and {n_hi}L for differential)")

    results = {}
    for n in [n_lo, n_hi]:
        pruned_dir = os.path.join(tmp_dir, f"pruned_{n}L")
        # Prune
        subprocess.run([sys.executable, prune_script,
                        "--rank-dir", model_dir,
                        "--num-layers", str(n),
                        "--output-dir", pruned_dir],
                       capture_output=True, check=True, timeout=300)

        # Bench
        tmp_json = os.path.join(tmp_dir, f"bench_{n}L.json")
        env = os.environ.copy()
        env["TRITON_BACKENDS_IN_TREE"] = "1"
        subprocess.run([sys.executable, bench_script,
                        "--model", pruned_dir,
                        "--input-len", str(input_len),
                        "--output-len", str(output_len),
                        "--num-prompts", "5", "--num-warmup", "2",
                        "--gpu-mem", str(gpu_mem),
                        "--output-json", tmp_json],
                       env=env, capture_output=True, timeout=600)

        with open(tmp_json) as f:
            data = json.load(f)
        results[n] = data

        # Cleanup pruned model to save disk
        import shutil
        shutil.rmtree(pruned_dir, ignore_errors=True)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Differential
    tpot_hi = results[n_hi]["tpot_median_ms"]
    tpot_lo = results[n_lo]["tpot_median_ms"]
    ttft_hi = results[n_hi]["ttft_median_ms"]
    ttft_lo = results[n_lo]["ttft_median_ms"]

    per_layer_tpot = (tpot_hi - tpot_lo) / (n_hi - n_lo)
    per_layer_ttft = (ttft_hi - ttft_lo) / (n_hi - n_lo)

    return {
        "n_lo": n_lo,
        "n_hi": n_hi,
        "tpot_lo": tpot_lo,
        "tpot_hi": tpot_hi,
        "ttft_lo": ttft_lo,
        "ttft_hi": ttft_hi,
        "per_layer_tpot_ms": round(per_layer_tpot, 4),
        "per_layer_ttft_ms": round(per_layer_ttft, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Compensate TP proxy results")
    ap.add_argument("--bench-results", required=True, help="From auto_bench.py")
    ap.add_argument("--model-dir", required=True, help="Original (or split rank_0) model dir")
    ap.add_argument("--pruned-layers", type=int, required=True)
    ap.add_argument("--original-layers", type=int, required=True)
    ap.add_argument("--tp-size", type=int, default=2)
    ap.add_argument("--platform", choices=["nvidia", "ppu"], default="nvidia")
    ap.add_argument("--pccl-ar", default="/usr/local/PPU_SDK/pccl_tools/all_reduce_perf")
    ap.add_argument("--pccl-ag", default="/usr/local/PPU_SDK/pccl_tools/all_gather_perf")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--skip-encoder", action="store_true",
                    help="Skip encoder block compensation (no layer pruning)")
    ap.add_argument("--output-json", default="compensated_results.json")
    args = ap.parse_args()

    # Load bench results
    with open(args.bench_results) as f:
        bench = json.load(f)

    cfg = load_model_config(args.model_dir)
    hidden = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    input_lens = [r["input_len"] for r in bench["results"] if "error" not in r]

    print(f"Model: hidden={hidden}, vocab={vocab}, "
          f"layers={args.original_layers} (pruned={args.pruned_layers})")
    print(f"Platform: {args.platform}, TP={args.tp_size}")
    print()

    # a) LM head (only for TP > 1)
    if args.tp_size > 1:
        print("=== a) LM head compensation ===")
        lm = measure_lm_head(hidden, vocab, args.tp_size, input_lens)
        print(f"  Decode delta: {lm['decode_delta_ms']:.3f} ms")
        for il, d in lm["prefill_deltas_ms"].items():
            print(f"  Prefill delta (input={il}): {d:.3f} ms")
        print()
    else:
        print("=== a) LM head: TP=1, no compensation needed ===\n")
        lm = {"decode_delta_ms": 0, "prefill_deltas_ms": {}}

    # b) Communication (only for TP > 1)
    if args.tp_size > 1:
        print("=== b) Communication compensation ===")
        if args.platform == "nvidia":
            comm = measure_comm_nvidia(hidden, vocab, args.original_layers, args.tp_size)
        else:
            comm = measure_comm_ppu(hidden, vocab, args.original_layers, args.tp_size,
                                    args.pccl_ar, args.pccl_ag)
        print(f"  AR: {comm['ar_per_call_us']:.1f} us/call × {comm['n_ar_per_step']} = "
              f"{comm['n_ar_per_step'] * comm['ar_per_call_us'] / 1000:.3f} ms")
        print(f"  AG: {comm['ag_per_call_us']:.1f} us/call × {comm['n_ag_per_step']} = "
              f"{comm['n_ag_per_step'] * comm['ag_per_call_us'] / 1000:.3f} ms")
        print(f"  Total/step: {comm['total_per_step_ms']:.3f} ms")
        print()
    else:
        print("=== b) Communication: TP=1, no compensation needed ===\n")
        comm = {"total_per_step_ms": 0, "ar_per_call_us": 0, "ag_per_call_us": 0,
                "n_ar_per_step": 0, "n_ag_per_step": 0}

    # c) Encoder block
    enc = None
    if not args.skip_encoder and args.pruned_layers < args.original_layers:
        print("=== c) Encoder block compensation (differential) ===")
        # Use first input_len for encoder measurement
        enc = measure_encoder_block(
            args.model_dir, args.pruned_layers,
            input_lens[0], bench["output_len"], args.gpu_mem,
        )
        print(f"  {enc['n_lo']}L TPOT={enc['tpot_lo']:.3f}ms, "
              f"{enc['n_hi']}L TPOT={enc['tpot_hi']:.3f}ms")
        print(f"  Per-layer TPOT: {enc['per_layer_tpot_ms']:.4f} ms")
        print(f"  Per-layer TTFT: {enc['per_layer_ttft_ms']:.4f} ms")
        removed = args.original_layers - args.pruned_layers
        print(f"  TPOT compensation: {removed} × {enc['per_layer_tpot_ms']:.4f} = "
              f"{removed * enc['per_layer_tpot_ms']:.3f} ms")
        print()

    # d) Apply compensations
    print("=== Compensated Results ===")
    print(f"{'input':>8} {'raw_ttft':>10} {'comp_ttft':>10} "
          f"{'raw_tpot':>10} {'comp_tpot':>10} {'raw_total':>10} {'comp_total':>10}")
    print("-" * 75)

    compensated = []
    for r in bench["results"]:
        if "error" in r:
            compensated.append(r)
            continue

        il = r["input_len"]
        raw_ttft = r["ttft_median_ms"]
        raw_tpot = r["tpot_median_ms"]
        raw_total = r.get("total_median_ms", 0)
        output_len = r.get("output_tokens", bench.get("output_len", 64))

        # TPOT: -lm_head +comm +encoder_block_comp
        comp_tpot = raw_tpot - lm["decode_delta_ms"] + comm["total_per_step_ms"]
        # TTFT: -lm_head +~0(comm overlap) +encoder_block_comp
        pf_delta = lm["prefill_deltas_ms"].get(str(il), lm["decode_delta_ms"])
        comp_ttft = raw_ttft - pf_delta  # comm overlap ≈ 0

        if enc and args.pruned_layers < args.original_layers:
            removed = args.original_layers - args.pruned_layers
            comp_tpot += removed * enc["per_layer_tpot_ms"]
            comp_ttft += removed * enc["per_layer_ttft_ms"]

        comp_total = comp_ttft + (output_len - 1) * comp_tpot

        print(f"{il:>8} {raw_ttft:>10.2f} {comp_ttft:>10.2f} "
              f"{raw_tpot:>10.3f} {comp_tpot:>10.3f} {raw_total:>10.1f} {comp_total:>10.1f}")

        compensated.append({
            **r,
            "comp_ttft_ms": round(comp_ttft, 3),
            "comp_tpot_ms": round(comp_tpot, 3),
            "comp_total_ms": round(comp_total, 3),
        })

    # Save
    output = {
        "model_dir": args.model_dir,
        "tp_size": args.tp_size,
        "platform": args.platform,
        "pruned_layers": args.pruned_layers,
        "original_layers": args.original_layers,
        "lm_head": lm,
        "communication": comm,
        "encoder_block": enc,
        "results": compensated,
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
