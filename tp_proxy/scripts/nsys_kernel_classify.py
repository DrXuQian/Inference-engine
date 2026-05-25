#!/usr/bin/env python3
"""
Classify nsys CUDA kernels into encoder-layer vs non-layer, compute
per-layer time and extrapolate to the full model.

Uses instance count to separate:
  - Init-only kernels: appear a fixed number of times regardless of num_prompts
  - Per-request kernels: scale with num_prompts → these are the inference kernels

Among per-request kernels, classify by name patterns:
  - Encoder-layer: marlin_moe, triton fused, attention, MoE routing, etc.
  - Non-layer: embedding, lm_head, sampling (topk)

Usage:
    python nsys_kernel_classify.py --nsys-rep /tmp/nsys_serve_profile.nsys-rep \\
        --num-layers 10 --original-layers 40
"""

import argparse
import csv
import io
import subprocess
import sys


def load_kernels(nsys_rep: str) -> list[dict]:
    """Load kernel summary from nsys report."""
    cmd = ["nsys", "stats", "-r", "cuda_gpu_kern_sum",
           "--format", "csv", "--force-export=true", nsys_rep]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    all_out = result.stdout + "\n" + result.stderr
    lines = all_out.split("\n")
    csv_start = None
    for i, line in enumerate(lines):
        if "Time (%)" in line and "Total Time" in line:
            csv_start = i
            break
    if csv_start is None:
        print("ERROR: no kernel summary CSV found")
        print(all_out[:500])
        sys.exit(1)

    reader = csv.DictReader(io.StringIO("\n".join(lines[csv_start:])))
    kernels = []
    for row in reader:
        try:
            kernels.append({
                "name": row["Name"].strip().strip('"'),
                "total_ns": int(float(row["Total Time (ns)"])),
                "instances": int(float(row["Instances"])),
                "avg_ns": int(float(row.get("Avg (ns)", 0))),
            })
        except (ValueError, KeyError):
            continue
    return kernels


# Kernels that are clearly per-encoder-layer (scale with layers)
ENCODER_PATTERNS = [
    "marlin_moe",           # fused GPTQ MoE expert GEMM
    "topkgating",           # MoE expert routing
    "moe_align_block",      # MoE token alignment
    "count_and_sort_expert", # MoE expert counting
    "act_and_mul",          # SiLU activation in FFN
    "sigmoid_kernel",       # gating
    "flash_fwd_splitkv",    # decode attention (paged/split-kv)
    "reshape_and_cache",    # KV cache update
    "fused_recurrent",      # Mamba/SSM recurrent
    "causal_conv1d",        # Mamba conv
    "chunk_gated_delta",    # GDN attention
    "chunk_fwd_kernel",     # chunked attention
    "chunk_scaled_dot",     # chunked attention
    "recompute_w_u",        # SSM recompute
    "fused_gdn",            # GDN fused
    "l2norm",               # L2 norm in attention
    "merge_16x16",          # merge kernel
    "layer_norm_fwd",       # per-layer norm
    "mrope",                # rotary embedding
    "triton_red_fused",     # triton fused reduction (norm+residual)
    "triton_per_fused",     # triton fused per-element
    "triton_poi_fused",     # triton fused pointwise
    "moe_forward_shared",   # shared expert forward
]

# Kernels that are NOT per-layer (init, sampling, embedding, etc.)
NON_LAYER_PATTERNS = [
    "flash_fwd_kernel.*96.*128",  # visual encoder (27 blocks, large dim)
    "gptq_marlin_repack",         # weight repacking (init only)
    "topk_topp_kernel",           # sampling (output only)
    "GeluCUDA",                   # visual encoder activation
]


def classify(kernels: list[dict], num_layers: int):
    """Classify kernels and compute per-layer time."""
    import re

    encoder_ns = 0
    non_layer_ns = 0
    encoder_list = []
    non_layer_list = []

    for k in kernels:
        name_lower = k["name"].lower()

        # Check non-layer patterns first (more specific)
        is_non_layer = any(re.search(p.lower(), name_lower) for p in NON_LAYER_PATTERNS)
        if is_non_layer:
            non_layer_ns += k["total_ns"]
            non_layer_list.append(k)
            continue

        # Check encoder patterns
        is_encoder = any(p.lower() in name_lower for p in ENCODER_PATTERNS)
        if is_encoder:
            encoder_ns += k["total_ns"]
            encoder_list.append(k)
            continue

        # Heuristic: high instance count relative to layers → likely per-layer
        # Low instance count → likely init
        if k["instances"] >= num_layers * 2:
            encoder_ns += k["total_ns"]
            encoder_list.append(k)
        else:
            non_layer_ns += k["total_ns"]
            non_layer_list.append(k)

    return encoder_ns, non_layer_ns, encoder_list, non_layer_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsys-rep", required=True, help="Path to .nsys-rep file")
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--original-layers", type=int, default=40)
    ap.add_argument("--wall-clock-ms", type=float, default=None,
                    help="Per-request wall clock latency (from bench serve output)")
    args = ap.parse_args()

    kernels = load_kernels(args.nsys_rep)
    N = args.num_layers

    encoder_ns, non_layer_ns, enc_list, nl_list = classify(kernels, N)
    total_ns = encoder_ns + non_layer_ns

    print(f"{'='*65}")
    print(f"Kernel Classification ({N} layers)")
    print(f"{'='*65}")
    print(f"  Total kernel time:   {total_ns/1e6:10.2f} ms")
    print(f"  Encoder (per-layer): {encoder_ns/1e6:10.2f} ms ({encoder_ns/total_ns*100:.1f}%)")
    print(f"  Non-layer (fixed):   {non_layer_ns/1e6:10.2f} ms ({non_layer_ns/total_ns*100:.1f}%)")
    print(f"  Per encoder block:   {encoder_ns/N/1e6:10.2f} ms (kernel time)")

    print(f"\nTop encoder kernels:")
    for k in sorted(enc_list, key=lambda x: -x["total_ns"])[:10]:
        print(f"  {k['total_ns']/1e6:8.2f} ms  ×{k['instances']:6d}  {k['name'][:65]}")

    print(f"\nTop non-layer kernels:")
    for k in sorted(nl_list, key=lambda x: -x["total_ns"])[:5]:
        print(f"  {k['total_ns']/1e6:8.2f} ms  ×{k['instances']:6d}  {k['name'][:65]}")

    # Extrapolate
    enc_frac = encoder_ns / total_ns
    orig = args.original_layers

    if args.wall_clock_ms:
        wall = args.wall_clock_ms
        wall_per_layer = wall * enc_frac / N
        wall_fixed = wall * (1 - enc_frac)
        extrap = wall_fixed + orig * wall_per_layer
        print(f"\n{'='*65}")
        print(f"Wall-clock extrapolation (enc_frac={enc_frac:.1%}):")
        print(f"  Per-layer wall: {wall_per_layer:.2f} ms")
        print(f"  Fixed wall:     {wall_fixed:.2f} ms")
        print(f"  Extrap {orig}L:   {wall_fixed:.2f} + {orig}×{wall_per_layer:.2f} = {extrap:.2f} ms")
    else:
        # Pure kernel-time extrapolation
        per_layer_ms = encoder_ns / N / 1e6
        fixed_ms = non_layer_ns / 1e6
        extrap = fixed_ms + orig * per_layer_ms
        print(f"\n{'='*65}")
        print(f"Kernel-time extrapolation:")
        print(f"  Per-layer: {per_layer_ms:.2f} ms")
        print(f"  Fixed:     {fixed_ms:.2f} ms")
        print(f"  Extrap {orig}L: {fixed_ms:.2f} + {orig}×{per_layer_ms:.2f} = {extrap:.2f} ms")
        print(f"  (Pass --wall-clock-ms for wall-clock based extrapolation)")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
