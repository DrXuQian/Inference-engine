#!/usr/bin/env python3
"""
Calculate decoding bandwidth utilization.

For each decode step (batch=1), the GPU reads active weights once.
MoE models only activate a subset of experts per token.

  bandwidth_util = active_weight_bytes / TPOT_seconds / peak_bandwidth

Usage:
    # From bench results
    python bandwidth_util.py \
        --model-dir /path/to/model \
        --tpot-ms 4.75 \
        --bandwidth-gb-s 900

    # From bench JSON
    python bandwidth_util.py \
        --model-dir /path/to/model \
        --bench-results bench.json \
        --bandwidth-gb-s 900

    # TP=2 (weights are split)
    python bandwidth_util.py \
        --model-dir /path/to/model \
        --tpot-ms 4.88 \
        --bandwidth-gb-s 900 \
        --tp-size 2
"""

import argparse
import json
import os


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    qc = cfg.get("quantization_config", {})
    return {
        "hidden_size": tc.get("hidden_size", 2048),
        "head_dim": tc.get("head_dim", 256),
        "num_attention_heads": tc.get("num_attention_heads", 16),
        "num_key_value_heads": tc.get("num_key_value_heads", 2),
        "num_hidden_layers": tc.get("num_hidden_layers", 40),
        "vocab_size": tc.get("vocab_size", 248320),
        "num_experts": tc.get("num_experts", 0),
        "num_experts_per_tok": tc.get("num_experts_per_tok", 0),
        "moe_intermediate_size": tc.get("moe_intermediate_size", 0),
        "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", 0),
        "intermediate_size": tc.get("intermediate_size", 0),  # dense FFN
        "attn_output_gate": tc.get("attn_output_gate", False),
        # Linear attention params
        "linear_num_key_heads": tc.get("linear_num_key_heads", 0),
        "linear_num_value_heads": tc.get("linear_num_value_heads", 0),
        "linear_key_head_dim": tc.get("linear_key_head_dim", 0),
        "linear_value_head_dim": tc.get("linear_value_head_dim", 0),
        "layer_types": tc.get("layer_types", []),
        "full_attention_interval": tc.get("full_attention_interval", 0),
        # Quantization
        "quant_bits": qc.get("bits", 16),
        "quant_method": qc.get("quant_method", ""),
        # Dynamic: which layers are NOT quantized
        "quant_dynamic": qc.get("dynamic", {}),
    }


def bytes_per_param(quant_bits: int, is_quantized: bool) -> float:
    """Bytes per parameter based on quantization."""
    if is_quantized:
        # GPTQ: qweight + scales + qzeros ≈ bits/8 + overhead
        # For int4: 0.5 bytes for weight + ~0.03 bytes for scales/zeros
        return quant_bits / 8 + 0.05
    else:
        return 2  # bf16


def compute_active_weights(cfg: dict, tp_size: int = 1) -> dict:
    """Compute active weight bytes read per decode step."""
    H = cfg["hidden_size"]
    N = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]

    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    has_gate = cfg["attn_output_gate"]

    n_experts = cfg["num_experts"]
    n_active = cfg["num_experts_per_tok"]
    moe_inter = cfg["moe_intermediate_size"]
    shared_inter = cfg["shared_expert_intermediate_size"]
    dense_inter = cfg["intermediate_size"]

    quant_bits = cfg["quant_bits"]
    quant_dynamic = cfg["quant_dynamic"]

    # Check what's NOT quantized (from dynamic config)
    # Pattern: "-:.*attn.*" means attn is NOT quantized
    attn_quantized = True
    expert_quantized = True
    shared_quantized = True
    for pattern, _ in quant_dynamic.items():
        pl = pattern.lower()
        if "attn" in pl and pl.startswith("-:"):
            attn_quantized = False
        if "shared_expert" in pl and pl.startswith("-:"):
            shared_quantized = False

    bpp_attn = bytes_per_param(quant_bits, attn_quantized)
    bpp_expert = bytes_per_param(quant_bits, expert_quantized)
    bpp_shared = bytes_per_param(quant_bits, shared_quantized)
    bpp_embed = 2  # always bf16

    # Determine layer types
    layer_types = cfg["layer_types"]
    if not layer_types:
        layer_types = ["full_attention"] * N

    n_full_attn = sum(1 for lt in layer_types[:N] if lt == "full_attention")
    n_linear_attn = N - n_full_attn

    # === Per full_attention layer ===
    q_dim = n_heads * head_dim * (2 if has_gate else 1)
    kv_dim = n_kv * head_dim
    o_dim = n_heads * head_dim
    attn_params = q_dim * H + kv_dim * H + kv_dim * H + H * o_dim  # q,k,v,o
    attn_bytes_full = attn_params * bpp_attn / tp_size

    # === Per linear_attention layer ===
    if cfg["linear_num_key_heads"] > 0:
        lin_k = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        lin_v = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
        lin_qkv = (lin_k + lin_k + lin_v) * H  # in_proj_qkv
        lin_z = lin_v * H  # in_proj_z
        lin_a = cfg["linear_num_value_heads"] * H  # in_proj_a
        lin_b = cfg["linear_num_value_heads"] * H  # in_proj_b
        lin_o = H * lin_v  # out_proj
        lin_conv = (lin_k + lin_k + lin_v) * 4  # conv1d kernel_size=4
        lin_other = cfg["linear_num_value_heads"] * 2  # A_log + dt_bias
        attn_params_linear = lin_qkv + lin_z + lin_a + lin_b + lin_o + lin_conv + lin_other
    else:
        attn_params_linear = attn_params  # same as full if no linear config
    attn_bytes_linear = attn_params_linear * bpp_attn / tp_size

    # === MoE per layer (only active experts loaded) ===
    if n_experts > 0 and n_active > 0:
        expert_params = n_active * 3 * H * moe_inter  # gate+up+down per active expert
        router_params = n_experts * H  # MoE gate/router (always loaded, bf16)
        shared_params = 3 * H * shared_inter if shared_inter > 0 else 0
        shared_gate_params = H if shared_inter > 0 else 0

        moe_bytes = (expert_params * bpp_expert / tp_size +
                     router_params * 2 +  # router always bf16
                     shared_params * bpp_shared / tp_size +
                     shared_gate_params * 2)
    else:
        # Dense FFN
        ffn_params = 3 * H * dense_inter  # gate+up+down
        moe_bytes = ffn_params * bpp_attn / tp_size

    # === Norms per layer ===
    norm_bytes = 2 * H * 2  # 2 norms × hidden × bf16

    # === Per layer total ===
    per_layer_full = attn_bytes_full + moe_bytes + norm_bytes
    per_layer_linear = attn_bytes_linear + moe_bytes + norm_bytes

    total_layers = n_full_attn * per_layer_full + n_linear_attn * per_layer_linear

    # === Base (per decode step) ===
    lm_head_bytes = V * H * bpp_embed  # always bf16, NOT split by TP (replicated)
    final_norm_bytes = H * 2

    total_active = total_layers + lm_head_bytes + final_norm_bytes

    return {
        "per_layer_full_attn_MB": round(per_layer_full / 1e6, 2),
        "per_layer_linear_attn_MB": round(per_layer_linear / 1e6, 2),
        "n_full_attn_layers": n_full_attn,
        "n_linear_attn_layers": n_linear_attn,
        "total_layers_MB": round(total_layers / 1e6, 2),
        "lm_head_MB": round(lm_head_bytes / 1e6, 2),
        "total_active_MB": round(total_active / 1e6, 2),
        "total_active_bytes": total_active,
        "n_active_experts": n_active,
        "n_total_experts": n_experts,
        "attn_quantized": attn_quantized,
        "expert_quantized": expert_quantized,
    }


def main():
    ap = argparse.ArgumentParser(description="Decode bandwidth utilization")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tpot-ms", type=float, default=None,
                    help="TPOT in ms (or read from --bench-results)")
    ap.add_argument("--bench-results", default=None,
                    help="bench.json to read TPOT from")
    ap.add_argument("--bandwidth-gb-s", type=float, required=True,
                    help="Peak memory bandwidth in GB/s (e.g., 900 for HBM3)")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    cfg = load_config(args.model_dir)
    active = compute_active_weights(cfg, args.tp_size)

    # Get TPOT
    tpots = {}
    if args.tpot_ms:
        tpots["cli"] = args.tpot_ms
    if args.bench_results:
        with open(args.bench_results) as f:
            bench = json.load(f)
        for r in bench.get("results", [bench]):
            if "tpot_median_ms" in r:
                tpots[f"input={r.get('input_len', '?')}"] = r["tpot_median_ms"]

    if not tpots:
        print("ERROR: provide --tpot-ms or --bench-results")
        return

    print(f"Model: {args.model_dir}")
    print(f"TP={args.tp_size}, bandwidth={args.bandwidth_gb_s} GB/s")
    print()
    print(f"=== Active weights per decode step ===")
    print(f"  Full attention layers: {active['n_full_attn_layers']} × {active['per_layer_full_attn_MB']:.1f} MB")
    print(f"  Linear attention layers: {active['n_linear_attn_layers']} × {active['per_layer_linear_attn_MB']:.1f} MB")
    print(f"  Total layers: {active['total_layers_MB']:.1f} MB")
    print(f"  LM head: {active['lm_head_MB']:.1f} MB")
    print(f"  Total active: {active['total_active_MB']:.1f} MB")
    print(f"  MoE: {active['n_active_experts']}/{active['n_total_experts']} experts active")
    print(f"  Attn quantized: {active['attn_quantized']}, Expert quantized: {active['expert_quantized']}")
    print()

    print(f"=== Bandwidth utilization ===")
    print(f"{'config':>20} {'TPOT':>8} {'BW used':>10} {'util':>8}")
    print("-" * 52)

    results = []
    for label, tpot in tpots.items():
        tpot_s = tpot / 1000
        bw_used = active["total_active_bytes"] / tpot_s / 1e9  # GB/s
        util = bw_used / args.bandwidth_gb_s * 100

        print(f"{label:>20} {tpot:>7.2f}ms {bw_used:>9.1f}GB/s {util:>7.1f}%")
        results.append({
            "label": label, "tpot_ms": tpot,
            "bw_used_gb_s": round(bw_used, 1),
            "util_pct": round(util, 1),
        })

    if args.output_json:
        output = {"active_weights": active, "bandwidth_gb_s": args.bandwidth_gb_s,
                  "tp_size": args.tp_size, "results": results}
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
