#!/usr/bin/env python3
"""
Print per-component weight breakdown for decode (batch=1, per step).

Shows exactly how many bytes each component reads per decode step,
accounting for quantization (GPTQ int4 vs bf16), TP split, MoE
active experts, and attention type (full vs linear).

Usage:
    python weight_breakdown.py --model-dir /path/to/model
    python weight_breakdown.py --model-dir /path/to/model --tp-size 2
    python weight_breakdown.py --model-dir /path/to/model --tp-size 2 --seq-len 100000
"""

import argparse
import json
import os
import sys


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    qc = cfg.get("quantization_config", {})

    layer_types = tc.get("layer_types", [])
    n_layers = tc.get("num_hidden_layers", 0)
    n_full = sum(1 for lt in layer_types[:n_layers] if lt == "full_attention") if layer_types else n_layers

    # Check what's NOT quantized
    quant_dynamic = qc.get("dynamic", {})
    attn_quantized = True
    shared_quantized = True
    for pattern in quant_dynamic:
        pl = pattern.lower()
        if pl.startswith("-:"):
            if "attn" in pl:
                attn_quantized = False
            if "shared_expert" in pl:
                shared_quantized = False

    return {
        "hidden_size": tc.get("hidden_size", 0),
        "num_hidden_layers": n_layers,
        "num_attention_heads": tc.get("num_attention_heads", 0),
        "num_key_value_heads": tc.get("num_key_value_heads", 0),
        "head_dim": tc.get("head_dim", 0),
        "vocab_size": tc.get("vocab_size", 0),
        "attn_output_gate": tc.get("attn_output_gate", False),
        # MoE
        "num_experts": tc.get("num_experts", 0),
        "num_experts_per_tok": tc.get("num_experts_per_tok", 0),
        "moe_intermediate_size": tc.get("moe_intermediate_size", 0),
        "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", 0),
        "intermediate_size": tc.get("intermediate_size", 0),
        # Linear attention
        "linear_num_key_heads": tc.get("linear_num_key_heads", 0),
        "linear_num_value_heads": tc.get("linear_num_value_heads", 0),
        "linear_key_head_dim": tc.get("linear_key_head_dim", 0),
        "linear_value_head_dim": tc.get("linear_value_head_dim", 0),
        # Layers
        "n_full_attn_layers": n_full,
        "n_lin_attn_layers": n_layers - n_full,
        "layer_types": layer_types[:n_layers],
        # Quantization
        "quant_bits": qc.get("bits", 16),
        "quant_method": qc.get("quant_method", "none"),
        "attn_quantized": attn_quantized,
        "shared_quantized": shared_quantized,
    }


def fmt_bytes(b):
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    if b >= 1e6:
        return f"{b/1e6:.2f} MB"
    if b >= 1e3:
        return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


def fmt_params(p):
    if p >= 1e9:
        return f"{p/1e9:.2f}B"
    if p >= 1e6:
        return f"{p/1e6:.2f}M"
    if p >= 1e3:
        return f"{p/1e3:.1f}K"
    return str(int(p))


def main():
    ap = argparse.ArgumentParser(description="Decode weight breakdown")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=0,
                    help="If set, also show KV cache size")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.model_dir)
    tp = args.tp_size
    H = cfg["hidden_size"]
    N = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    has_gate = cfg["attn_output_gate"]
    n_full = cfg["n_full_attn_layers"]
    n_lin = cfg["n_lin_attn_layers"]

    qbits = cfg["quant_bits"]
    bpp_q = qbits / 8  # int4 = 0.5
    bpp_f = 2  # bf16
    bpp_attn = bpp_q if cfg["attn_quantized"] else bpp_f
    bpp_shared = bpp_q if cfg["shared_quantized"] else bpp_f

    n_experts = cfg["num_experts"]
    n_active = cfg["num_experts_per_tok"]
    moe_inter = cfg["moe_intermediate_size"]
    shared_inter = cfg["shared_expert_intermediate_size"]
    dense_inter = cfg["intermediate_size"]

    lin_k_heads = cfg["linear_num_key_heads"]
    lin_v_heads = cfg["linear_num_value_heads"]
    lin_k_dim = cfg["linear_key_head_dim"]
    lin_v_dim = cfg["linear_value_head_dim"]
    lin_k = lin_k_heads * lin_k_dim
    lin_v = lin_v_heads * lin_v_dim

    print(f"Model: {args.model_dir}")
    print(f"  hidden={H}, layers={N} ({n_full} full + {n_lin} linear)")
    print(f"  heads={n_heads}, kv_heads={n_kv}, head_dim={head_dim}, gate={has_gate}")
    print(f"  vocab={V}")
    if n_experts > 0:
        print(f"  MoE: {n_experts} experts, {n_active} active, inter={moe_inter}")
        print(f"  shared_expert: inter={shared_inter}")
    else:
        print(f"  Dense FFN: inter={dense_inter}")
    if lin_k > 0:
        print(f"  Linear attn: qk_heads={lin_k_heads}, v_heads={lin_v_heads}, "
              f"k_dim={lin_k_dim}, v_dim={lin_v_dim}")
    print(f"  Quant: {cfg['quant_method']} {qbits}bit, "
          f"attn_quantized={cfg['attn_quantized']}, shared_quantized={cfg['shared_quantized']}")
    print(f"  TP={tp}, batch={args.batch_size}")
    print()

    rows = []  # (name, params, bytes_per_rank, precision, count)

    def add(name, params, bpp, count=1):
        per_rank = params * bpp / tp
        rows.append((name, params, per_rank, bpp, count))

    # === Full attention layer ===
    q_dim = n_heads * head_dim * (2 if has_gate else 1)
    kv_dim = n_kv * head_dim
    o_dim = n_heads * head_dim

    add("full_attn.q_proj", H * q_dim, bpp_attn, n_full)
    add("full_attn.k_proj", H * kv_dim, bpp_attn, n_full)
    add("full_attn.v_proj", H * kv_dim, bpp_attn, n_full)
    add("full_attn.o_proj", o_dim * H, bpp_attn, n_full)

    # === Linear attention layer ===
    if lin_k > 0:
        add("lin_attn.in_proj_qkv", H * (lin_k * 2 + lin_v), bpp_attn, n_lin)
        add("lin_attn.in_proj_z", H * lin_v, bpp_attn, n_lin)
        add("lin_attn.in_proj_a", H * lin_v_heads, bpp_attn, n_lin)
        add("lin_attn.in_proj_b", H * lin_v_heads, bpp_attn, n_lin)
        add("lin_attn.out_proj", lin_v * H, bpp_attn, n_lin)
        add("lin_attn.conv1d", (lin_k * 2 + lin_v) * 4, bpp_f, n_lin)
        add("lin_attn.A_log+dt_bias", lin_v_heads * 2, bpp_f, n_lin)

    # === MoE / FFN ===
    if n_experts > 0 and n_active > 0:
        add(f"moe.active_experts(×{n_active}).gate_proj", n_active * H * moe_inter, bpp_q, N)
        add(f"moe.active_experts(×{n_active}).up_proj", n_active * H * moe_inter, bpp_q, N)
        add(f"moe.active_experts(×{n_active}).down_proj", n_active * moe_inter * H, bpp_q, N)
        add(f"moe.router", n_experts * H, bpp_f, N)
        if shared_inter > 0:
            add("moe.shared_expert.gate_proj", H * shared_inter, bpp_shared, N)
            add("moe.shared_expert.up_proj", H * shared_inter, bpp_shared, N)
            add("moe.shared_expert.down_proj", shared_inter * H, bpp_shared, N)
            add("moe.shared_expert_gate", H, bpp_f, N)
    else:
        add("ffn.gate_proj", H * dense_inter, bpp_q, N)
        add("ffn.up_proj", H * dense_inter, bpp_q, N)
        add("ffn.down_proj", dense_inter * H, bpp_q, N)

    # === Norms ===
    add("layer_norm (×2)", H * 2, bpp_f, N)

    # === LM head + final norm ===
    # LM head is replicated (not split by TP)
    rows.append(("lm_head", V * H, V * H * bpp_f, bpp_f, 1))  # not /tp
    add("final_norm", H, bpp_f, 1)

    # Print table
    print(f"{'Component':<45} {'Precision':>8} {'Params':>10} {'×Layers':>8} "
          f"{'Per Rank':>12} {'Total/Rank':>12}")
    print("=" * 105)

    total_bytes = 0
    total_params = 0
    section = ""
    for name, params, per_rank, bpp, count in rows:
        # Section headers
        cur_section = name.split(".")[0]
        if cur_section != section:
            if section:
                print()
            section = cur_section

        prec = f"int{qbits}" if bpp == bpp_q and bpp < 2 else "bf16"
        total_per_rank = per_rank * count
        total_bytes += total_per_rank
        total_params += params * count

        print(f"  {name:<43} {prec:>8} {fmt_params(params):>10} {'×'+str(count):>8} "
              f"{fmt_bytes(per_rank):>12} {fmt_bytes(total_per_rank):>12}")

    print()
    print(f"{'TOTAL WEIGHTS':<45} {'':>8} {fmt_params(total_params):>10} {'':>8} "
          f"{'':>12} {fmt_bytes(total_bytes):>12}")

    # === KV cache ===
    if args.seq_len > 0:
        S = args.seq_len
        B = args.batch_size
        kv_per_rank = n_kv if tp > n_kv else n_kv // tp

        print()
        print(f"--- KV Cache (seq_len={S}, batch={B}) ---")

        full_kv = n_full * 2 * kv_per_rank * head_dim * S * 2 * B
        print(f"  Full attn KV:  {n_full} layers × 2(K+V) × {kv_per_rank} heads × "
              f"{head_dim} dim × {S} seq × 2B × {B} batch = {fmt_bytes(full_kv)}")

        lin_state = 0
        if lin_k_heads > 0:
            lin_state = n_lin * lin_k_heads * lin_k_dim * lin_k_dim * 2 * B
            print(f"  Linear state:  {n_lin} layers × {lin_k_heads} heads × "
                  f"{lin_k_dim}² × 2B × {B} batch = {fmt_bytes(lin_state)}")

        total_kv = full_kv + lin_state
        print(f"  Total KV:      {fmt_bytes(total_kv)}")

        grand_total = total_bytes + total_kv
        print()
        print(f"  Weights + KV = {fmt_bytes(total_bytes)} + {fmt_bytes(total_kv)} "
              f"= {fmt_bytes(grand_total)}")

        if args.batch_size > 1:
            print(f"  (KV cache × {B} for batch={B}, weights shared)")


if __name__ == "__main__":
    main()
