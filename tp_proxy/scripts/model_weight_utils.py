#!/usr/bin/env python3
"""
Unified model weight / bandwidth calculation for decode step.

Single source of truth for:
  - Active weight bytes per decode step per GPU
  - BW floor (theoretical min TPOT)
  - INT4 projected weight bytes
  - BW utilization

Used by: compensate_scenarios, generate_report, analyze_decode_bw

Usage:
    from model_weight_utils import decode_weight_bytes, print_decode_bw

    # From model config.json
    info = decode_weight_bytes("/path/to/config.json", tp=2)
    print_decode_bw(info, comp_tpot_ms=6.88, peak_bw=680)

    # From model name shortcut
    info = decode_weight_bytes("qwen3-30b-a3b", tp=2)
"""

import json
import os


# Known model configs (shortcut names)
KNOWN_MODELS = {
    "qwen3-30b-a3b": {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_hidden_layers": 48,
        "intermediate_size": 6144,             # dense FFN (unused in MoE layers)
        "shared_expert_intermediate_size": 0,  # NO shared expert
        "moe_intermediate_size": 768,          # per routed expert
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "vocab_size": 151936,
        "attn_output_gate": False,
        "layer_types": [],
        "quant_bits": 16,
        "quant_bytes_per_param": 2,            # BF16
    },
    "qwen3-30b-a3b-gptq-int4": {
        "hidden_size": 2048,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_hidden_layers": 48,
        "intermediate_size": 6144,
        "shared_expert_intermediate_size": 0,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "vocab_size": 151936,
        "attn_output_gate": False,
        "layer_types": [],
        "quant_bits": 4,
        "quant_bytes_per_param": 0.5,          # GPTQ-INT4
    },
}


def load_model_config(path_or_name: str) -> dict:
    """Load model config from path or known name."""
    if path_or_name in KNOWN_MODELS:
        return dict(KNOWN_MODELS[path_or_name])

    # Try as directory path
    config_path = path_or_name
    if os.path.isdir(path_or_name):
        config_path = os.path.join(path_or_name, "config.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        return {
            "hidden_size": tc["hidden_size"],
            "num_attention_heads": tc["num_attention_heads"],
            "num_key_value_heads": tc.get("num_key_value_heads", tc["num_attention_heads"]),
            "head_dim": tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
            "num_hidden_layers": tc["num_hidden_layers"],
            "intermediate_size": tc.get("intermediate_size", 0),
            "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", 0),
            "moe_intermediate_size": tc.get("moe_intermediate_size", 0),
            "num_experts": tc.get("num_local_experts", tc.get("num_experts", 0)),
            "num_experts_per_tok": tc.get("num_experts_per_tok", 0),
            "vocab_size": tc["vocab_size"],
            "quant_bytes_per_param": 2,  # default BF16, caller can override
        }

    raise ValueError(f"Cannot load config from: {path_or_name}")


def decode_weight_bytes(path_or_name: str, tp: int = 1,
                        quant_bpp: float = None,
                        num_layers: int = None) -> dict:
    """Compute per-GPU weight bytes read per decode step.

    For MoE: only top-k active experts counted (weight READ, not storage).
    TP: col/row split attention + experts, replicate lm_head (or /tp if split).

    Args:
        path_or_name: model config path, directory, or known model name
        tp: tensor parallel size
        quant_bpp: override bytes_per_param (e.g., 0.5 for INT4, 2 for BF16)
        num_layers: override num_hidden_layers (e.g., for full model after prune)

    Returns dict with:
        attn_bytes, shared_bytes, moe_bytes, lm_head_bytes, router_bytes,
        total_bytes, total_gb, bf16_gb, int4_gb, int4_scale
    """
    cfg = load_model_config(path_or_name)

    H = cfg["hidden_size"]
    L = num_layers or cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    q_heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    qd = q_heads * hd
    kvd = kv_heads * hd
    bpp = quant_bpp or cfg.get("quant_bytes_per_param", 2)

    is_moe = cfg.get("num_experts", 0) > 0
    moe_ffn = cfg.get("moe_intermediate_size", 0)
    shared_ffn = cfg.get("shared_expert_intermediate_size", 0)  # 0 if no shared expert
    top_k = cfg.get("num_experts_per_tok", 0)
    n_experts = cfg.get("num_experts", 0)

    # Attention per layer: Q + K + V + O, col/row split by TP
    # Q: (H, qd) col split → per rank reads H × (qd/tp)
    # K: (H, kvd) col split → per rank reads H × max(kvd/tp, hd)
    # V: same as K
    # O: (qd, H) row split → per rank reads (qd/tp) × H
    kv_per_rank = max(kv_heads // tp, 1) * hd
    attn_params_per_layer = (H * (qd // tp)           # Q
                             + H * kv_per_rank         # K
                             + H * kv_per_rank         # V
                             + (qd // tp) * H)         # O

    # Shared expert or dense FFN
    if shared_ffn > 0:
        # MoE shared expert
        shared_params_per_layer = 3 * H * (shared_ffn // tp)
    elif not is_moe:
        # Dense model: use intermediate_size as FFN
        dense_ffn = cfg.get("intermediate_size", H * 4)
        shared_params_per_layer = 3 * H * (dense_ffn // tp)
    else:
        # MoE without shared expert
        shared_params_per_layer = 0

    # MoE routed experts: top-k active, each col/row split by TP
    if is_moe and moe_ffn > 0:
        expert_params = 3 * H * (moe_ffn // tp)
        moe_params_per_layer = top_k * expert_params
    else:
        moe_params_per_layer = 0

    # Per-layer total
    layer_params = attn_params_per_layer + shared_params_per_layer + moe_params_per_layer

    # lm_head: vocab × hidden, TP-split (each GPU reads vocab/tp × hidden)
    lm_head_params = V * H // tp

    # Router: hidden × num_experts per layer, replicated
    router_params = H * n_experts * L if is_moe else 0

    # Total bytes
    attn_bytes = attn_params_per_layer * L * bpp
    shared_bytes = shared_params_per_layer * L * bpp
    moe_bytes = moe_params_per_layer * L * bpp
    lm_head_bytes = lm_head_params * 2  # always BF16
    router_bytes = router_params * 2     # always BF16

    total_bytes = attn_bytes + shared_bytes + moe_bytes + lm_head_bytes + router_bytes

    # Also compute BF16 and INT4 projections for comparison
    layer_bf16 = layer_params * L * 2
    layer_int4 = layer_params * L * 0.5
    other_bf16 = lm_head_bytes + router_bytes  # always BF16

    return {
        "attn_bytes": attn_bytes,
        "shared_bytes": shared_bytes,
        "moe_bytes": moe_bytes,
        "lm_head_bytes": lm_head_bytes,
        "router_bytes": router_bytes,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / 1e9,
        # BF16 / INT4 projections
        "bf16_gb": (layer_bf16 + other_bf16) / 1e9,
        "int4_gb": (layer_int4 + other_bf16) / 1e9,
        "int4_scale": (layer_int4 + other_bf16) / (layer_bf16 + other_bf16),
        # Breakdown
        "layers": L,
        "tp": tp,
        "bpp": bpp,
        "is_moe": is_moe,
        "layer_params": layer_params,
        "lm_head_params": lm_head_params,
    }


def print_decode_bw(info: dict, comp_tpot_ms: float = 0, peak_bw: float = 680):
    """Print decode BW analysis."""
    print(f"  Weight/GPU ({info['layers']}L, TP={info['tp']}, {info['bpp']}B/param):")
    print(f"    attn:    {info['attn_bytes']/1e9:.3f}GB")
    print(f"    shared:  {info['shared_bytes']/1e9:.3f}GB")
    print(f"    moe:     {info['moe_bytes']/1e9:.3f}GB")
    print(f"    lm_head: {info['lm_head_bytes']/1e9:.3f}GB (BF16)")
    print(f"    router:  {info['router_bytes']/1e6:.1f}MB")
    print(f"    total:   {info['total_gb']:.3f}GB")

    if peak_bw > 0:
        bw_floor = info['total_gb'] / peak_bw * 1000
        print(f"  BW floor:  {bw_floor:.2f}ms @ {peak_bw}GB/s")
        if comp_tpot_ms > 0:
            bw_util = bw_floor / comp_tpot_ms * 100
            print(f"  BW util:   {bw_util:.0f}% (comp_TPOT={comp_tpot_ms:.2f}ms)")

    # INT4 projection
    print(f"  INT4 projection:")
    print(f"    BF16: {info['bf16_gb']:.3f}GB → INT4: {info['int4_gb']:.3f}GB (scale={info['int4_scale']:.2f}x)")
    if comp_tpot_ms > 0:
        int4_tpot = comp_tpot_ms * info['int4_scale']
        int4_floor = info['int4_gb'] / peak_bw * 1000
        print(f"    est_TPOT: {int4_tpot:.2f}ms  BW_floor: {int4_floor:.2f}ms")


def rescale_metrics(metrics: dict, info: dict,
                    src_flops: float = 0, tgt_flops: float = 0,
                    src_bw: float = 0, tgt_bw: float = 0) -> dict:
    """Rescale TTFT/TPOT from source platform to target platform.

    TTFT (prefill, compute-bound): scales with FLOPS ratio
    TPOT (decode, BW-bound): scales with weight_bytes / BW

    Args:
        metrics: dict with ttft, tpot, tps, total, output_tokens
        info: from decode_weight_bytes() (for INT4 projection)
        src_flops: source platform peak TFLOPS (0 = no scale)
        tgt_flops: target platform peak TFLOPS
        src_bw: source platform peak BW GB/s (0 = no scale)
        tgt_bw: target platform peak BW GB/s

    Returns: new metrics dict with rescaled values
    """
    if not metrics:
        return None

    ttft = metrics["ttft"]
    tpot = metrics["tpot"]
    output_tokens = metrics.get("output_tokens", 64)

    # TTFT: compute-bound → scale with FLOPS ratio
    if src_flops > 0 and tgt_flops > 0:
        ttft = ttft * (src_flops / tgt_flops)

    # TPOT: BW-bound → scale with BW ratio
    if src_bw > 0 and tgt_bw > 0:
        tpot = tpot * (src_bw / tgt_bw)

    tps = 1000 / tpot if tpot > 0 else 0
    total = ttft + (output_tokens - 1) * tpot

    return {
        "ttft": ttft,
        "tpot": tpot,
        "tps": tps,
        "total": total,
        "output_tokens": output_tokens,
    }


def rescale_int4(metrics: dict, info: dict,
                 src_flops: float = 0, tgt_flops: float = 0,
                 tgt_bw: float = 0, src_bw: float = 0,
                 kernel_breakdown: dict = None) -> dict:
    """Rescale to INT4 on target platform using kernel-level breakdown.

    TTFT: scale with FLOPS ratio (compute-bound)
    TPOT: from kernel breakdown:
      marlin + moe → scale by BW ratio × (int4_bpp / bf16_bpp = 0.25)
      flash_attn + other → scale by BW ratio only (precision unchanged)

    Args:
        metrics: dict with ttft, tpot
        info: from decode_weight_bytes()
        kernel_breakdown: dict with marlin_ms, moe_ms, fa_ms, other_ms
            (from compensated JSON tail.kernel_breakdown)
        src/tgt flops/bw: platform params
    """
    if not metrics:
        return None
    if not kernel_breakdown:
        raise ValueError("kernel_breakdown required for INT4 rescale. "
                         "Re-run compensate to generate trace kernel breakdown.")

    ttft = metrics["ttft"]
    tpot = metrics["tpot"]  # comp_tpot (already layer-scaled to full model)
    output_tokens = metrics.get("output_tokens", 64)

    # TTFT: compute-bound
    if src_flops > 0 and tgt_flops > 0:
        ttft = ttft * (src_flops / tgt_flops)

    # Use kernel_breakdown RATIO (not absolute time, since it's from pruned model)
    # gemm_int4_frac: fraction of decode step that's INT4-quantizable GEMM
    # Rest (lm_head + fa + other): stays BF16 or non-weight
    gemm_frac = kernel_breakdown.get("gemm_int4_frac", 0)
    if gemm_frac <= 0:
        # Compute from times if frac not stored
        gemm = kernel_breakdown.get("gemm_int4_ms", kernel_breakdown.get("marlin_ms", 0) + kernel_breakdown.get("moe_ms", 0))
        lmh = kernel_breakdown.get("lm_head_ms", 0)
        fa = kernel_breakdown.get("fa_ms", 0)
        other = kernel_breakdown.get("other_ms", 0)
        total_k = gemm + lmh + fa + other
        gemm_frac = gemm / total_k if total_k > 0 else 0.7

    # TPOT: apply ratio to comp_tpot (full model)
    # gemm_int4: BW ratio × 0.25 (BF16→INT4 weight reduction)
    # rest (lm_head BF16 + FA + other): BW ratio × 1.0
    bw_ratio = (src_bw / tgt_bw) if (src_bw > 0 and tgt_bw > 0) else 1.0

    tpot_int4 = tpot * bw_ratio * (gemm_frac * 0.25 + (1 - gemm_frac))

    # Clamp to BW floor
    if tgt_bw > 0:
        bw_floor = info["int4_gb"] / tgt_bw * 1000
        tpot_int4 = max(tpot_int4, bw_floor)

    tps = 1000 / tpot_int4 if tpot_int4 > 0 else 0
    total = ttft + (output_tokens - 1) * tpot_int4

    return {
        "ttft": ttft,
        "tpot": tpot_int4,
        "tps": tps,
        "total": total,
        "output_tokens": output_tokens,
    }


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3-30b-a3b"
    tp = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    info = decode_weight_bytes(model, tp=tp)
    print_decode_bw(info, comp_tpot_ms=6.88, peak_bw=680)
