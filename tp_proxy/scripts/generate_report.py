#!/usr/bin/env python3
"""
Generate performance report table from compensated_ppu.json results.

Usage:
    python generate_report.py [--results-dir ./results] [--format markdown|csv]
    python generate_report.py --results-dir ./results \
        --peak-flops 312 --peak-bw 2039
        # --peak-flops: single GPU peak FP16 TFLOPS (e.g. A100=312, H100=990)
        # --peak-bw:    single GPU peak memory bandwidth in GB/s (e.g. A100=2039, H100=3350)
"""

import argparse
import json
import math
import os
import sys


def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_ms(val_ms: float) -> str:
    """Format milliseconds to human-readable string."""
    if val_ms < 1:
        return f"{val_ms * 1000:.1f}us"
    if val_ms < 1000:
        return f"{val_ms:.2f}ms"
    if val_ms < 60_000:
        return f"{val_ms / 1000:.2f}s"
    if val_ms < 3600_000:
        return f"{val_ms / 60_000:.2f}min"
    return f"{val_ms / 3600_000:.2f}h"


# ---------------------------------------------------------------------------
# Hardware specs (override via --peak-flops and --peak-bw)
# ---------------------------------------------------------------------------
DEFAULT_PEAK_FLOPS_TFLOPS = 0    # set via CLI
DEFAULT_PEAK_BW_GB_S = 0         # set via CLI
SCENARIO_07_TPS = (1, 2, 4)
SCENARIO_07_OUTLENS = (200, 500)


def load_model_config(model_dir: str) -> dict | None:
    """Load model config.json, return None if not found."""
    for p in [os.path.join(model_dir, "config.json")]:
        if os.path.exists(p):
            with open(p) as f:
                cfg = json.load(f)
            tc = cfg.get("text_config", cfg)
            qc = cfg.get("quantization_config", {})
            return {
                "hidden_size": tc.get("hidden_size", 2048),
                "num_hidden_layers": tc.get("num_hidden_layers", 24),
                "num_attention_heads": tc.get("num_attention_heads", 16),
                "num_key_value_heads": tc.get("num_key_value_heads", 2),
                "head_dim": tc.get("head_dim", 256),
                "vocab_size": tc.get("vocab_size", 248320),
                "num_experts": tc.get("num_experts", 0),
                "num_experts_per_tok": tc.get("num_experts_per_tok", 0),
                "moe_intermediate_size": tc.get("moe_intermediate_size", 0),
                "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size", 0),
                "intermediate_size": tc.get("intermediate_size", 0),
                "layer_types": tc.get("layer_types", []),
                "linear_num_key_heads": tc.get("linear_num_key_heads", 0),
                "linear_num_value_heads": tc.get("linear_num_value_heads", 0),
                "linear_key_head_dim": tc.get("linear_key_head_dim", 0),
                "linear_value_head_dim": tc.get("linear_value_head_dim", 0),
                "attn_output_gate": tc.get("attn_output_gate", False),
                "quant_bits": qc.get("bits", 16),
                "quant_dynamic": qc.get("dynamic", {}),
            }
    return None


def count_full_attn_layers(cfg: dict) -> tuple[int, int]:
    """Return (n_full_attn_layers, n_linear_attn_layers)."""
    N = cfg["num_hidden_layers"]
    layer_types = cfg.get("layer_types", [])
    if layer_types:
        n_full = sum(1 for lt in layer_types[:N] if lt == "full_attention")
        return n_full, N - n_full
    return N, 0  # default: all full attention


def compute_prefill_flops(cfg: dict, seq_len: int, tp_size: int = 1,
                          batch_size: int = 1) -> float:
    """Compute total FLOPs for one prefill step (forward only).

    With batch>1, FLOPs scale linearly (batch independent prefills).
    For each linear layer: FLOPs = 2 × M × K × N
    Attention: 2 × seq × seq × head_dim × n_heads (for QK^T and attn×V)
    """
    H = cfg["hidden_size"]
    N = cfg["num_hidden_layers"]
    S = seq_len
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim", H // n_heads)
    has_gate = cfg.get("attn_output_gate", False)
    V = cfg["vocab_size"]

    n_experts = cfg.get("num_experts", 0)
    n_active = cfg.get("num_experts_per_tok", 0)
    moe_inter = cfg.get("moe_intermediate_size", 0)
    shared_inter = cfg.get("shared_expert_intermediate_size", 0)
    dense_inter = cfg.get("intermediate_size", H * 4)

    n_full_layers, n_lin_layers = count_full_attn_layers(cfg)

    # --- Full attention layer FLOPs ---
    # QKV projections: Q=[S,H]×[H, n_heads*head_dim*(2 if gate)], K/V=[S,H]×[H, n_kv*head_dim]
    q_out = n_heads * head_dim * (2 if has_gate else 1)
    qkv_flops = 2 * S * H * (q_out + 2 * n_kv * head_dim)
    # Attention: QK^T = [S, head_dim] × [head_dim, S] per head, then attn×V
    attn_flops = 2 * 2 * n_heads * S * S * head_dim  # QK^T + attn*V
    # Output projection: [S, n_heads*head_dim] × [n_heads*head_dim, H]
    o_flops = 2 * S * (n_heads * head_dim) * H

    full_attn_flops = qkv_flops + attn_flops + o_flops

    # --- Linear attention layer FLOPs (no S×S attention) ---
    lin_k_dim = cfg.get("linear_num_key_heads", 0) * cfg.get("linear_key_head_dim", 0)
    lin_v_dim = cfg.get("linear_num_value_heads", 0) * cfg.get("linear_value_head_dim", 0)
    if lin_k_dim > 0:
        lin_proj_flops = 2 * S * H * (lin_k_dim * 2 + lin_v_dim)  # QKV
        lin_proj_flops += 2 * S * H * lin_v_dim  # z proj
        lin_proj_flops += 2 * S * lin_v_dim * H  # out proj
        lin_attn_flops = lin_proj_flops
    else:
        lin_attn_flops = full_attn_flops  # fallback

    # --- MoE / FFN FLOPs per layer ---
    if n_experts > 0 and n_active > 0:
        # active experts: gate+up+down, each [S,H]×[H,inter] or [S,inter]×[inter,H]
        expert_flops = n_active * 3 * 2 * S * H * moe_inter
        shared_flops = 3 * 2 * S * H * shared_inter if shared_inter > 0 else 0
        ffn_flops = expert_flops + shared_flops
    else:
        ffn_flops = 3 * 2 * S * H * dense_inter

    # --- Total ---
    total = (n_full_layers * (full_attn_flops + ffn_flops) +
             n_lin_layers * (lin_attn_flops + ffn_flops))
    # LM head
    total += 2 * S * H * V
    # Per-GPU (TP splits compute)
    # Note: batch prefills run in parallel on GPU, TTFT measures wall clock
    # for one request. FLOPs = single request FLOPs (not × batch).
    total = total / tp_size

    return total


def compute_decode_bytes(cfg: dict, seq_len: int, tp_size: int = 1,
                         batch_size: int = 1) -> float:
    """Compute total bytes read per decode step: weights + KV cache.

    Weights: read once per step, shared across batch (no duplication).
    LM head is counted in logical TP form (vocab-parallel, /tp). The local
    split/pruned rank model may keep a full lm_head only to remain runnable
    under MIG simulation; compensated timing divides lm_head by TP.
    KV cache: each request has its own KV → total = batch × per_request_KV.
    KV cache: for each full-attention layer, read K and V of all seq_len tokens
              KV per layer = 2 × n_kv_heads × head_dim × seq_len × 2 (bf16)
              Linear attention: fixed state, negligible
    """
    H = cfg["hidden_size"]
    N = cfg["num_hidden_layers"]
    V = cfg["vocab_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg.get("num_key_value_heads", n_heads)
    head_dim = cfg.get("head_dim", H // n_heads)
    has_gate = cfg.get("attn_output_gate", False)

    n_experts = cfg.get("num_experts", 0)
    n_active = cfg.get("num_experts_per_tok", 0)
    moe_inter = cfg.get("moe_intermediate_size", 0)
    shared_inter = cfg.get("shared_expert_intermediate_size", 0)
    dense_inter = cfg.get("intermediate_size", H * 4)

    quant_bits = cfg.get("quant_bits", 16)
    bpp_q = quant_bits / 8 if quant_bits < 16 else 2  # quantized: int4 = 0.5 bytes
    bpp_f = 2  # bf16

    # Determine what's quantized from dynamic config (same logic as bandwidth_util.py)
    quant_dynamic = cfg.get("quant_dynamic", {})
    attn_quantized = True
    shared_quantized = True
    for pattern in quant_dynamic:
        pl = pattern.lower()
        if pl.startswith("-:"):
            if "attn" in pl:
                attn_quantized = False
            if "shared_expert" in pl:
                shared_quantized = False

    bpp_attn = bpp_q if attn_quantized else bpp_f
    bpp_shared = bpp_q if shared_quantized else bpp_f

    n_full_layers, n_lin_layers = count_full_attn_layers(cfg)

    # --- Weights per layer (active) ---
    # Full attention
    q_dim = n_heads * head_dim * (2 if has_gate else 1)
    kv_dim = n_kv * head_dim
    attn_params_full = (q_dim * H + 2 * kv_dim * H + n_heads * head_dim * H)
    attn_bytes_full = attn_params_full * bpp_attn / tp_size

    # Linear attention
    lin_k = cfg.get("linear_num_key_heads", 0) * cfg.get("linear_key_head_dim", 0)
    lin_v = cfg.get("linear_num_value_heads", 0) * cfg.get("linear_value_head_dim", 0)
    if lin_k > 0:
        lin_params = (lin_k * 2 + lin_v) * H + lin_v * H + lin_v * H  # qkv + z + out
        attn_bytes_lin = lin_params * bpp_attn / tp_size
    else:
        attn_bytes_lin = attn_bytes_full

    # MoE / FFN
    if n_experts > 0 and n_active > 0:
        expert_bytes = n_active * 3 * H * moe_inter * bpp_q / tp_size
        router_bytes = n_experts * H * bpp_f  # router always bf16
        shared_bytes = 3 * H * shared_inter * bpp_shared / tp_size if shared_inter > 0 else 0
        ffn_bytes = expert_bytes + router_bytes + shared_bytes
    else:
        ffn_bytes = 3 * H * dense_inter * bpp_q / tp_size

    norm_bytes = 2 * H * 2  # 2 norms × bf16

    weight_full = attn_bytes_full + ffn_bytes + norm_bytes
    weight_lin = attn_bytes_lin + ffn_bytes + norm_bytes
    total_weight = n_full_layers * weight_full + n_lin_layers * weight_lin
    # LM head is logically vocab-parallel in TP, even if the local MIG
    # simulation keeps a replicated tensor so rank_0 can run standalone.
    total_weight += V * H * bpp_f / max(tp_size, 1)
    total_weight += H * 2  # final norm

    # --- KV cache ---
    # Full attention: KV grows with seq_len
    #   Per layer = 2(K+V) × kv_heads_per_rank × head_dim × seq_len × 2 (bf16)
    kv_heads_per_rank = n_kv if tp_size > n_kv else n_kv // tp_size
    full_kv_per_layer = 2 * kv_heads_per_rank * head_dim * seq_len * 2
    total_full_kv = n_full_layers * full_kv_per_layer

    # Linear attention: fixed recurrent state S = K^T × V (seq-independent)
    #   Per layer = qk_heads × key_dim × key_dim × 2 (bf16)
    lin_qk_heads = cfg.get("linear_num_key_heads", 0)
    lin_key_dim = cfg.get("linear_key_head_dim", 0)
    if lin_qk_heads > 0 and lin_key_dim > 0:
        lin_state_per_layer = lin_qk_heads * lin_key_dim * lin_key_dim * 2
    else:
        lin_state_per_layer = 0
    total_lin_kv = n_lin_layers * lin_state_per_layer

    # KV cache is per-request, duplicated across batch
    total_kv = (total_full_kv + total_lin_kv) * batch_size

    return total_weight + total_kv, total_weight, total_kv


def _find_hidden_size(comp_path: str) -> int:
    """Try to find hidden_size from model config near a compensated.json."""
    import glob as _glob
    base = os.path.dirname(comp_path)
    for pattern in [
        os.path.join(base, "model", "rank_0_*", "config.json"),
        os.path.join(base, "model", "split", "rank_0", "config.json"),
        os.path.join(base, "..", "model", "rank_0_*", "config.json"),
        os.path.join(base, "..", "model", "split", "rank_0", "config.json"),
    ]:
        for p in _glob.glob(pattern):
            cfg = load_model_config(p.replace("/config.json", ""))
            if cfg:
                return cfg["hidden_size"]
    meta_path = os.path.join(base, "model", "split_meta.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(base, "..", "model", "split_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        orig = meta.get("original_model", "")
        if orig:
            cfg = load_model_config(orig)
            if cfg:
                return cfg["hidden_size"]
    return 0


def apply_comm_model(data: dict, bw_gbps: float, lat_us: float,
                     input_len: int, batch: int = 1,
                     comp_path: str = "") -> None:
    """Recompute compensated metrics with modeled communication.

    Per-operation latency = lat_us + data_bytes / (bw_gbps * 1e9).
    Two AllReduce ops per layer (attn output + MLP output).
    Modifies data["results"] in place.
    """
    if not data or data.get("tp_size", 1) <= 1:
        return

    tp = data["tp_size"]
    original_layers = data.get("original_layers", 1)
    layer_scale = data.get("layer_scale", 1)
    hidden = data.get("hidden_size", 0)
    tail = data.get("tail", {})

    if not hidden and comp_path:
        hidden = _find_hidden_size(comp_path)
        if hidden:
            data["hidden_size"] = hidden

    if not hidden:
        print(f"  WARNING: no hidden_size in {os.path.basename(comp_path or '?')}, "
              f"re-run compensate_ppu.py to add it")
        return
    if not tail.get("encoder_ms"):
        return

    encoder_ms = tail["encoder_ms"]
    lm_head_ms = tail.get("lm_head_ms", 0)
    sampling_ms = tail.get("sampling_ms", 0)
    lm_head_final = lm_head_ms / tp

    n_ops = original_layers * 2
    lat_ms = lat_us / 1000.0
    bw_bytes_per_ms = bw_gbps * 1e6

    decode_data = hidden * batch * 2
    decode_comm = n_ops * (lat_ms + decode_data / bw_bytes_per_ms)

    prefill_data = hidden * input_len * batch * 2
    prefill_comm = n_ops * (lat_ms + prefill_data / bw_bytes_per_ms)

    comp_tpot = encoder_ms * layer_scale + lm_head_final + sampling_ms + decode_comm

    ttft = tail.get("ttft_ms", 0)
    prefill_enc = max(ttft - lm_head_ms - sampling_ms, 0)
    comp_ttft = prefill_enc * layer_scale + lm_head_final + sampling_ms + prefill_comm

    for r in data.get("results", []):
        if "error" in r:
            continue
        kv_extra = r.get("kv_extra_tpot_ms", 0)
        ot = r.get("output_tokens", 64)
        tpot = comp_tpot + kv_extra
        r["comp_tpot_ms"] = round(tpot, 3)
        r["comp_ttft_ms"] = round(comp_ttft, 3)
        r["comp_total_ms"] = round(comp_ttft + (ot - 1) * tpot, 3)

    data["communication"] = {
        "method": "model",
        "bw_gbps": bw_gbps,
        "latency_us": lat_us,
        "decode_comm_ms": round(decode_comm, 4),
        "prefill_comm_ms": round(prefill_comm, 4),
    }


def get_metrics(data: dict) -> dict | None:
    """Extract comp_ttft, comp_tpot, tps, total from compensated json."""
    results = data.get("results", [])
    # Take first non-error result (or average if multiple input_lens)
    valid = [r for r in results if "error" not in r and "comp_tpot_ms" in r]
    if not valid:
        return None

    # Use the result matching the scenario's target input_len (usually only one)
    r = valid[0]
    ttft = r["comp_ttft_ms"]
    tpot = r["comp_tpot_ms"]
    tps = 1000.0 / tpot if tpot > 0 else 0
    total = r.get("comp_total_ms", ttft + (r.get("output_tokens", 64) - 1) * tpot)
    output_tokens = r.get("output_tokens", data.get("results", [{}])[0].get("output_len", 64))

    return {
        "ttft": ttft,
        "tpot": tpot,
        "tps": tps,
        "total": total,
        "output_tokens": output_tokens,
    }


def print_scenario(title: str, rows: list[tuple[str, dict | None]], fmt: str):
    """Print a scenario block with model rows."""
    if fmt == "csv":
        for model_name, m in rows:
            if m:
                print(f"{title},{model_name},{m['ttft']:.2f},{m['tpot']:.3f},"
                      f"{m['tps']:.1f},{m['total']:.1f}")
            else:
                print(f"{title},{model_name},N/A,N/A,N/A,N/A")
    else:
        print(f"\n### {title}\n")
        print(f"| 模型 | TTFT | TPOT | TPS | 总延迟 |")
        print(f"|------|------|------|-----|--------|")
        for model_name, m in rows:
            if m:
                print(f"| {model_name} | {fmt_ms(m['ttft'])} | {fmt_ms(m['tpot'])} | "
                      f"{m['tps']:.1f} tok/s | {fmt_ms(m['total'])} |")
            else:
                print(f"| {model_name} | N/A | N/A | N/A | N/A |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    ap.add_argument("--format", choices=["markdown", "csv"], default="markdown")
    ap.add_argument("--peak-flops", type=float, default=0,
                    help="Single GPU peak FP16 TFLOPS (e.g. A100=312, H100=990)")
    ap.add_argument("--peak-bw", type=float, default=0,
                    help="Single GPU peak memory bandwidth GB/s (e.g. A100=2039, H100=3350)")
    ap.add_argument("--src-flops", type=float, default=0,
                    help="Source platform peak TFLOPS (for rescaling prefill)")
    ap.add_argument("--tgt-flops", type=float, default=0,
                    help="Target platform peak TFLOPS")
    ap.add_argument("--src-bw", type=float, default=0,
                    help="Source platform peak BW GB/s (for rescaling decode)")
    ap.add_argument("--tgt-bw", type=float, default=0,
                    help="Target platform peak BW GB/s")
    ap.add_argument("--comm-bw", type=str, default="",
                    help="Interconnect BW in GB/s. Single value or per-TP: '2:100,4:50'")
    ap.add_argument("--comm-latency", type=str, default="",
                    help="Fixed per-op latency in us. Single value or per-TP: '2:3,4:5'")
    ap.add_argument("--label", choices=["ai_station", "vla"], default=None,
                    help="Filter scenarios: ai_station=01-06 only, vla=07 only")
    args = ap.parse_args()
    rd = args.results_dir
    fmt = args.format
    show_ai = args.label != "vla"
    show_vla = args.label != "ai_station"

    def _parse_tp_param(s: str) -> dict[int, float]:
        if not s:
            return {}
        try:
            return {0: float(s)}
        except ValueError:
            return {int(k): float(v) for k, v in
                    (p.split(":") for p in s.split(","))}

    comm_bw_map = _parse_tp_param(args.comm_bw)
    comm_lat_map = _parse_tp_param(args.comm_latency)
    comm_override = bool(comm_bw_map)

    def _load(path, input_len=0, batch=1):
        d = load_json(path)
        if d and comm_override:
            tp = d.get("tp_size", 1)
            bw = comm_bw_map.get(tp, comm_bw_map.get(0, 0))
            lat = comm_lat_map.get(tp, comm_lat_map.get(0, 0))
            if bw > 0:
                apply_comm_model(d, bw, lat, input_len, batch, comp_path=path)
        return d

    if comm_override and fmt == "markdown":
        parts = []
        for tp in sorted(k for k in comm_bw_map if k > 0):
            bw = comm_bw_map.get(tp, 0)
            lat = comm_lat_map.get(tp, 0)
            parts.append(f"TP={tp}: {lat}us + data/{bw}GB/s")
        if 0 in comm_bw_map:
            parts.append(f"default: {comm_lat_map.get(0,0)}us + data/{comm_bw_map[0]}GB/s")
        print(f"\n> **Communication model**: {'; '.join(parts)}  ")
        print(f"> (2 AllReduce per layer, bf16 activations)\n")

    if fmt == "csv":
        print("场景,模型,TTFT(ms),TPOT(ms),TPS(tok/s),总延迟(ms)")

    has_rescale = (args.src_flops > 0 and args.tgt_flops > 0) or \
                  (args.src_bw > 0 and args.tgt_bw > 0)
    if has_rescale:
        from model_weight_utils import rescale_metrics, decode_weight_bytes
        sf, tf = args.src_flops, args.tgt_flops
        sb, tb = args.src_bw, args.tgt_bw
        tgt_parts = []
        if tf > 0:
            tgt_parts.append(f"{tf}T")
        if tb > 0:
            tgt_parts.append(f"{tb}GB/s")
        tgt_label = f"→ Target ({'/'.join(tgt_parts)})"

    # =========================================================================
    # 1. Code Completion (1.5K input, 50 output)
    # =========================================================================
    if show_ai:
        d01 = _load(os.path.join(rd, "01_code_completion_35B", "compensated.json"), 1536)
        m01 = get_metrics(d01) if d01 else None
        print_scenario(
            "Code Completion (1.5K input, 50 output)",
            [("Qwen3.5-35B-A3B-GPTQ-INT4", m01)],
            fmt,
        )

    # =========================================================================
    # 2. Chat Q&A (25K input, 1K output)
    # =========================================================================
    if show_ai:
        d02 = _load(os.path.join(rd, "02_chat_27B", "compensated.json"), 25600)
        m02 = get_metrics(d02) if d02 else None

        d03_tp1 = _load(os.path.join(rd, "03_chat_122B", "tp1", "compensated.json"), 25600)
        m03_tp1 = get_metrics(d03_tp1) if d03_tp1 else None

        d03_tp2 = _load(os.path.join(rd, "03_chat_122B", "tp2", "compensated.json"), 25600)
        m03_tp2 = get_metrics(d03_tp2) if d03_tp2 else None

        print_scenario(
            "Chat Q&A (25K input, 1K output)",
            [
                ("Qwen3.5-27B", m02),
                ("Qwen3.5-122B-A10B-GPTQ-INT4 TP=1", m03_tp1),
                ("Qwen3.5-122B-A10B-GPTQ-INT4 TP=2", m03_tp2),
            ],
            fmt,
        )

    # =========================================================================
    # 3. Agent Full Task (100K first + 9×20K hit)
    # =========================================================================
    if show_ai:
        # First call (100K input, 3K output)
        agent_first = {}
        for scenario, tp_list in [("04_agent_122B", [1, 2]), ("05_agent_397B", [2, 4])]:
            for tp in tp_list:
                d = _load(os.path.join(rd, scenario, f"tp{tp}", "compensated.json"), 102400)
                agent_first[(scenario, tp)] = get_metrics(d) if d else None

        # Hit calls (20K input, 3K output)
        agent_hit = {}
        for scenario, tp_list in [("04b_agent_hit_122B", [1, 2]), ("05b_agent_hit_397B", [2, 4])]:
            for tp in tp_list:
                d = _load(os.path.join(rd, scenario, f"tp{tp}", "compensated.json"), 20480)
                agent_hit[(scenario, tp)] = get_metrics(d) if d else None

        # Map hit scenarios to first-call scenarios
        hit_map = {
            ("04b_agent_hit_122B", 1): ("04_agent_122B", 1),
            ("04b_agent_hit_122B", 2): ("04_agent_122B", 2),
            ("05b_agent_hit_397B", 2): ("05_agent_397B", 2),
            ("05b_agent_hit_397B", 4): ("05_agent_397B", 4),
        }

        # Model display names
        model_names = {
            ("04_agent_122B", 1): "Qwen3.5-122B-A10B-GPTQ-INT4 TP=1",
            ("04_agent_122B", 2): "Qwen3.5-122B-A10B-GPTQ-INT4 TP=2",
            ("05_agent_397B", 2): "Qwen3.5-397B-A17B-GPTQ-INT4 TP=2",
            ("05_agent_397B", 4): "Qwen3.5-397B-A17B-GPTQ-INT4 TP=4",
        }

        # Compute full task total latency: first + 9 × hit
        if fmt == "markdown":
            print(f"\n### Agent Full Task (1M tokens = 100K first + 9×100K@80%hit)")
            print(f"\n**全任务总延迟:**\n")
            print(f"| 模型 | 第一次(100K) | 后续×9(20K) | 全任务总延迟 |")
            print(f"|------|-------------|-------------|-------------|")
        else:
            print()

        for first_key in [("04_agent_122B", 1), ("04_agent_122B", 2),
                           ("05_agent_397B", 2), ("05_agent_397B", 4)]:
            hit_key = [hk for hk, fk in hit_map.items() if fk == first_key]
            hit_key = hit_key[0] if hit_key else None

            mf = agent_first.get(first_key)
            mh = agent_hit.get(hit_key) if hit_key else None
            name = model_names[first_key]

            if mf and mh:
                full_total = mf["total"] + 9 * mh["total"]
                if fmt == "csv":
                    print(f"Agent全任务,{name},{mf['total']:.1f},{mh['total']:.1f},{full_total:.1f},")
                else:
                    print(f"| {name} | {fmt_ms(mf['total'])} | {fmt_ms(mh['total'])} | {fmt_ms(full_total)} |")
            else:
                if fmt == "csv":
                    print(f"Agent全任务,{name},N/A,N/A,N/A,")
                else:
                    print(f"| {name} | N/A | N/A | N/A |")

        # First call detail
        print_scenario(
            "Agent 第一次调用 (100K input, 3K output)",
            [(model_names[k], agent_first[k]) for k in
             [("04_agent_122B", 1), ("04_agent_122B", 2),
              ("05_agent_397B", 2), ("05_agent_397B", 4)]],
            fmt,
        )

        # Hit call detail
        hit_names = {
            ("04b_agent_hit_122B", 1): "Qwen3.5-122B-A10B-GPTQ-INT4 TP=1",
            ("04b_agent_hit_122B", 2): "Qwen3.5-122B-A10B-GPTQ-INT4 TP=2",
            ("05b_agent_hit_397B", 2): "Qwen3.5-397B-A17B-GPTQ-INT4 TP=2",
            ("05b_agent_hit_397B", 4): "Qwen3.5-397B-A17B-GPTQ-INT4 TP=4",
        }
        print_scenario(
            "Agent 后续调用 (20K input@80%hit, 3K output)",
            [(hit_names[k], agent_hit[k]) for k in
             [("04b_agent_hit_122B", 1), ("04b_agent_hit_122B", 2),
              ("05b_agent_hit_397B", 2), ("05b_agent_hit_397B", 4)]],
            fmt,
        )

    # =========================================================================
    # 4. Batch Sweep (04c/05c)
    # =========================================================================
    if show_ai:
        batch_rows_122b = []
        batch_rows_397b = []
        for b in [1, 2, 4, 8]:
            for tp, rows in [(1, batch_rows_122b), (2, batch_rows_122b)]:
                d = _load(os.path.join(rd, "04c_agent_batch_122B", f"tp{tp}",
                                       f"compensated_batch{b}.json"), 102400, b)
                m = get_metrics(d) if d else None
                rows.append((f"TP={tp} batch={b}", m))
            for tp in [2, 4]:
                d = _load(os.path.join(rd, "05c_agent_batch_397B", f"tp{tp}",
                                       f"compensated_batch{b}.json"), 102400, b)
                m = get_metrics(d) if d else None
                batch_rows_397b.append((f"TP={tp} batch={b}", m))

        if any(m for _, m in batch_rows_122b):
            print_scenario(
                "Agent Batch Sweep 122B (100K input, 3K output)",
                batch_rows_122b, fmt)
        if any(m for _, m in batch_rows_397b):
            print_scenario(
                "Agent Batch Sweep 397B (100K input, 3K output)",
                batch_rows_397b, fmt)

    # =========================================================================
    # 5. RAG Repo Understanding (800K input, 3K output)
    # =========================================================================
    if show_ai:
        d06 = _load(os.path.join(rd, "06_rag_35B", "compensated.json"), 819200)
        m06 = get_metrics(d06) if d06 else None
        print_scenario(
            "RAG Repo Understanding (800K input, 3K output)",
            [("Qwen3.5-35B-A3B-GPTQ-INT4", m06)],
            fmt,
        )

    # =========================================================================
    # AI Station → Target Platform (rescaled)
    # =========================================================================
    if show_ai and has_rescale:
        _rs = lambda m: rescale_metrics(m, {}, sf, tf, sb, tb)

        print_scenario(
            f"Code Completion {tgt_label} (1.5K input, 50 output)",
            [("Qwen3.5-35B-A3B-GPTQ-INT4", _rs(m01))],
            fmt,
        )

        print_scenario(
            f"Chat Q&A {tgt_label} (25K input, 1K output)",
            [
                ("Qwen3.5-27B", _rs(m02)),
                ("Qwen3.5-122B-A10B-GPTQ-INT4 TP=1", _rs(m03_tp1)),
                ("Qwen3.5-122B-A10B-GPTQ-INT4 TP=2", _rs(m03_tp2)),
            ],
            fmt,
        )

        af_rs = {k: _rs(v) for k, v in agent_first.items()}
        ah_rs = {k: _rs(v) for k, v in agent_hit.items()}

        if fmt == "markdown":
            print(f"\n### Agent Full Task {tgt_label} (1M tokens = 100K first + 9×100K@80%hit)")
            print(f"\n**全任务总延迟:**\n")
            print(f"| 模型 | 第一次(100K) | 后续×9(20K) | 全任务总延迟 |")
            print(f"|------|-------------|-------------|-------------|")
        else:
            print()

        for first_key in [("04_agent_122B", 1), ("04_agent_122B", 2),
                           ("05_agent_397B", 2), ("05_agent_397B", 4)]:
            hit_key = [hk for hk, fk in hit_map.items() if fk == first_key]
            hit_key = hit_key[0] if hit_key else None
            mf = af_rs.get(first_key)
            mh = ah_rs.get(hit_key) if hit_key else None
            name = model_names[first_key]

            if mf and mh:
                full_total = mf["total"] + 9 * mh["total"]
                if fmt == "csv":
                    print(f"Agent全任务{tgt_label},{name},{mf['total']:.1f},{mh['total']:.1f},{full_total:.1f},")
                else:
                    print(f"| {name} | {fmt_ms(mf['total'])} | {fmt_ms(mh['total'])} | {fmt_ms(full_total)} |")
            else:
                if fmt == "csv":
                    print(f"Agent全任务{tgt_label},{name},N/A,N/A,N/A,")
                else:
                    print(f"| {name} | N/A | N/A | N/A |")

        print_scenario(
            f"Agent 第一次调用 {tgt_label} (100K input, 3K output)",
            [(model_names[k], af_rs[k]) for k in
             [("04_agent_122B", 1), ("04_agent_122B", 2),
              ("05_agent_397B", 2), ("05_agent_397B", 4)]],
            fmt,
        )

        print_scenario(
            f"Agent 后续调用 {tgt_label} (20K input@80%hit, 3K output)",
            [(hit_names[k], ah_rs[k]) for k in
             [("04b_agent_hit_122B", 1), ("04b_agent_hit_122B", 2),
              ("05b_agent_hit_397B", 2), ("05b_agent_hit_397B", 4)]],
            fmt,
        )

        if any(m for _, m in batch_rows_122b):
            print_scenario(
                f"Agent Batch Sweep 122B {tgt_label} (100K input, 3K output)",
                [(name, _rs(m)) for name, m in batch_rows_122b], fmt)
        if any(m for _, m in batch_rows_397b):
            print_scenario(
                f"Agent Batch Sweep 397B {tgt_label} (100K input, 3K output)",
                [(name, _rs(m)) for name, m in batch_rows_397b], fmt)

        print_scenario(
            f"RAG Repo Understanding {tgt_label} (800K input, 3K output)",
            [("Qwen3.5-35B-A3B-GPTQ-INT4", _rs(m06))],
            fmt,
        )

    # =========================================================================
    # 6. Qwen3-30B-A3B (1.5K input, 200/500 output)
    # =========================================================================
    if show_vla:
        # INT4 projected — read directly from compensated JSON
        def get_int4_metrics(d):
            if not d or "int4" not in d:
                return None
            i = d["int4"]
            return {
                "ttft": i["comp_ttft_ms"],
                "tpot": i["comp_tpot_ms"],
                "tps": i["tps"],
                "total": i["total_ms"],
                "output_tokens": d.get("results", [{}])[0].get("output_tokens", 64),
            }

        d07 = {}
        for tp in SCENARIO_07_TPS:
            for out_len in SCENARIO_07_OUTLENS:
                path = os.path.join(
                    rd, "07_qwen3_30b_a3b", f"tp{tp}",
                    f"compensated_{out_len}.json")
                data = _load(path, 1536)
                d07[(tp, out_len)] = data

        bf16_rows = []
        int4_rows = []
        for tp in SCENARIO_07_TPS:
            for out_len in SCENARIO_07_OUTLENS:
                data = d07[(tp, out_len)]
                metric = get_metrics(data) if data else None
                bf16_rows.append((f"TP={tp} Output={out_len}", metric))
                int4 = get_int4_metrics(data)
                int4_rows.append((f"TP={tp} Output={out_len} (INT4)", int4))
                if metric and not int4:
                    print(
                        f"  WARNING: no 'int4' for TP={tp} output={out_len}. "
                        "Re-run compensate_scenarios/07")

        # BF16 measured (source platform)
        print_scenario(
            "Qwen3-30B-A3B BF16 measured (1.5K input)",
            bf16_rows,
            fmt,
        )

        print_scenario(
            "Qwen3-30B-A3B INT4 projected (1.5K input)",
            int4_rows,
            fmt,
        )

        # BF16/INT4 → target platform (if --src/tgt given)
        if has_rescale:
            bf16_tgt_rows = []
            int4_tgt_rows = []
            for tp in SCENARIO_07_TPS:
                info = decode_weight_bytes("qwen3-30b-a3b", tp=tp)
                for out_len in SCENARIO_07_OUTLENS:
                    data = d07[(tp, out_len)]
                    metric = get_metrics(data) if data else None
                    int4 = get_int4_metrics(data)
                    label = f"TP={tp} Output={out_len}"
                    bf16_tgt_rows.append((
                        label,
                        rescale_metrics(metric, info, sf, tf, sb, tb)
                        if metric else None,
                    ))
                    int4_tgt_rows.append((
                        f"{label} (INT4)",
                        rescale_metrics(int4, info, sf, tf, sb, tb)
                        if int4 else None,
                    ))

            print_scenario(
                f"Qwen3-30B-A3B BF16 {tgt_label} (1.5K input)",
                bf16_tgt_rows,
                fmt,
            )
            print_scenario(
                f"Qwen3-30B-A3B INT4 {tgt_label} (1.5K input)",
                int4_tgt_rows,
                fmt,
            )

    # =========================================================================
    # 5. Prefill MFU + Decode Bandwidth Utilization
    # =========================================================================
    if args.peak_flops > 0 or args.peak_bw > 0:
        # Collect all scenarios: (name, model_dir, input_len, output_len, tp_size, metrics)
        scenarios = []
        # (name, scenario_dir, sub_dir, input_len, output_len, tp, batch)
        scenario_defs = [
            ("01 Code Completion 35B", "01_code_completion_35B", None, 1536, 50, 1, 1),
            ("02 Chat 27B", "02_chat_27B", None, 25600, 1024, 1, 1),
            ("03 Chat 122B TP=1", "03_chat_122B", "tp1", 25600, 1024, 1, 1),
            ("03 Chat 122B TP=2", "03_chat_122B", "tp2", 25600, 1024, 2, 1),
            ("04 Agent 122B TP=1", "04_agent_122B", "tp1", 102400, 3072, 1, 1),
            ("04 Agent 122B TP=2", "04_agent_122B", "tp2", 102400, 3072, 2, 1),
            ("05 Agent 397B TP=2", "05_agent_397B", "tp2", 102400, 3072, 2, 1),
            ("05 Agent 397B TP=4", "05_agent_397B", "tp4", 102400, 3072, 4, 1),
            ("06 RAG 35B", "06_rag_35B", None, 819200, 3072, 1, 1),
        ]
        for tp in SCENARIO_07_TPS:
            for out_len in SCENARIO_07_OUTLENS:
                scenario_defs.append((
                    f"07 30B-A3B TP={tp} out={out_len}",
                    "07_qwen3_30b_a3b",
                    f"tp{tp}",
                    1536,
                    out_len,
                    tp,
                    1,
                    f"compensated_{out_len}.json",
                ))
        # Add batch scenarios (04c/05c)
        for b in [1, 2, 4, 8]:
            scenario_defs.append(
                (f"04c 122B TP=1 B={b}", "04c_agent_batch_122B", "tp1", 102400, 3072, 1, b))
            scenario_defs.append(
                (f"04c 122B TP=2 B={b}", "04c_agent_batch_122B", "tp2", 102400, 3072, 2, b))
            scenario_defs.append(
                (f"05c 397B TP=2 B={b}", "05c_agent_batch_397B", "tp2", 102400, 3072, 2, b))
            scenario_defs.append(
                (f"05c 397B TP=4 B={b}", "05c_agent_batch_397B", "tp4", 102400, 3072, 4, b))

        # Filter scenario_defs based on --label
        if not show_vla:
            scenario_defs = [s for s in scenario_defs if not s[1].startswith("07_")]
        if not show_ai:
            scenario_defs = [s for s in scenario_defs if s[1].startswith("07_")]

        if fmt == "markdown":
            print(f"\n### Prefill MFU & Decode Bandwidth Utilization")
            if args.peak_flops > 0:
                print(f"\nPeak compute: {args.peak_flops} TFLOPS (FP16)")
            if args.peak_bw > 0:
                print(f"Peak bandwidth: {args.peak_bw} GB/s")
            print()
            cols = "| 场景 | Prefill MFU |" if args.peak_flops > 0 else "| 场景 |"
            if args.peak_bw > 0:
                cols += " Decode BW Util | Weights | KV Cache |"
            print(cols)
            sep = "|------|"
            if args.peak_flops > 0:
                sep += "------------|"
            if args.peak_bw > 0:
                sep += "---------------|---------|----------|"
            print(sep)

        for entry in scenario_defs:
            name, sc_dir, sub, input_len, output_len, tp, batch = entry[:7]
            custom_file = entry[7] if len(entry) > 7 else None

            if sub:
                base = os.path.join(rd, sc_dir, sub)
            else:
                base = os.path.join(rd, sc_dir)

            # Compensated file path
            if custom_file:
                comp_file = os.path.join(base, custom_file)
            elif batch > 1:
                comp_file = os.path.join(base, f"compensated_batch{batch}.json")
            else:
                comp_file = os.path.join(base, "compensated.json")
            comp = _load(comp_file, input_len, batch)
            m = get_metrics(comp) if comp else None

            # Find model config (batch scenarios reuse model from parent scenario)
            import glob as _glob
            model_dirs = _glob.glob(os.path.join(base, "model", "rank_0_*L"))
            if not model_dirs:
                model_dirs = _glob.glob(os.path.join(base, "model", "split", "rank_0"))
            # Fallback: batch scenarios (04c/05c) reuse model from 04/05
            if not model_dirs:
                parent = sc_dir.replace("04c_agent_batch_122B", "04_agent_122B") \
                               .replace("05c_agent_batch_397B", "05_agent_397B")
                if parent != sc_dir:
                    parent_base = os.path.join(rd, parent, sub) if sub else os.path.join(rd, parent)
                    model_dirs = _glob.glob(os.path.join(parent_base, "model", "rank_0_*L"))
            cfg = load_model_config(model_dirs[0]) if model_dirs else None

            # Scenario 07: use known full model config (all fields included)
            if sc_dir == "07_qwen3_30b_a3b":
                from model_weight_utils import load_model_config as _load_mwu
                cfg = _load_mwu("qwen3-30b-a3b")

            # Pruned config has TP-split values and truncated layer_types.
            # Try to load the ORIGINAL model config from split_meta.json.
            elif cfg and comp:
                orig_cfg = None
                meta_path = os.path.join(base, "model", "split_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path) as _f:
                        meta = json.load(_f)
                    orig_model = meta.get("original_model", "")
                    if orig_model:
                        orig_cfg = load_model_config(orig_model)

                if orig_cfg:
                    # Use original config directly — correct heads, layers, layer_types
                    cfg = orig_cfg
                else:
                    # Fallback: restore from pruned config
                    orig_layers = comp.get("original_layers", cfg["num_hidden_layers"])
                    cfg["num_hidden_layers"] = orig_layers
                    lt = cfg.get("layer_types", [])
                    if lt and len(lt) < orig_layers:
                        cycle = len(lt)
                        cfg["layer_types"] = [lt[i % cycle] for i in range(orig_layers)]
                    if tp > 1:
                        for key in ("num_attention_heads", "num_key_value_heads",
                                    "linear_num_key_heads", "linear_num_value_heads",
                                    "moe_intermediate_size", "shared_expert_intermediate_size"):
                            if key in cfg:
                                cfg[key] = max(cfg[key] * tp, 1)

            mfu_str = "N/A"
            bw_str = "N/A"
            w_str = "N/A"
            kv_str = "N/A"

            if cfg and m:
                if args.peak_flops > 0:
                    flops = compute_prefill_flops(cfg, input_len, tp, batch)
                    ttft_s = m["ttft"] / 1000
                    actual_tflops = flops / ttft_s / 1e12 if ttft_s > 0 else 0
                    mfu = actual_tflops / args.peak_flops * 100
                    mfu_str = f"{mfu:.1f}%"

                if args.peak_bw > 0:
                    # avg_seq_len during decode = input_len + output_len/2
                    avg_seq = input_len + output_len // 2
                    total_bytes, weight_bytes, kv_bytes = compute_decode_bytes(
                        cfg, avg_seq, tp, batch)
                    tpot_s = m["tpot"] / 1000
                    bw_used = total_bytes / tpot_s / 1e9 if tpot_s > 0 else 0
                    bw_util = bw_used / args.peak_bw * 100
                    bw_str = f"{bw_util:.1f}%"
                    w_str = f"{weight_bytes / 1e9:.2f}GB"
                    kv_str = f"{kv_bytes / 1e9:.2f}GB (×{batch})"

            if fmt == "csv":
                print(f"Utilization,{name},{mfu_str},{bw_str},{w_str},{kv_str}")
            else:
                row = f"| {name} |"
                if args.peak_flops > 0:
                    row += f" {mfu_str} |"
                if args.peak_bw > 0:
                    row += f" {bw_str} | {w_str} | {kv_str} |"
                print(row)

    # =========================================================================
    # Summary
    # =========================================================================
    if fmt == "markdown":
        print("\n---")
        print("*Generated by generate_report.py from compensated results*")


if __name__ == "__main__":
    main()
