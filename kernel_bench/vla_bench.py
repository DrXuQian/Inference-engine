#!/usr/bin/env python3
"""
VLA (Vision-Language-Action) model benchmark via PyTorch tracing.

Profiles three components independently:
  1. VIT: spatio-temporal vision transformer (24L, hidden=1024)
  2. LLM: language model prefill (36L, hidden=2560)
  3. DiT: diffusion action transformer, 50 denoise steps (18L, hidden=1024)

Usage:
    # Full benchmark (all 3 components)
    python vla_bench.py

    # Under profiler
    nsys profile -t cuda --cuda-graph-trace=node -o vla_trace python vla_bench.py
    # or on PPU:
    asys profile -t hggc,acdnn,acblas,hgtx -o vla_trace python vla_bench.py

    # Single component
    python vla_bench.py --component vit
    python vla_bench.py --component llm
    python vla_bench.py --component dit

    # Custom params
    python vla_bench.py --dtype bf16 --warmup 10 --iters 50
"""

import argparse
import time

import torch
import torch.nn as nn

try:
    import nvtx
    has_nvtx = True
except ImportError:
    has_nvtx = False


# ============================================================
# VIT: Spatio-Temporal Vision Transformer
# ============================================================
class SpatioTemporalVIT(nn.Module):
    """
    24 layers, hidden=1024, heads=16.
    Every 4 layers: cross-attention with historical KV cache.
    Input: 6 cameras × 225 tokens = 1350 tokens
    History: 17 frames × 3 cameras × 225 tokens = 11475 tokens (for cross-attn)
    """
    def __init__(self, hidden=1024, heads=16, depth=24, cross_every=4,
                 ffn_mult=4, dtype=torch.bfloat16):
        super().__init__()
        self.depth = depth
        self.cross_every = cross_every

        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads,
                dim_feedforward=hidden * ffn_mult,
                batch_first=True, dtype=dtype, norm_first=True,
            ) for _ in range(depth)
        ])
        # Cross-attention layers (every cross_every layers)
        n_cross = depth // cross_every
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, batch_first=True, dtype=dtype)
            for _ in range(n_cross)
        ])
        self.proj_out = nn.Linear(hidden, 2560, dtype=dtype)

    def forward(self, x, history_kv):
        cross_idx = 0
        for i in range(self.depth):
            x = self.self_attn_layers[i](x)
            if (i + 1) % self.cross_every == 0:
                # Cross-attention with historical frames
                x_res = x
                x, _ = self.cross_attn_layers[cross_idx](x, history_kv, history_kv)
                x = x + x_res
                cross_idx += 1
        return self.proj_out(x)


# ============================================================
# LLM: Language Model (prefill only)
# ============================================================
class LLMPrefill(nn.Module):
    """
    36 layers, hidden=2560, heads=32, kv_heads=8.
    GQA not directly supported by nn.TransformerEncoder, use MHA as proxy.
    Input: 1550 tokens (1350 visual + 200 task)
    """
    def __init__(self, hidden=2560, heads=32, layers=36,
                 ffn_mult=4, dtype=torch.bfloat16):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads,
                dim_feedforward=hidden * ffn_mult,
                batch_first=True, dtype=dtype, norm_first=True,
            ),
            num_layers=layers,
        )

    def forward(self, x):
        return self.encoder(x)


# ============================================================
# DiT: Diffusion Action Transformer
# ============================================================
class ActionDiT(nn.Module):
    """
    18 layers, hidden=1024, heads=8.
    Cross-attention with LLM KV cache (1550 tokens × 2560 → projected to 1024).
    Input: 51 tokens (50 action steps + 1 robot state) × 1024
    Runs 50 diffusion denoising steps.
    """
    def __init__(self, hidden=1024, heads=8, layers=18,
                 llm_hidden=2560, ffn_mult=4, dtype=torch.bfloat16):
        super().__init__()
        self.proj_in = nn.Linear(62, hidden, dtype=dtype)  # action dim → hidden
        self.kv_proj = nn.Linear(llm_hidden, hidden, dtype=dtype)  # LLM hidden → DiT hidden
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=heads,
                    dim_feedforward=hidden * ffn_mult,
                    batch_first=True, dtype=dtype, norm_first=True,
                ),
                'cross_attn': nn.MultiheadAttention(
                    hidden, heads, batch_first=True, dtype=dtype
                ),
            }))
        self.proj_out = nn.Linear(hidden, 62, dtype=dtype)  # hidden → action dim

    def forward(self, action_tokens, llm_kv):
        x = self.proj_in(action_tokens)
        kv = self.kv_proj(llm_kv)
        for layer in self.layers:
            x = layer['self_attn'](x)
            x_res = x
            x, _ = layer['cross_attn'](x, kv, kv)
            x = x + x_res
        return self.proj_out(x)


# ============================================================
# Benchmark harness
# ============================================================

def bench_component(name, model, inputs, warmup=10, iters=50, use_compile=True):
    """Benchmark a single component with optional torch.compile."""
    device = next(model.parameters()).device

    if use_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
            print(f"  [{name}] torch.compile enabled")
        except Exception as e:
            print(f"  [{name}] torch.compile failed ({e}), using eager")

    # Warmup
    if has_nvtx:
        rng = nvtx.start_range(f"{name}_warmup", color="red")
    with torch.no_grad():
        for _ in range(warmup):
            model(*inputs)
    torch.cuda.synchronize()
    if has_nvtx:
        nvtx.end_range(rng)

    # Benchmark
    torch.cuda.synchronize()
    if has_nvtx:
        rng = nvtx.start_range(f"{name}_bench", color="blue")

    times = []
    with torch.no_grad():
        for i in range(iters):
            if has_nvtx:
                rng_iter = nvtx.start_range(f"{name}_iter_{i}", color="green")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(*inputs)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            if has_nvtx:
                nvtx.end_range(rng_iter)

    if has_nvtx:
        nvtx.end_range(rng)

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    p90 = times[len(times) * 9 // 10]
    print(f"  [{name}] median={median:.2f}ms  p10={p10:.2f}ms  p90={p90:.2f}ms")
    return median


def main():
    ap = argparse.ArgumentParser(description="VLA model component benchmark")
    ap.add_argument("--component", choices=["vit", "llm", "dit", "all"], default="all")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    ap.add_argument("--device", default="cuda:0")
    # VIT params
    ap.add_argument("--vit-cams", type=int, default=6)
    ap.add_argument("--vit-tokens-per-cam", type=int, default=225)
    ap.add_argument("--vit-history-frames", type=int, default=17)
    ap.add_argument("--vit-history-cams", type=int, default=3)
    # LLM params
    ap.add_argument("--llm-task-tokens", type=int, default=200)
    # DiT params
    ap.add_argument("--dit-denoise-steps", type=int, default=50)
    ap.add_argument("--dit-action-dim", type=int, default=62)
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    dev = args.device
    use_compile = not args.no_compile

    n_vis_tokens = args.vit_cams * args.vit_tokens_per_cam  # 6×225=1350
    n_hist_tokens = args.vit_history_frames * args.vit_history_cams * args.vit_tokens_per_cam  # 17×3×225=11475
    n_llm_tokens = n_vis_tokens + args.llm_task_tokens  # 1350+200=1550
    n_dit_tokens = args.dit_denoise_steps + 1  # 50+1=51

    print("=" * 60)
    print("VLA Benchmark")
    print(f"  dtype={args.dtype}, device={dev}, compile={use_compile}")
    print(f"  warmup={args.warmup}, iters={args.iters}")
    print(f"  VIT: {n_vis_tokens} tokens, history={n_hist_tokens} tokens")
    print(f"  LLM: {n_llm_tokens} tokens (prefill)")
    print(f"  DiT: {n_dit_tokens} tokens × {args.dit_denoise_steps} denoise steps")
    print("=" * 60)

    results = {}

    # === VIT ===
    if args.component in ("vit", "all"):
        print("\n=== VIT (Spatio-Temporal, 24L×1024) ===")
        vit = SpatioTemporalVIT(dtype=dtype).to(dev).eval()
        x_vit = torch.randn(1, n_vis_tokens, 1024, dtype=dtype, device=dev)
        hist = torch.randn(1, n_hist_tokens, 1024, dtype=dtype, device=dev)
        params = sum(p.numel() for p in vit.parameters()) / 1e6
        print(f"  params: {params:.1f}M")
        results["vit_ms"] = bench_component("VIT", vit, (x_vit, hist),
                                            args.warmup, args.iters, use_compile)
        del vit, x_vit, hist
        torch.cuda.empty_cache()

    # === LLM ===
    if args.component in ("llm", "all"):
        print("\n=== LLM (Prefill, 36L×2560) ===")
        llm = LLMPrefill(dtype=dtype).to(dev).eval()
        x_llm = torch.randn(1, n_llm_tokens, 2560, dtype=dtype, device=dev)
        params = sum(p.numel() for p in llm.parameters()) / 1e6
        print(f"  params: {params:.1f}M")
        results["llm_ms"] = bench_component("LLM", llm, (x_llm,),
                                            args.warmup, args.iters, use_compile)
        # Save LLM output for DiT cross-attention
        with torch.no_grad():
            llm_out = llm(x_llm).detach()
        del llm, x_llm
        torch.cuda.empty_cache()

    # === DiT ===
    if args.component in ("dit", "all"):
        print(f"\n=== DiT (Action, 18L×1024, {args.dit_denoise_steps} steps) ===")
        dit = ActionDiT(dtype=dtype).to(dev).eval()
        # LLM KV for cross-attention
        if "llm_out" not in dir():
            llm_out = torch.randn(1, n_llm_tokens, 2560, dtype=dtype, device=dev)
        action_input = torch.randn(1, n_dit_tokens, args.dit_action_dim, dtype=dtype, device=dev)
        params = sum(p.numel() for p in dit.parameters()) / 1e6
        print(f"  params: {params:.1f}M (per step)")

        # Benchmark SINGLE step first
        results["dit_step_ms"] = bench_component("DiT_1step", dit, (action_input, llm_out),
                                                 args.warmup, args.iters, use_compile)

        # Benchmark full 50-step denoising
        def dit_full_denoise():
            x = action_input
            for _ in range(args.dit_denoise_steps):
                x_proj = dit(x, llm_out)
            return x_proj

        print(f"\n  DiT full ({args.dit_denoise_steps} steps):")
        if has_nvtx:
            rng = nvtx.start_range("DiT_full_warmup", color="red")
        with torch.no_grad():
            for _ in range(args.warmup):
                dit_full_denoise()
        torch.cuda.synchronize()
        if has_nvtx:
            nvtx.end_range(rng)

        times = []
        with torch.no_grad():
            for i in range(args.iters):
                if has_nvtx:
                    rng = nvtx.start_range(f"DiT_full_{i}", color="green")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                dit_full_denoise()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
                if has_nvtx:
                    nvtx.end_range(rng)

        times.sort()
        median = times[len(times) // 2]
        print(f"  [DiT_full] median={median:.2f}ms ({median/args.dit_denoise_steps:.2f}ms/step)")
        results["dit_full_ms"] = median

        del dit, action_input, llm_out
        torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 60)
    print("Summary")
    print("-" * 40)
    total = 0
    for k, v in results.items():
        print(f"  {k:20s}: {v:.2f} ms")
        if k in ("vit_ms", "llm_ms", "dit_full_ms"):
            total += v
    if total > 0:
        print(f"  {'TOTAL':20s}: {total:.2f} ms")
        print(f"  {'FPS':20s}: {1000/total:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
