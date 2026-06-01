#!/usr/bin/env python3
"""
VLA (Vision-Language-Action) model benchmark — VIT and DiT components.

LLM part runs separately via vLLM (see run_vla_bench.sh).

Components:
  1. VIT: spatio-temporal vision transformer
     - 24 layers, hidden=1024, heads=16
     - Input: 6 cam × 225 pos × 4 patches = 5400 tokens
     - Every 4 layers: causal cross-attention with 17-frame history (11475 tokens)
     - Output: aggregate 4 patches → 1, project to 2560D → 6×225 = 1350 tokens
  2. DiT: diffusion action transformer
     - 18 layers, hidden=1024, heads=8
     - Input: 51 tokens (50 action steps + 1 robot state) × 62D → projected to 1024
     - Cross-attention with LLM KV cache (1550 tokens × 2560 → projected to 1024)
     - 10 denoising steps (default), each = full forward pass
     - Output: 51 tokens → project to 62D

Usage:
    python vla_bench.py                           # both VIT + DiT
    python vla_bench.py --component vit           # VIT only
    python vla_bench.py --component dit           # DiT only
    python vla_bench.py --dtype bf16 --no-compile # eager mode
    python vla_bench.py --cuda-graph              # CUDA Graph replay

    # Under profiler (use --cuda-graph-trace=node to expand graph kernels)
    nsys profile -t cuda --cuda-graph-trace=node -o vla_trace \
        python vla_bench.py --cuda-graph
    asys profile -t hggc,acdnn,acblas,hgtx -o vla_trace \
        python vla_bench.py --cuda-graph

    # Analyze GEMM breakdown + FP8/FP4 projection
    python trace_gemm_scale.py vla_trace.sqlite --nvtx-filter "VIT_0"
    python trace_gemm_scale.py vla_trace.sqlite --nvtx-filter "DiT_1step_0"
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

try:
    import nvtx
    has_nvtx = True
except ImportError:
    has_nvtx = False


# ============================================================
# VIT: Spatio-Temporal Vision Transformer
# ============================================================
# Reference: Qwen2-VL vision encoder (ViT-L: hidden=1024, heads=16, depth=24)
# Extended with spatio-temporal cross-attention to historical frames.

class VITBlock(nn.Module):
    """Single VIT block: self-attention + FFN, pre-norm. Uses FlashAttention via SDPA."""
    def __init__(self, hidden, heads, ffn_dim, dtype):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden, dtype=dtype)
        self.qkv = nn.Linear(hidden, 3 * hidden, dtype=dtype)
        self.out_proj = nn.Linear(hidden, hidden, dtype=dtype)
        self.norm2 = nn.LayerNorm(hidden, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden, dtype=dtype),
        )

    def forward(self, x):
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, heads, N, head_dim)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, C)
        h = self.out_proj(h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention with historical frames, pre-norm. Uses FlashAttention via SDPA."""
    def __init__(self, hidden, heads, dtype):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm_q = nn.LayerNorm(hidden, dtype=dtype)
        self.norm_kv = nn.LayerNorm(hidden, dtype=dtype)
        self.q_proj = nn.Linear(hidden, hidden, dtype=dtype)
        self.k_proj = nn.Linear(hidden, hidden, dtype=dtype)
        self.v_proj = nn.Linear(hidden, hidden, dtype=dtype)
        self.out_proj = nn.Linear(hidden, hidden, dtype=dtype)

    def forward(self, x, kv):
        B, Nq, C = x.shape
        Nkv = kv.shape[1]
        h = self.norm_q(x)
        kv_n = self.norm_kv(kv)
        q = self.q_proj(h).reshape(B, Nq, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_n).reshape(B, Nkv, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_n).reshape(B, Nkv, self.heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, Nq, C)
        h = self.out_proj(h)
        return x + h


class SpatioTemporalVIT(nn.Module):
    """
    Spatio-temporal VIT: spatial self-attention + temporal self-attention.

    Architecture:
      ①a/①b Spatial: per-camera self-attention on 225 tokens (post patch-merge)
        - 3 target cameras + 3 current cameras processed independently
        - 24 layers, hidden=1024, heads=16
      ①b+ Temporal: every cross_every=4 layers, temporal self-attention
        - Each spatial position attends across (history_frames+1) timesteps
        - batch = n_cams/2 × tokens_per_cam = 675 positions
        - seq = history_frames + 1 = 18
        - 6 temporal layers total

    Input: (B, n_cams × tokens_per_cam, hidden) = (B, 1350, 1024)
      - 225 tokens per camera (already patch-merged, NOT 225×4)
    History: (B, n_cams/2 × history_frames × tokens_per_cam, hidden)
      - 17 frames × 225 tokens per camera × 3 cameras
    Output: (B, n_cams × tokens_per_cam, out_dim) = (B, 1350, 2560)
    """
    def __init__(self, hidden=1024, heads=16, depth=24, cross_every=4,
                 patch_factor=4, n_cams=6, tokens_per_cam=225,
                 history_frames=17,
                 out_dim=2560, ffn_dim=4096, dtype=torch.bfloat16):
        super().__init__()
        self.depth = depth
        self.cross_every = cross_every
        self.patch_factor = patch_factor
        self.n_cams = n_cams
        self.tokens_per_cam = tokens_per_cam
        self.history_frames = history_frames
        self.n_temporal_steps = history_frames + 1  # 18
        self.cams_with_history = n_cams // 2  # 3

        # Patch merge: 225×4 patches → 225 tokens (aggregate 4 patches per position)
        self.patch_merge = nn.Linear(hidden * patch_factor, hidden, dtype=dtype)

        # Spatial self-attention blocks (shared, per-camera)
        self.blocks = nn.ModuleList([
            VITBlock(hidden, heads, ffn_dim, dtype) for _ in range(depth)
        ])

        # Temporal self-attention blocks (every cross_every layers)
        n_temporal = depth // cross_every
        self.temporal_blocks = nn.ModuleList([
            VITBlock(hidden, heads, ffn_dim, dtype) for _ in range(n_temporal)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(hidden, dtype=dtype)
        self.proj_out = nn.Linear(hidden, out_dim, dtype=dtype)

    def forward(self, x, history):
        """
        x: (B, n_cams * tokens_per_cam * patch_factor, hidden) = (B, 5400, 1024)
           900 patches per camera × 6 cameras
        history: (B, cams_with_history * history_frames * tokens_per_cam, hidden)
           = (B, 3 * 17 * 225, hidden) = (B, 11475, 1024)
        Returns: (B, n_cams * tokens_per_cam, out_dim) = (B, 1350, 2560)
        """
        B = x.shape[0]
        C = x.shape[2]
        NC = self.n_cams           # 6
        S = self.tokens_per_cam    # 225
        PF = self.patch_factor     # 4
        S_full = S * PF            # 900 (full seq before merge)
        NC_hist = self.cams_with_history  # 3
        HF = self.history_frames   # 17
        NT = self.n_temporal_steps # 18

        # Per-camera spatial: (B, NC*S_full, C) → (B*NC, 900, C)
        x = x.view(B * NC, S_full, C)

        # History for temporal attention: (B, NC_hist*HF*S, C) → (B, NC_hist, HF, S, C)
        hist = history.view(B, NC_hist, HF, S, C)

        temporal_idx = 0
        for i in range(self.depth):
            # Spatial self-attention on 900 tokens per camera
            x = self.blocks[i](x)  # (B*NC, 900, 1024)

            if (i + 1) % self.cross_every == 0:
                # Temporal attention: pool 4 patches → 1 position, attend across time
                # (B*NC, 900, C) → (B, NC, 225, 4, C) → pool → (B, NC, 225, C)
                x_all = x.view(B, NC, S, PF, C)
                x_pooled = x_all.mean(dim=3)  # (B, NC, 225, C)

                # Current cameras with history (last NC_hist)
                x_curr = x_pooled[:, NC_hist:, :, :]  # (B, 3, 225, C)

                # Stack with history: (B, 3, 18, 225, C)
                temporal_seq = torch.cat([hist, x_curr.unsqueeze(2)], dim=2)

                # Reshape: (B*3*225, 18, C)
                temporal_seq = temporal_seq.permute(0, 1, 3, 2, 4).reshape(B * NC_hist * S, NT, C)

                # Temporal self-attention
                temporal_seq = self.temporal_blocks[temporal_idx](temporal_seq)  # (B*675, 18, C)

                # Extract current timestep, broadcast back to 4 patches
                x_temporal = temporal_seq[:, -1, :].view(B, NC_hist, S, 1, C)
                x_temporal = x_temporal.expand(-1, -1, -1, PF, -1)  # (B, 3, 225, 4, C)

                # Update current cameras (residual add)
                x_all = x_all.clone()
                x_all[:, NC_hist:] = x_all[:, NC_hist:] + x_temporal

                # Back to (B*NC, 900, C)
                x = x_all.reshape(B * NC, S_full, C)
                temporal_idx += 1

        # Patch merge at the END: (B*NC, 900, C) → (B, NC*225, 4*C) → merge → (B, 1350, C)
        x = x.view(B, NC, S, PF, C)
        x = x.reshape(B, NC * S, PF * C)
        x = self.patch_merge(x)  # (B, 1350, 1024)
        x = self.norm_out(x)
        return self.proj_out(x)  # (B, 1350, 2560)


# ============================================================
# DiT: Diffusion Action Transformer
# ============================================================

class DiTBlock(nn.Module):
    """DiT block: self-attention + cross-attention + FFN, pre-norm. Uses FlashAttention via SDPA."""
    def __init__(self, hidden, heads, ffn_dim, dtype):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        # Self-attention
        self.norm1 = nn.LayerNorm(hidden, dtype=dtype)
        self.sa_qkv = nn.Linear(hidden, 3 * hidden, dtype=dtype)
        self.sa_out = nn.Linear(hidden, hidden, dtype=dtype)
        # Cross-attention with LLM KV
        self.norm2 = nn.LayerNorm(hidden, dtype=dtype)
        self.norm_kv = nn.LayerNorm(hidden, dtype=dtype)
        self.ca_q = nn.Linear(hidden, hidden, dtype=dtype)
        self.ca_k = nn.Linear(hidden, hidden, dtype=dtype)
        self.ca_v = nn.Linear(hidden, hidden, dtype=dtype)
        self.ca_out = nn.Linear(hidden, hidden, dtype=dtype)
        # FFN
        self.norm3 = nn.LayerNorm(hidden, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ffn_dim, dtype=dtype),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden, dtype=dtype),
        )

    def forward(self, x, kv):
        B, N, C = x.shape
        Nkv = kv.shape[1]
        # Self-attention
        h = self.norm1(x)
        qkv = self.sa_qkv(h).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, C)
        x = x + self.sa_out(h)
        # Cross-attention
        h = self.norm2(x)
        kv_n = self.norm_kv(kv)
        q = self.ca_q(h).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.ca_k(kv_n).reshape(B, Nkv, self.heads, self.head_dim).transpose(1, 2)
        v = self.ca_v(kv_n).reshape(B, Nkv, self.heads, self.head_dim).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, C)
        x = x + self.ca_out(h)
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class ActionDiT(nn.Module):
    """
    Diffusion Action Transformer.

    Architecture:
      - 18 layers, hidden=1024, heads=8
      - Each layer: self-attn + cross-attn(LLM KV) + FFN
      - Input: action(50×62) + state(1×62) → project to 1024 → 51 tokens
      - Cross-attn KV: LLM output (1550 tokens × 2560) → project to 1024
      - Output: 51 tokens × 1024 → project to 62D
      - Runs N denoising steps (default 50)
    """
    def __init__(self, hidden=1024, heads=8, layers=18,
                 action_dim=62, llm_hidden=2560,
                 ffn_dim=4096, dtype=torch.bfloat16):
        super().__init__()
        self.proj_in = nn.Linear(action_dim, hidden, dtype=dtype)
        self.kv_proj = nn.Linear(llm_hidden, hidden, dtype=dtype)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden, heads, ffn_dim, dtype) for _ in range(layers)
        ])

        self.norm_out = nn.LayerNorm(hidden, dtype=dtype)
        self.proj_out = nn.Linear(hidden, action_dim, dtype=dtype)

    def forward(self, action_tokens, llm_kv):
        """
        action_tokens: (B, 51, 62)
        llm_kv: (B, 1550, 2560)
        Returns: (B, 51, 62)
        """
        x = self.proj_in(action_tokens)       # (B, 51, 1024)
        kv = self.kv_proj(llm_kv)             # (B, 1550, 1024)
        for block in self.blocks:
            x = block(x, kv)
        x = self.norm_out(x)
        return self.proj_out(x)               # (B, 51, 62)


# ============================================================
# CUDA Graph capture
# ============================================================

def capture_cuda_graph(fn, *example_args, warmup=3, stream=None):
    """Capture a function call as a CUDA Graph for replay."""
    s = stream or torch.cuda.Stream()
    # Warmup on side stream
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    return graph


# ============================================================
# Benchmark harness
# ============================================================

def bench_component(name, fn, warmup=10, iters=50, use_cuda_graph=False):
    """Benchmark a callable with NVTX markers and optional CUDA Graph."""
    # Warmup (eager)
    if has_nvtx:
        rng = nvtx.start_range(f"{name}_warmup", color="red")
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    if has_nvtx:
        nvtx.end_range(rng)

    # Capture CUDA Graph if requested
    graph = None
    if use_cuda_graph:
        try:
            graph = capture_cuda_graph(fn)
            print(f"  [{name}] CUDA Graph captured")
        except Exception as e:
            print(f"  [{name}] CUDA Graph capture failed ({e}), using eager")

    run_fn = graph.replay if graph else fn

    # Benchmark
    times = []
    with torch.no_grad():
        for i in range(iters):
            if has_nvtx:
                rng = nvtx.start_range(f"{name}_{i}", color="green")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_fn()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            if has_nvtx:
                nvtx.end_range(rng)

    times.sort()
    median = times[len(times) // 2]
    p10 = times[len(times) // 10] if len(times) >= 10 else times[0]
    p90 = times[len(times) * 9 // 10] if len(times) >= 10 else times[-1]
    mode = "cudagraph" if graph else "eager"
    print(f"  [{name}] ({mode}) median={median:.2f}ms  p10={p10:.2f}ms  p90={p90:.2f}ms")
    return median


def main():
    ap = argparse.ArgumentParser(description="VLA VIT/DiT benchmark")
    ap.add_argument("--component", choices=["vit", "dit", "all"], default="all",
                    help="Which component to benchmark (LLM uses vLLM separately)")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="Capture forward as CUDA Graph for replay (eliminates launch overhead)")
    ap.add_argument("--device", default="cuda:0")
    # VIT
    ap.add_argument("--vit-hidden", type=int, default=1024)
    ap.add_argument("--vit-heads", type=int, default=16)
    ap.add_argument("--vit-depth", type=int, default=24)
    ap.add_argument("--vit-cross-every", type=int, default=4)
    ap.add_argument("--vit-cams", type=int, default=6)
    ap.add_argument("--vit-tokens-per-cam", type=int, default=225)
    ap.add_argument("--vit-history-frames", type=int, default=17)
    ap.add_argument("--vit-out-dim", type=int, default=2560)
    ap.add_argument("--vit-ffn-dim", type=int, default=4096,
                    help="VIT FFN intermediate size (Qwen3-VL: 4096)")
    # DiT
    ap.add_argument("--dit-hidden", type=int, default=1024)
    ap.add_argument("--dit-heads", type=int, default=8)
    ap.add_argument("--dit-layers", type=int, default=18)
    ap.add_argument("--dit-ffn-dim", type=int, default=4096,
                    help="DiT FFN intermediate size")
    ap.add_argument("--dit-action-dim", type=int, default=62)
    ap.add_argument("--dit-denoise-steps", type=int, default=10)
    ap.add_argument("--dit-llm-hidden", type=int, default=2560)
    ap.add_argument("--dit-llm-tokens", type=int, default=1550)
    # Output
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    dev = args.device
    use_compile = not args.no_compile
    use_cudagraph = args.cuda_graph

    patch_factor = 4
    n_vit_input = args.vit_cams * args.vit_tokens_per_cam * patch_factor  # 6×225×4=5400
    n_vit_tokens = args.vit_cams * args.vit_tokens_per_cam  # 6×225=1350 (after merge)
    n_cams_hist = args.vit_cams // 2  # 3 current cameras have history
    n_hist_per_cam = args.vit_history_frames * args.vit_tokens_per_cam  # 17×225
    n_hist = n_cams_hist * n_hist_per_cam  # 3×17×225=11475
    n_dit_tokens = args.dit_denoise_steps + 1

    print("=" * 60)
    print("VLA Benchmark (VIT + DiT)")
    print(f"  dtype={args.dtype}, device={dev}, compile={use_compile}")
    print(f"  VIT: {n_vit_input} input → {n_vit_tokens} merged tokens ({args.vit_cams}cam × {args.vit_tokens_per_cam}tok), {args.vit_depth}L×{args.vit_hidden}")
    print(f"       spatial: {args.vit_tokens_per_cam} tokens/cam, temporal: {args.vit_history_frames+1} steps × {n_cams_hist*args.vit_tokens_per_cam} positions")
    print(f"  DiT: {n_dit_tokens} tokens × {args.dit_denoise_steps} steps, {args.dit_layers}L×{args.dit_hidden}")
    print(f"  LLM: run separately via vLLM (Qwen3-4B, {args.dit_llm_tokens} tokens prefill)")
    print("=" * 60)

    results = {}

    # === VIT ===
    if args.component in ("vit", "all"):
        print(f"\n{'='*60}")
        print(f"VIT: {args.vit_depth}L×{args.vit_hidden}")
        print(f"  spatial: {args.vit_cams}cam × {args.vit_tokens_per_cam}tok, {args.vit_depth}L")
        print(f"  temporal: {n_cams_hist*args.vit_tokens_per_cam} positions × {args.vit_history_frames+1} steps, {args.vit_depth//args.vit_cross_every}L")
        print(f"{'='*60}")

        vit = SpatioTemporalVIT(
            hidden=args.vit_hidden, heads=args.vit_heads,
            depth=args.vit_depth, cross_every=args.vit_cross_every,
            n_cams=args.vit_cams,
            tokens_per_cam=args.vit_tokens_per_cam,
            history_frames=args.vit_history_frames,
            out_dim=args.vit_out_dim,
            ffn_dim=args.vit_ffn_dim, dtype=dtype,
        ).to(dev).eval()

        x_vit = torch.randn(1, n_vit_input, args.vit_hidden, dtype=dtype, device=dev)
        hist = torch.randn(1, n_hist, args.vit_hidden, dtype=dtype, device=dev)

        params = sum(p.numel() for p in vit.parameters()) / 1e6
        print(f"  params: {params:.1f}M")

        if use_compile:
            try:
                vit = torch.compile(vit, mode="max-autotune")
                print("  torch.compile: enabled")
            except Exception as e:
                print(f"  torch.compile: failed ({e})")

        results["vit_ms"] = bench_component("VIT", lambda: vit(x_vit, hist),
                                            args.warmup, args.iters, use_cudagraph)
        del vit, x_vit, hist
        torch.cuda.empty_cache()

    # === DiT ===
    if args.component in ("dit", "all"):
        print(f"\n{'='*60}")
        print(f"DiT: {args.dit_layers}L×{args.dit_hidden}, {n_dit_tokens} tokens")
        print(f"  cross-attn with LLM KV: {args.dit_llm_tokens}×{args.dit_llm_hidden}")
        print(f"  {args.dit_denoise_steps} denoising steps")
        print(f"{'='*60}")

        dit = ActionDiT(
            hidden=args.dit_hidden, heads=args.dit_heads,
            layers=args.dit_layers, action_dim=args.dit_action_dim,
            llm_hidden=args.dit_llm_hidden, ffn_dim=args.dit_ffn_dim,
            dtype=dtype,
        ).to(dev).eval()

        llm_kv = torch.randn(1, args.dit_llm_tokens, args.dit_llm_hidden, dtype=dtype, device=dev)
        action_in = torch.randn(1, n_dit_tokens, args.dit_action_dim, dtype=dtype, device=dev)

        params = sum(p.numel() for p in dit.parameters()) / 1e6
        print(f"  params: {params:.1f}M")

        if use_compile:
            try:
                dit = torch.compile(dit, mode="max-autotune")
                print("  torch.compile: enabled")
            except Exception as e:
                print(f"  torch.compile: failed ({e})")

        # Single step
        results["dit_1step_ms"] = bench_component(
            "DiT_1step", lambda: dit(action_in, llm_kv),
            args.warmup, args.iters, use_cudagraph)

        # Full denoising (50 steps) — no CUDA Graph (loop not capturable as single graph)
        def dit_full():
            x = action_in
            for _ in range(args.dit_denoise_steps):
                x = dit(x, llm_kv)
            return x

        results["dit_full_ms"] = bench_component(
            f"DiT_{args.dit_denoise_steps}steps", dit_full,
            max(args.warmup // 5, 3), max(args.iters // 5, 10),
            use_cuda_graph=False)  # loop can't be captured as one graph

        del dit, llm_kv, action_in
        torch.cuda.empty_cache()

    # === Summary ===
    print(f"\n{'='*60}")
    print("Results")
    print("-" * 40)
    for k, v in results.items():
        print(f"  {k:25s}: {v:8.2f} ms")
    if "dit_full_ms" in results:
        print(f"  {'dit_per_step_ms':25s}: {results['dit_full_ms']/args.dit_denoise_steps:8.2f} ms")
    total_vit_dit = results.get("vit_ms", 0) + results.get("dit_full_ms", 0)
    if total_vit_dit > 0:
        print(f"  {'vit+dit_total':25s}: {total_vit_dit:8.2f} ms")
    print(f"\n  NOTE: Add LLM prefill time from vLLM trace for full pipeline latency")
    print("=" * 60)

    # Save
    if args.output_json:
        out = {
            "component_results": results,
            "config": {
                "vit": {"hidden": args.vit_hidden, "depth": args.vit_depth,
                        "in_tokens": n_vit_tokens, "out_tokens": n_vit_tokens,
                        "history_tokens": n_hist},
                "dit": {"hidden": args.dit_hidden, "layers": args.dit_layers,
                        "tokens": n_dit_tokens, "denoise_steps": args.dit_denoise_steps,
                        "llm_kv_tokens": args.dit_llm_tokens},
            },
            "dtype": args.dtype,
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.output_json}")


if __name__ == "__main__":
    main()
