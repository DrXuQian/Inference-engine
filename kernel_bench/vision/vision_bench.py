#!/usr/bin/env python3
"""
Vision model microbenchmarks for PyTorch trace and CUDA Graph replay.

Default workloads:
  vit:
    4 images, 480x480, ViT-L-like synthetic trunk
  clipdino:
    1 image, 480x480, CLIP ViT-B/16 vision tower from
    openai/clip-vit-base-patch16 config

The models use synthetic weights and inputs. This is intended to measure kernel
shape and runtime behavior, not model accuracy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import nvtx
    has_nvtx = True
except ImportError:
    has_nvtx = False


CLIP_VIT_B16_CONFIG = {
    "source": "https://huggingface.co/openai/clip-vit-base-patch16/blob/main/config.json",
    "config_image_size": 224,
    "patch_size": 16,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "projection_dim": 512,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
}


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def make_activation(name: str) -> nn.Module:
    if name == "quick_gelu":
        return QuickGELU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported activation: {name}")


class ViTBlock(nn.Module):
    def __init__(
        self,
        hidden: int,
        heads: int,
        mlp_dim: int,
        act: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden % heads != 0:
            raise ValueError(f"hidden={hidden} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden, eps=layer_norm_eps)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden, eps=layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_dim),
            make_activation(act),
            nn.Linear(mlp_dim, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(
            batch, tokens, 3, self.heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        h = h.transpose(1, 2).reshape(batch, tokens, hidden)
        x = x + self.out_proj(h)
        return x + self.mlp(self.norm2(x))


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        hidden: int,
        layers: int,
        heads: int,
        mlp_dim: int,
        in_channels: int = 3,
        projection_dim: int | None = None,
        act: str = "gelu",
        layer_norm_eps: float = 1e-5,
        pool: str = "cls",
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if pool not in ("cls", "mean"):
            raise ValueError("pool must be cls or mean")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.hidden = hidden
        self.pool = pool

        self.patch_embed = nn.Conv2d(
            in_channels, hidden, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden))
        self.pre_ln = nn.LayerNorm(hidden, eps=layer_norm_eps)
        self.blocks = nn.ModuleList(
            [
                ViTBlock(hidden, heads, mlp_dim, act, layer_norm_eps)
                for _ in range(layers)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden, eps=layer_norm_eps)
        self.proj = (
            nn.Linear(hidden, projection_dim, bias=False)
            if projection_dim is not None
            else nn.Identity()
        )
        self._init_parameters()

    @property
    def tokens(self) -> int:
        return self.num_patches + 1

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.01)

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(images.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pre_ln(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.final_ln(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(images)
        if self.pool == "cls":
            pooled = tokens[:, 0]
        else:
            pooled = tokens[:, 1:].mean(dim=1)
        return self.proj(pooled)


class ClipDinoVision(nn.Module):
    def __init__(
        self,
        image_size: int = 480,
    ) -> None:
        super().__init__()
        cfg = CLIP_VIT_B16_CONFIG
        self.trunk = VisionTransformer(
            image_size=image_size,
            patch_size=cfg["patch_size"],
            hidden=cfg["hidden_size"],
            layers=cfg["num_hidden_layers"],
            heads=cfg["num_attention_heads"],
            mlp_dim=cfg["intermediate_size"],
            projection_dim=cfg["projection_dim"],
            act=cfg["hidden_act"],
            layer_norm_eps=cfg["layer_norm_eps"],
            pool="cls",
        )

    @property
    def tokens(self) -> int:
        return self.trunk.tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.trunk(images)


def capture_cuda_graph(fn: Callable[[], object], warmup: int = 3,
                       stream=None) -> torch.cuda.CUDAGraph:
    """Same CUDA Graph capture flow as vla/vla_bench.py."""
    s = stream or torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    return graph


def bench_component(
    name: str,
    fn: Callable[[], object],
    warmup: int = 10,
    iters: int = 50,
    use_cuda_graph: bool = False,
) -> float:
    """Benchmark a callable using the same timing method as vla/vla_bench.py."""
    if has_nvtx:
        rng = nvtx.start_range(f"{name}_warmup", color="red")
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    if has_nvtx:
        nvtx.end_range(rng)

    graph = None
    if use_cuda_graph:
        try:
            graph = capture_cuda_graph(fn)
            print(f"  [{name}] CUDA Graph captured")
        except Exception as e:
            print(f"  [{name}] CUDA Graph capture failed ({e}), using eager")

    run_fn = graph.replay if graph else fn

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


def maybe_trace_model(
    model: nn.Module,
    example: tuple[torch.Tensor, ...],
    enabled: bool,
    name: str,
) -> nn.Module:
    if not enabled:
        return model
    print(f"[{name}] torch.jit.trace: capturing static PyTorch graph")
    with torch.inference_mode():
        traced = torch.jit.trace(model, example, strict=False)
        traced = torch.jit.freeze(traced.eval())
    return traced


def count_params_m(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def make_vit(args: argparse.Namespace, dtype: torch.dtype, device: str):
    model = VisionTransformer(
        image_size=args.vit_image_size,
        patch_size=args.vit_patch_size,
        hidden=args.vit_hidden,
        layers=args.vit_layers,
        heads=args.vit_heads,
        mlp_dim=args.vit_mlp_dim,
        projection_dim=args.vit_projection_dim if args.vit_projection_dim > 0 else None,
        act=args.vit_act,
        pool=args.vit_pool,
    ).to(device=device, dtype=dtype).eval()
    images = torch.randn(
        args.vit_batch,
        3,
        args.vit_image_size,
        args.vit_image_size,
        device=device,
        dtype=dtype,
    )
    return model, (images,)


def make_clipdino(args: argparse.Namespace, dtype: torch.dtype, device: str):
    model = ClipDinoVision(
        image_size=args.clipdino_image_size,
    ).to(device=device, dtype=dtype).eval()
    images = torch.randn(
        args.clipdino_batch,
        3,
        args.clipdino_image_size,
        args.clipdino_image_size,
        device=device,
        dtype=dtype,
    )
    return model, (images,)


def add_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision benchmark using VLA-compatible timing")
    parser.add_argument("--component", choices=["vit", "clipdino", "all"], default="all")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--torch-trace", action="store_true", help="Run torch.jit.trace before benchmarking")
    parser.add_argument("--cuda-graph", action="store_true", help="Capture and replay the forward with CUDA Graph")
    parser.add_argument("--output-json", default=None)

    parser.add_argument("--vit-batch", type=int, default=4)
    parser.add_argument("--vit-image-size", type=int, default=480)
    parser.add_argument("--vit-patch-size", type=int, default=16)
    parser.add_argument("--vit-hidden", type=int, default=1024)
    parser.add_argument("--vit-layers", type=int, default=24)
    parser.add_argument("--vit-heads", type=int, default=16)
    parser.add_argument("--vit-mlp-dim", type=int, default=4096)
    parser.add_argument("--vit-projection-dim", type=int, default=0)
    parser.add_argument("--vit-act", choices=["gelu", "quick_gelu"], default="gelu")
    parser.add_argument("--vit-pool", choices=["cls", "mean"], default="cls")

    parser.add_argument("--clipdino-batch", type=int, default=1)
    parser.add_argument("--clipdino-image-size", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = add_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be >= 0 and iters must be > 0")

    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    torch.set_float32_matmul_precision("high")

    print("=" * 80)
    print("Vision Benchmark (VIT + CLIP-DINO)")
    print(f"component={args.component} dtype={args.dtype} device={args.device}")
    print(f"torch_trace={args.torch_trace} cuda_graph={args.cuda_graph}")
    print("timing=VLA-compatible sync + time.perf_counter median")
    print(
        "CLIP config source: "
        f"{CLIP_VIT_B16_CONFIG['source']} "
        f"(bench image_size={args.clipdino_image_size})"
    )
    print("=" * 80)

    results = {}
    config = {"vit": None, "clipdino": None}

    if args.component in ("vit", "all"):
        model, example = make_vit(args, dtype, args.device)
        params_m = count_params_m(model)
        model = maybe_trace_model(model, example, args.torch_trace, "vit")
        tokens = (args.vit_image_size // args.vit_patch_size) ** 2 + 1
        holder = {}

        def run_vit():
            holder["vit"] = model(*example)
            return holder["vit"]

        print(
            f"vit: batch={args.vit_batch} image={args.vit_image_size} "
            f"patch={args.vit_patch_size} tokens={tokens} "
            f"layers={args.vit_layers} hidden={args.vit_hidden} params={params_m:.1f}M"
        )
        results["vit_ms"] = bench_component(
            "VIT", run_vit, args.warmup, args.iters, args.cuda_graph)
        config["vit"] = {
            "batch": args.vit_batch,
            "image_size": args.vit_image_size,
            "patch_size": args.vit_patch_size,
            "tokens": tokens,
            "hidden": args.vit_hidden,
            "layers": args.vit_layers,
            "heads": args.vit_heads,
            "mlp_dim": args.vit_mlp_dim,
            "params_m": params_m,
        }
        del model, example, holder
        torch.cuda.empty_cache()

    if args.component in ("clipdino", "all"):
        model, example = make_clipdino(args, dtype, args.device)
        params_m = count_params_m(model)
        model = maybe_trace_model(model, example, args.torch_trace, "clipdino")
        tokens = (args.clipdino_image_size // CLIP_VIT_B16_CONFIG["patch_size"]) ** 2 + 1
        holder = {}

        def run_clipdino():
            holder["clipdino"] = model(*example)
            return holder["clipdino"]

        print(
            f"clipdino: batch={args.clipdino_batch} image={args.clipdino_image_size} "
            f"patch={CLIP_VIT_B16_CONFIG['patch_size']} tokens={tokens} "
            f"layers={CLIP_VIT_B16_CONFIG['num_hidden_layers']} "
            f"hidden={CLIP_VIT_B16_CONFIG['hidden_size']} "
            f"params={params_m:.1f}M"
        )
        results["clipdino_ms"] = bench_component(
            "CLIPDINO", run_clipdino, args.warmup, args.iters, args.cuda_graph)
        config["clipdino"] = {
            "batch": args.clipdino_batch,
            "image_size": args.clipdino_image_size,
            "patch_size": CLIP_VIT_B16_CONFIG["patch_size"],
            "tokens": tokens,
            "hidden": CLIP_VIT_B16_CONFIG["hidden_size"],
            "layers": CLIP_VIT_B16_CONFIG["num_hidden_layers"],
            "heads": CLIP_VIT_B16_CONFIG["num_attention_heads"],
            "mlp_dim": CLIP_VIT_B16_CONFIG["intermediate_size"],
            "params_m": params_m,
        }
        del model, example, holder
        torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print("Results")
    print("-" * 40)
    for key, value in results.items():
        print(f"  {key:25s}: {value:8.2f} ms")
    print("=" * 80)

    if args.output_json:
        output = {
            "component_results": results,
            "config": {k: v for k, v in config.items() if v is not None},
            "dtype": args.dtype,
            "timing": "sync_perf_counter_median",
            "clip_vit_b16_config": CLIP_VIT_B16_CONFIG,
            "args": vars(args),
        }
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved JSON: {path}")


if __name__ == "__main__":
    main()
