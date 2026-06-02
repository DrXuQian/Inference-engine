#!/usr/bin/env python3
"""
Vision model microbenchmarks for PyTorch trace and CUDA Graph replay.

Default workloads:
  vit:
    4 images, 480x480, ViT-L-like synthetic trunk
  clipdino:
    1 image, 480x480, CLIP ViT-B/16 trunk from openai/clip-vit-base-patch16
    config plus a small DINO-style MLP head

The models use synthetic weights and inputs. This is intended to measure kernel
shape and runtime behavior, not model accuracy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


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


@dataclass
class BenchResult:
    name: str
    mode: str
    batch: int
    image_size: int
    patch_size: int
    tokens: int
    params_m: float
    warmup: int
    iters: int
    median_ms: float
    mean_ms: float
    p10_ms: float
    p90_ms: float
    min_ms: float
    max_ms: float


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


class DinoHead(nn.Module):
    def __init__(self, hidden: int, head_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClipDinoVision(nn.Module):
    def __init__(
        self,
        image_size: int = 480,
        dino_head_dim: int = 2048,
        dino_out_dim: int = 1024,
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
        self.dino_head = DinoHead(cfg["hidden_size"], dino_head_dim, dino_out_dim)

    @property
    def tokens(self) -> int:
        return self.trunk.tokens

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.trunk.forward_tokens(images)
        cls = tokens[:, 0]
        clip = self.trunk.proj(cls)
        dino = self.dino_head(cls)
        return clip, dino


@contextmanager
def nvtx_range(name: str, enabled: bool):
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * pct))))
    return sorted_values[idx]


def capture_cuda_graph(run_fn: Callable[[], object], warmup: int) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(max(warmup, 1)):
            run_fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_fn()
    torch.cuda.synchronize()
    return graph


def bench_callable(
    name: str,
    run_fn: Callable[[], object],
    batch: int,
    image_size: int,
    patch_size: int,
    tokens: int,
    params_m: float,
    warmup: int,
    iters: int,
    use_cuda_graph: bool,
    nvtx: bool,
) -> BenchResult:
    torch.cuda.synchronize()
    with torch.inference_mode(), nvtx_range(f"{name}_warmup", nvtx):
        for _ in range(warmup):
            run_fn()
    torch.cuda.synchronize()

    mode = "eager"
    if use_cuda_graph:
        with torch.inference_mode():
            graph = capture_cuda_graph(run_fn, warmup=3)
        run_fn = graph.replay
        mode = "cudagraph"

    times: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        for i in range(iters):
            with nvtx_range(f"{name}_{mode}_{i}", nvtx):
                start.record()
                run_fn()
                end.record()
                end.synchronize()
            times.append(start.elapsed_time(end))

    ordered = sorted(times)
    result = BenchResult(
        name=name,
        mode=mode,
        batch=batch,
        image_size=image_size,
        patch_size=patch_size,
        tokens=tokens,
        params_m=params_m,
        warmup=warmup,
        iters=iters,
        median_ms=statistics.median(ordered),
        mean_ms=statistics.fmean(ordered),
        p10_ms=percentile(ordered, 0.10),
        p90_ms=percentile(ordered, 0.90),
        min_ms=ordered[0],
        max_ms=ordered[-1],
    )
    print(
        f"{name:12s} {mode:10s} batch={batch:<2d} tokens={tokens:<5d} "
        f"median={result.median_ms:.3f} ms mean={result.mean_ms:.3f} ms "
        f"p10={result.p10_ms:.3f} p90={result.p90_ms:.3f}"
    )
    return result


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
        dino_head_dim=args.clipdino_dino_head_dim,
        dino_out_dim=args.clipdino_dino_out_dim,
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
    parser = argparse.ArgumentParser(description="Vision PyTorch trace + CUDA Graph benchmark")
    parser.add_argument("--component", choices=["vit", "clipdino", "all"], default="all")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--torch-trace", action="store_true", help="Run torch.jit.trace before benchmarking")
    parser.add_argument("--cuda-graph", action="store_true", help="Capture and replay the forward with CUDA Graph")
    parser.add_argument("--no-nvtx", action="store_true", help="Disable NVTX ranges around timed iterations")
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
    parser.add_argument("--clipdino-dino-head-dim", type=int, default=2048)
    parser.add_argument("--clipdino-dino-out-dim", type=int, default=1024)
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
    print("Vision Benchmark: PyTorch trace + CUDA Graph")
    print(f"component={args.component} dtype={args.dtype} device={args.device}")
    print(f"torch_trace={args.torch_trace} cuda_graph={args.cuda_graph}")
    print(
        "CLIP config source: "
        f"{CLIP_VIT_B16_CONFIG['source']} "
        f"(bench image_size={args.clipdino_image_size})"
    )
    print("=" * 80)

    results: list[BenchResult] = []
    nvtx_enabled = not args.no_nvtx

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
        results.append(
            bench_callable(
                "vit",
                run_vit,
                args.vit_batch,
                args.vit_image_size,
                args.vit_patch_size,
                tokens,
                params_m,
                args.warmup,
                args.iters,
                args.cuda_graph,
                nvtx_enabled,
            )
        )
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
            f"hidden={CLIP_VIT_B16_CONFIG['hidden_size']} params={params_m:.1f}M"
        )
        results.append(
            bench_callable(
                "clipdino",
                run_clipdino,
                args.clipdino_batch,
                args.clipdino_image_size,
                CLIP_VIT_B16_CONFIG["patch_size"],
                tokens,
                params_m,
                args.warmup,
                args.iters,
                args.cuda_graph,
                nvtx_enabled,
            )
        )
        del model, example, holder
        torch.cuda.empty_cache()

    print("\nResults")
    print("-" * 80)
    for result in results:
        print(
            f"{result.name:12s} mode={result.mode:10s} "
            f"median={result.median_ms:.3f} ms mean={result.mean_ms:.3f} ms"
        )

    if args.output_json:
        output = {
            "args": vars(args),
            "clip_vit_b16_config": CLIP_VIT_B16_CONFIG,
            "results": [asdict(r) for r in results],
        }
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved JSON: {path}")


if __name__ == "__main__":
    main()
