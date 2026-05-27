#!/usr/bin/env python3
"""
Find the batch size at which compensated per-request TPS drops to 50%.

Sweeps concurrency from 1 to max, measures TPOT at each level,
applies layer-scale compensation, finds 50% TPS drop point.

Usage:
    # Server already running:
    python find_critical_batch.py \
        --model /path/to/model \
        --input-len 4096 --output-len 128 \
        --base-url http://127.0.0.1:8000

    # Auto start/stop server:
    python find_critical_batch.py \
        --model /path/to/model \
        --input-len 4096 --output-len 128 \
        --start-server --gpu-mem 0.9

    # With compensation (from split_meta.json or explicit):
    python find_critical_batch.py \
        --model /path/to/pruned_model \
        --input-len 4096 --output-len 128 \
        --base-url http://127.0.0.1:8000 \
        --tail-ms 0.15 --pruned-layers 8 --original-layers 48
"""

import argparse
import json
import os
import subprocess
import sys
import time

import requests


def wait_for_server(base_url: str, server_proc=None) -> bool:
    """Block until server /health returns 200. Only fail if process dies."""
    t0 = time.time()
    while True:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                elapsed = int(time.time() - t0)
                print(f"  Server ready ({elapsed}s)")
                return True
        except Exception:
            pass
        if server_proc and server_proc.poll() is not None:
            print(f"  Server process exited with code {server_proc.returncode}")
            return False
        elapsed = int(time.time() - t0)
        if elapsed > 0 and elapsed % 30 == 0:
            print(f"  Waiting for server... ({elapsed}s)")
        time.sleep(2)


def parse_metrics(text: str) -> dict:
    """Parse TPOT/TTFT/throughput from vllm bench serve output."""
    tpot = ttft = throughput = None
    for line in text.split("\n"):
        if "warning" in line.lower() or "Warning" in line:
            continue
        ll = line.lower().strip()
        if "median" in ll and ("tpot" in ll or "inter-token" in ll):
            try:
                tpot = float(line.split(":")[-1].strip().replace("ms", "").strip())
            except ValueError:
                pass
        if "median" in ll and "ttft" in ll:
            try:
                ttft = float(line.split(":")[-1].strip().replace("ms", "").strip())
            except ValueError:
                pass
        if "request throughput" in ll:
            try:
                throughput = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass
    return {"tpot_ms": tpot, "ttft_ms": ttft, "throughput_rps": throughput}


def bench_concurrency(base_url: str, model: str, input_len: int,
                      output_len: int, concurrency: int,
                      num_prompts: int = 10) -> dict:
    """Run vllm bench serve at given concurrency."""
    n = max(num_prompts, concurrency + 2)
    cmd = [
        "vllm", "bench", "serve",
        "--model", model,
        "--base-url", base_url,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(n),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--trust-remote-code",
    ]
    env = os.environ.copy()
    env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    output = result.stdout + "\n" + result.stderr
    m = parse_metrics(output)
    m["concurrency"] = concurrency
    m["raw_output"] = output
    return m


def load_compensation(model_dir: str, args) -> dict:
    """Load compensation params from split_meta.json or CLI args."""
    tail_ms = args.tail_ms
    pruned = args.pruned_layers
    original = args.original_layers

    # Try split_meta.json
    for d in [model_dir, os.path.dirname(model_dir)]:
        meta_path = os.path.join(d, "split_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if pruned is None:
                pruned = meta.get("pruned_layers")
            if original is None:
                original = meta.get("original_layers")
            break

    # Defaults
    if pruned is None or original is None:
        pruned = pruned or 1
        original = original or pruned

    layer_scale = original / pruned if pruned > 0 else 1
    if tail_ms is None:
        tail_ms = 0  # no trace available

    return {
        "tail_ms": tail_ms,
        "pruned": pruned,
        "original": original,
        "layer_scale": layer_scale,
    }


def compensate_tpot(raw_tpot: float, comp: dict) -> float:
    """Apply layer-scale compensation to raw TPOT.
    comp_TPOT = (raw - tail) × scale + tail
    """
    encoder = max(raw_tpot - comp["tail_ms"], 0)
    return encoder * comp["layer_scale"] + comp["tail_ms"]


def main():
    ap = argparse.ArgumentParser(description="Find critical batch size (TPS drops to 50%)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--start-server", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--port", type=int, default=8000)
    # Compensation params
    ap.add_argument("--tail-ms", type=float, default=None,
                    help="Tail time (lm_head+sampling) from trace. 0 = no compensation.")
    ap.add_argument("--pruned-layers", type=int, default=None)
    ap.add_argument("--original-layers", type=int, default=None)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    base_url = args.base_url
    if args.start_server:
        base_url = f"http://127.0.0.1:{args.port}"

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Load compensation
    comp = load_compensation(args.model, args)
    has_comp = comp["layer_scale"] > 1

    # Start server if requested
    server_proc = None
    if args.start_server:
        max_model_len = args.input_len + args.output_len + 64
        env = os.environ.copy()
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        server_proc = subprocess.Popen(
            ["vllm", "serve", args.model,
             "--host", "127.0.0.1", "--port", str(args.port),
             "--tensor-parallel-size", "1",
             "--max-model-len", str(max_model_len),
             "--trust-remote-code", "--no-enable-prefix-caching",
             "--gpu-memory-utilization", str(args.gpu_mem)],
            env=env,
        )
        print(f"Starting server (port {args.port})...")

    # Always wait for server to be ready before benchmarking
    print(f"Waiting for server at {base_url} ...")
    if not wait_for_server(base_url, server_proc):
        print("ERROR: server not available")
        sys.exit(1)

    try:
        print(f"\nModel: {args.model}")
        print(f"Input: {args.input_len}, Output: {args.output_len}")
        print(f"Batch sizes: {batch_sizes}")
        if has_comp:
            print(f"Compensation: layers {comp['pruned']}/{comp['original']} "
                  f"(scale={comp['layer_scale']:.2f}x), tail={comp['tail_ms']:.4f}ms")
        print()

        header = f"{'batch':>6} {'raw_tpot':>10}"
        if has_comp:
            header += f" {'comp_tpot':>10} {'comp_tps':>10}"
        else:
            header += f" {'TPS':>10}"
        header += f" {'TPS%':>8} {'throughput':>12}"
        print(header)
        print("-" * len(header))

        results = []
        baseline_tps = None
        critical_batch = None

        for bs in batch_sizes:
            r = bench_concurrency(base_url, args.model, args.input_len,
                                  args.output_len, bs, args.num_prompts)

            if r["tpot_ms"] is None:
                print(f"{bs:>6} {'FAIL':>10}")
                useful = [l for l in r["raw_output"].split("\n")
                          if l.strip() and "warning" not in l.lower()]
                for line in useful[-5:]:
                    print(f"       {line.strip()[:120]}")
                results.append({"concurrency": bs, "error": "no tpot"})
                continue

            raw_tpot = r["tpot_ms"]
            if has_comp:
                comp_tpot = compensate_tpot(raw_tpot, comp)
            else:
                comp_tpot = raw_tpot

            tps = round(1000 / comp_tpot, 1) if comp_tpot > 0 else 0

            if baseline_tps is None:
                baseline_tps = tps

            tps_pct = tps / baseline_tps * 100 if baseline_tps else 0
            tp_str = f"{r['throughput_rps']:.2f} rps" if r["throughput_rps"] else ""

            row = f"{bs:>6} {raw_tpot:>10.2f}"
            if has_comp:
                row += f" {comp_tpot:>10.2f} {tps:>10.1f}"
            else:
                row += f" {tps:>10.1f}"
            row += f" {tps_pct:>7.1f}% {tp_str:>12}"
            print(row)

            if critical_batch is None and tps_pct <= 50:
                critical_batch = bs

            results.append({
                "concurrency": bs,
                "raw_tpot_ms": raw_tpot,
                "comp_tpot_ms": comp_tpot if has_comp else raw_tpot,
                "tps": tps,
                "tps_pct": round(tps_pct, 1),
                "throughput_rps": r["throughput_rps"],
            })

        print()
        if critical_batch:
            print(f"Critical batch size (TPS <= 50%): {critical_batch}")
        elif baseline_tps:
            print(f"TPS did not drop to 50% within tested range (max batch={batch_sizes[-1]})")
        else:
            print("ERROR: could not measure baseline TPS")

        if args.output_json:
            output = {
                "model": args.model,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "compensation": comp if has_comp else None,
                "baseline_tps": baseline_tps,
                "critical_batch": critical_batch,
                "results": results,
            }
            with open(args.output_json, "w") as f:
                json.dump(output, f, indent=2)
            print(f"Saved to {args.output_json}")

    finally:
        if server_proc:
            print("\nStopping server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    main()
