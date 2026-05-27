#!/usr/bin/env python3
"""
Find the batch size at which per-request TPS drops to 50% of batch=1.

Sweeps concurrency from 1 to max, measures TPOT at each level.
Per-request TPS = 1000 / TPOT. Finds where it drops to half of baseline.

Usage:
    # Start server first, then run:
    python find_critical_batch.py \
        --model /path/to/model \
        --input-len 102400 --output-len 1024 \
        --base-url http://127.0.0.1:8000

    # Or let the script start/stop the server:
    python find_critical_batch.py \
        --model /path/to/model \
        --input-len 102400 --output-len 1024 \
        --start-server --gpu-mem 0.9

    # Custom batch sizes:
    python find_critical_batch.py \
        --model /path/to/model \
        --input-len 102400 --output-len 1024 \
        --batch-sizes 1,2,4,8,16,32,64
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import requests


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def bench_concurrency(base_url: str, model: str, input_len: int,
                      output_len: int, concurrency: int,
                      num_prompts: int = 10) -> dict:
    """Run benchmark at given concurrency, return TPOT stats."""
    # Use at least concurrency+2 prompts for stable measurement
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

    # Parse output for TPOT
    tpot = None
    ttft = None
    throughput = None
    for line in result.stdout.split("\n"):
        line = line.strip()
        # Match: "Median TPOT (ms): 9.09" or "Median Inter-token Latency: 9.09 ms"
        if "median" in line.lower() and ("tpot" in line.lower() or "inter-token" in line.lower()):
            try:
                tpot = float(line.split(":")[-1].strip().replace("ms", "").strip())
            except ValueError:
                pass
        # Match: "Median TTFT (ms): 4425.85"
        if "median" in line.lower() and "ttft" in line.lower():
            try:
                ttft = float(line.split(":")[-1].strip().replace("ms", "").strip())
            except ValueError:
                pass
        if "request throughput" in line.lower():
            try:
                throughput = float(line.split(":")[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass

    # Also try parsing JSON output if available
    for line in result.stdout.split("\n"):
        line = line.strip()
        if line.startswith("{") and "tpot" in line.lower():
            try:
                d = json.loads(line)
                tpot = tpot or d.get("median_tpot_ms") or d.get("tpot_median_ms")
                ttft = ttft or d.get("median_ttft_ms") or d.get("ttft_median_ms")
            except json.JSONDecodeError:
                pass

    return {
        "concurrency": concurrency,
        "tpot_ms": tpot,
        "ttft_ms": ttft,
        "throughput_rps": throughput,
        "tps": round(1000 / tpot, 1) if tpot and tpot > 0 else None,
        "stdout": result.stdout[-500:] if not tpot else "",
    }


def main():
    ap = argparse.ArgumentParser(description="Find critical batch size (TPS drops to 50%)")
    ap.add_argument("--model", required=True, help="Model path")
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, default=128,
                    help="Output len per request (default: 128, keep short for faster sweep)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128",
                    help="Comma-separated batch sizes to test")
    ap.add_argument("--num-prompts", type=int, default=10,
                    help="Number of prompts per batch size test")
    ap.add_argument("--start-server", action="store_true",
                    help="Start vllm serve automatically")
    ap.add_argument("--gpu-mem", type=float, default=0.9)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    base_url = args.base_url
    if args.start_server:
        base_url = f"http://127.0.0.1:{args.port}"

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

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
        if not wait_for_server(base_url):
            print("ERROR: server failed to start")
            server_proc.kill()
            sys.exit(1)
        print("Server ready\n")

    try:
        print(f"Model: {args.model}")
        print(f"Input: {args.input_len}, Output: {args.output_len}")
        print(f"Batch sizes: {batch_sizes}")
        print()
        print(f"{'batch':>6} {'TPOT(ms)':>10} {'TPS':>8} {'TPS%':>8} {'throughput':>12}")
        print("-" * 50)

        results = []
        baseline_tps = None
        critical_batch = None

        for bs in batch_sizes:
            r = bench_concurrency(base_url, args.model, args.input_len,
                                  args.output_len, bs, args.num_prompts)
            results.append(r)

            if r["tps"] is None:
                print(f"{bs:>6} {'FAIL':>10} {'':>8} {'':>8} {'':>12}")
                if r["stdout"]:
                    print(f"       Last output: {r['stdout'][:200]}")
                continue

            if baseline_tps is None:
                baseline_tps = r["tps"]

            tps_pct = r["tps"] / baseline_tps * 100 if baseline_tps else 0
            tp_str = f"{r['throughput_rps']:.2f} rps" if r["throughput_rps"] else ""

            print(f"{bs:>6} {r['tpot_ms']:>10.2f} {r['tps']:>8.1f} {tps_pct:>7.1f}% {tp_str:>12}")

            if critical_batch is None and tps_pct <= 50:
                critical_batch = bs

        print()
        if critical_batch:
            print(f"Critical batch size (TPS ≤ 50%): {critical_batch}")
        elif baseline_tps:
            print(f"TPS did not drop to 50% within tested range (max batch={batch_sizes[-1]})")
        else:
            print("ERROR: could not measure baseline TPS")

        # Save results
        if args.output_json:
            output = {
                "model": args.model,
                "input_len": args.input_len,
                "output_len": args.output_len,
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
