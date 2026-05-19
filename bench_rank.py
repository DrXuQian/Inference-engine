#!/usr/bin/env python3
"""
Benchmark a TP-split rank with vLLM using controlled token IDs.

For vocab-parallel split models, all input token IDs must be < vocab_size
in the split config (e.g. 124160 for TP=2 of 248320).

Usage:
    python bench_rank.py --model /path/to/rank_0
    python bench_rank.py --model /path/to/rank_0 --num-prompts 32 --input-len 256 --output-len 128
"""

import argparse
import json
import time

from vllm import LLM, SamplingParams


def make_safe_prompts(
    num_prompts: int,
    input_len: int,
    vocab_size: int,
    safe_token_id: int = 256,
) -> list[dict]:
    """Generate TokensPrompt dicts with IDs guaranteed < vocab_size."""
    assert safe_token_id < vocab_size, (
        f"safe_token_id {safe_token_id} >= vocab_size {vocab_size}"
    )
    return [{"prompt_token_ids": [safe_token_id] * input_len}
            for _ in range(num_prompts)]


def main():
    ap = argparse.ArgumentParser(description="Benchmark split rank with vLLM")
    ap.add_argument("--model", required=True, help="Path to rank directory")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=512)
    ap.add_argument("--gpu-mem", type=float, default=0.9,
                    help="gpu_memory_utilization")
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    args = ap.parse_args()

    # Read vocab_size from split config
    with open(f"{args.model}/config.json") as f:
        cfg = json.load(f)
    vocab_size = cfg.get("text_config", cfg)["vocab_size"]
    print(f"vocab_size in config: {vocab_size}")

    # Build prompts as token ID lists (bypass tokenizer entirely)
    prompts = make_safe_prompts(args.num_prompts, args.input_len, vocab_size)

    sampling = SamplingParams(
        max_tokens=args.output_len,
        temperature=0.0,
        ignore_eos=True,  # force full output_len generation
    )

    print(f"Loading model from {args.model} ...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="auto",
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_mem,
    )

    # Warmup
    print("Warmup ...")
    llm.generate(prompts[:1], sampling_params=sampling)

    # Benchmark
    print(f"Benchmarking: {args.num_prompts} prompts, "
          f"input_len={args.input_len}, output_len={args.output_len}")
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling)
    elapsed = time.perf_counter() - t0

    total_input = args.num_prompts * args.input_len
    total_output = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_tokens = total_input + total_output

    print(f"\n{'='*50}")
    print(f"Elapsed:          {elapsed:.2f}s")
    print(f"Total input:      {total_input} tokens")
    print(f"Total output:     {total_output} tokens")
    print(f"Throughput:       {total_tokens / elapsed:.1f} tok/s")
    print(f"Output tok/s:     {total_output / elapsed:.1f} tok/s")
    print(f"Latency/prompt:   {elapsed / args.num_prompts * 1000:.1f} ms")


if __name__ == "__main__":
    main()
