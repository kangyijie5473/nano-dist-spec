"""Speculative decoding example: small draft model + large target model.

Usage:
    python examples/speculative_decode.py \
        --target /path/to/Qwen3-1.7B \
        --draft  /path/to/Qwen3-0.6B

The draft model generates K candidate tokens autoregressively, the target
model verifies them in a single forward pass, and rejection sampling ensures
the output distribution matches the target model exactly.
"""

import argparse
import time

from nano_dist_spec import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="Large target model")
    parser.add_argument("--draft", type=str, required=True, help="Small draft model")
    parser.add_argument("--K", type=int, default=5, help="Speculative tokens per round")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Target model: {args.target}")
    print(f"Draft model:  {args.draft}")
    print(f"Speculative tokens per round: {args.K}")
    print()

    llm = LLM(
        args.target,
        dtype="float16",
        draft_model_path=args.draft,
        num_speculative_tokens=args.K,
    )

    prompts = ["Explain how speculative decoding works in LLM inference:"]
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    start = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - start

    for out in outputs:
        n_tokens = len(out.token_ids)
        print(f"Prompt:  {out.prompt}")
        print(f"Output:  {out.text}")
        print(f"Tokens:  {n_tokens}")
        print(f"Time:    {elapsed:.2f}s")
        print(f"Speed:   {n_tokens / elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
