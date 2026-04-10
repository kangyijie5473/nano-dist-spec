"""Basic single-GPU inference example.

Usage:
    python examples/basic_inference.py --model /path/to/Qwen3-0.6B
"""

import argparse

from nano_dist_spec import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    llm = LLM(args.model, dtype="float16")

    prompts = [
        "Explain distributed inference in one paragraph:",
        "What is speculative decoding?",
        "Write a Python function that computes fibonacci numbers:",
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(f"Prompt:  {output.prompt}")
        print(f"Output:  {output.text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
