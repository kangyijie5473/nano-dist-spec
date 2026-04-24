"""Multi-GPU tensor parallel inference example.

Usage:
    torchrun --nproc_per_node=2 examples/tensor_parallel.py --model /path/to/model

Each GPU holds 1/N of the attention heads and MLP weights. Communication
happens via AllReduce at each RowParallelLinear layer.
"""

import argparse

import torch.distributed as dist
from transformers import AutoTokenizer

from nano_dist_spec import LLM, SamplingParams


def build_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
    """Wrap raw user text in the model's chat template when available.

    DeepSeek-R1 series additionally requires `<think>\\n` to be prefilled
    after `<｜Assistant｜>` to reliably enter the reasoning chain.
    """
    if tokenizer.chat_template is None:
        return user_text

    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if prompt.endswith("<｜Assistant｜>"):
        prompt += "<think>\n"

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--prompt", type=str, default="Explain tensor parallelism in LLM inference:",
        help="Raw user question; chat template will be applied automatically.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = build_prompt(tokenizer, args.prompt)

    llm = LLM(
        args.model,
        tensor_parallel_size=world_size,
        dtype="float16",
        device=f"cuda:{rank}",
    )

    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    outputs = llm.generate([prompt], params)

    if rank == 0:
        for out in outputs:
            print(f"Prompt: {out.prompt}")
            print(f"Output: {out.text}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
