"""Multi-GPU tensor parallel inference example.

Usage:
    torchrun --nproc_per_node=2 examples/tensor_parallel.py --model /path/to/model

Each GPU holds 1/N of the attention heads and MLP weights. Communication
happens via AllReduce at each RowParallelLinear layer.
"""

import argparse

import torch.distributed as dist

from nano_dist_spec import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    llm = LLM(
        args.model,
        tensor_parallel_size=world_size,
        dtype="float16",
        device=f"cuda:{rank}",
    )

    prompts = ["Explain tensor parallelism in LLM inference:"]
    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    outputs = llm.generate(prompts, params)

    if rank == 0:
        for out in outputs:
            print(f"Prompt: {out.prompt}")
            print(f"Output: {out.text}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
