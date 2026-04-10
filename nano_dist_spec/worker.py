"""Distributed worker for Tensor-Parallel inference.

When using multiple GPUs, each rank runs a Worker that:
  1. Holds its own shard of the model (1/N of attention heads & MLP).
  2. Participates in AllReduce communication via NCCL.
  3. Executes the same sequence of operations in lock-step with other workers
     (SPMD — Single Program, Multiple Data).

Launch with:
    torchrun --nproc_per_node=N -m nano_dist_spec.worker --model /path/to/model

Rank 0 acts as the controller: it tokenizes input, broadcasts token ids and
positions to all ranks, and collects the final logits for sampling.
"""

import argparse
import os
from typing import List, Optional

import torch
import torch.distributed as dist

from .config import CacheConfig, ModelConfig
from .engine import LLMEngine
from .sampling import SamplingParams


def init_distributed():
    """Initialize torch.distributed with NCCL backend."""
    if dist.is_initialized():
        return
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)


def run_worker(
    model_path: str,
    prompts: Optional[List[str]] = None,
    sampling_params: Optional[SamplingParams] = None,
    num_gpu_blocks: Optional[int] = None,
):
    """Entry point for each TP worker process."""
    init_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cache_cfg = CacheConfig(num_gpu_blocks=num_gpu_blocks)
    engine = LLMEngine(
        model_path=model_path,
        tp_size=world_size,
        dtype=torch.float16,
        device=f"cuda:{rank}",
        cache_config=cache_cfg,
    )

    if rank == 0 and prompts:
        params = sampling_params or SamplingParams()
        outputs = engine.generate(prompts, params)
        for out in outputs:
            print(f"Prompt: {out.prompt}")
            print(f"Output: {out.text}\n")

    if dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description="nano-dist-spec TP Worker")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-gpu-blocks", type=int, default=None)
    args = parser.parse_args()

    params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    run_worker(
        model_path=args.model,
        prompts=[args.prompt],
        sampling_params=params,
        num_gpu_blocks=args.num_gpu_blocks,
    )


if __name__ == "__main__":
    main()
