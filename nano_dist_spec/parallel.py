"""Tensor parallelism primitives: ColumnParallel, RowParallel, VocabParallel.

In Tensor Parallelism (TP), we split the model's weight matrices across GPUs.
Each GPU holds a shard and communicates via AllReduce/AllGather to produce
results identical to a single-GPU execution.

Key insight for Transformer TP:
  - MLP: gate_proj/up_proj use ColumnParallel (split output dim),
          down_proj uses RowParallel (split input dim, AllReduce output).
  - Attention: Q/K/V projections use ColumnParallel (each GPU handles a
               subset of heads), O projection uses RowParallel.
  - This arrangement requires only ONE AllReduce per Attention layer and
    ONE AllReduce per MLP layer (both in RowParallel.forward).
"""

import torch
import torch.nn as nn
import torch.distributed as dist


def get_tp_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_tp_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def tensor_split(tensor: torch.Tensor, rank: int, world_size: int, dim: int = 0):
    """Split a tensor along `dim` and return the shard for `rank`."""
    assert tensor.shape[dim] % world_size == 0, (
        f"Cannot split dim {dim} of size {tensor.shape[dim]} into {world_size} shards"
    )
    chunk_size = tensor.shape[dim] // world_size
    return tensor.narrow(dim, rank * chunk_size, chunk_size).contiguous()


# ---------------------------------------------------------------------------
# Communication primitives
# ---------------------------------------------------------------------------

class _AllReduceFunc(torch.autograd.Function):
    """AllReduce: sum partial results from all TP ranks (in-place)."""

    @staticmethod
    def forward(ctx, input_: torch.Tensor) -> torch.Tensor:
        if get_tp_world_size() == 1:
            return input_
        dist.all_reduce(input_)
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    return _AllReduceFunc.apply(x)


def all_gather_last_dim(x: torch.Tensor) -> torch.Tensor:
    """AllGather along the last dimension."""
    world_size = get_tp_world_size()
    if world_size == 1:
        return x
    gathered = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=-1)


# ---------------------------------------------------------------------------
# Parallel layers
# ---------------------------------------------------------------------------

class ColumnParallelLinear(nn.Module):
    """Linear layer split along the **output** dimension.

    Each GPU stores W[:, rank*chunk : (rank+1)*chunk].
    Forward: y_local = x @ W_shard  (no communication needed).
    The output is a partition of the full output and is typically consumed
    by a subsequent RowParallelLinear.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, tp_size: int = 1):
        super().__init__()
        self.tp_size = tp_size
        assert out_features % tp_size == 0
        self.out_per_rank = out_features // tp_size
        self.linear = nn.Linear(in_features, self.out_per_rank, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class RowParallelLinear(nn.Module):
    """Linear layer split along the **input** dimension.

    Each GPU stores W[rank*chunk : (rank+1)*chunk, :].
    Forward: y_partial = x_shard @ W_shard, then AllReduce to get full y.
    This layer performs the only communication in each sub-layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = False, tp_size: int = 1):
        super().__init__()
        self.tp_size = tp_size
        assert in_features % tp_size == 0
        self.in_per_rank = in_features // tp_size
        self.linear = nn.Linear(self.in_per_rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.tp_size > 1:
            out = all_reduce(out)
        return out


class VocabParallelEmbedding(nn.Module):
    """Embedding split along the vocabulary dimension.

    Each GPU stores embeddings for vocab[rank*chunk : (rank+1)*chunk].
    Out-of-range token ids produce zero vectors; AllReduce merges results.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, tp_size: int = 1):
        super().__init__()
        self.tp_size = tp_size
        self.num_embeddings = num_embeddings
        pad = (tp_size - num_embeddings % tp_size) % tp_size
        self.padded_vocab = num_embeddings + pad
        self.per_rank = self.padded_vocab // tp_size
        self.rank = get_tp_rank()
        self.start = self.rank * self.per_rank
        self.end = self.start + self.per_rank
        self.embedding = nn.Embedding(self.per_rank, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size == 1:
            return self.embedding(x)
        mask = (x >= self.start) & (x < self.end)
        local_ids = (x - self.start).clamp(0, self.per_rank - 1)
        out = self.embedding(local_ids)
        out[~mask] = 0.0
        return all_reduce(out)
