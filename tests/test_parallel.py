"""Tests for tensor parallelism primitives (CPU, tp_size=1)."""

import torch
import torch.nn as nn
from nano_dist_spec.parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    tensor_split,
)


def test_tensor_split():
    t = torch.arange(24).reshape(4, 6)
    s0 = tensor_split(t, rank=0, world_size=2, dim=0)
    s1 = tensor_split(t, rank=1, world_size=2, dim=0)
    assert s0.shape == (2, 6)
    assert s1.shape == (2, 6)
    assert torch.equal(torch.cat([s0, s1], dim=0), t)

    s0c = tensor_split(t, rank=0, world_size=3, dim=1)
    assert s0c.shape == (4, 2)


def test_column_parallel_linear():
    col = ColumnParallelLinear(8, 16, bias=False, tp_size=1)
    x = torch.randn(2, 8)
    y = col(x)
    assert y.shape == (2, 16)

    col2 = ColumnParallelLinear(8, 16, bias=False, tp_size=2)
    y2 = col2(x)
    assert y2.shape == (2, 8)  # 16 / 2 = 8 per rank


def test_row_parallel_linear():
    row = RowParallelLinear(16, 8, bias=False, tp_size=1)
    x = torch.randn(2, 16)
    y = row(x)
    assert y.shape == (2, 8)

    row2 = RowParallelLinear(16, 8, bias=False, tp_size=2)
    x2 = torch.randn(2, 8)  # input per rank = 16/2 = 8
    y2 = row2(x2)
    assert y2.shape == (2, 8)


def test_vocab_parallel_embedding():
    emb = VocabParallelEmbedding(100, 32, tp_size=1)
    ids = torch.tensor([0, 50, 99])
    out = emb(ids)
    assert out.shape == (3, 32)


def test_column_row_compose():
    """Column -> Row should produce the same shape as a single Linear."""
    in_dim, mid_dim, out_dim = 16, 32, 16
    col = ColumnParallelLinear(in_dim, mid_dim, tp_size=1)
    row = RowParallelLinear(mid_dim, out_dim, tp_size=1)
    x = torch.randn(4, in_dim)
    y = row(col(x))
    assert y.shape == (4, out_dim)


if __name__ == "__main__":
    test_tensor_split()
    test_column_parallel_linear()
    test_row_parallel_linear()
    test_vocab_parallel_embedding()
    test_column_row_compose()
    print("All parallel tests passed!")
