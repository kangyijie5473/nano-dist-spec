"""Tests for KV cache block allocation and management."""

import torch
from nano_dist_spec.kv_cache import BlockAllocator, KVCache, KVCacheManager


def test_block_allocator():
    alloc = BlockAllocator(num_blocks=8)
    assert alloc.num_free == 8

    b0 = alloc.allocate()
    b1 = alloc.allocate()
    assert alloc.num_free == 6
    assert b0 != b1

    alloc.free(b0)
    assert alloc.num_free == 7

    b2 = alloc.allocate()
    assert b2 == b0  # LIFO reuse


def test_kv_cache_shape():
    cache = KVCache(
        num_layers=2, num_kv_heads=4, head_dim=8,
        num_blocks=10, block_size=4, device=torch.device("cpu"),
    )
    k, v = cache.get_kv(0)
    assert k.shape == (40, 4, 8)  # 10 * 4 = 40 slots
    assert v.shape == (40, 4, 8)


def test_kv_cache_manager_allocate():
    alloc = BlockAllocator(32)
    mgr = KVCacheManager(block_size=4, allocator=alloc)

    mgr.allocate_seq(seq_id=0, num_tokens=10)
    assert mgr.context_lens[0] == 10
    assert len(mgr.block_tables[0]) == 3  # ceil(10/4) = 3

    mgr.allocate_seq(seq_id=1, num_tokens=4)
    assert len(mgr.block_tables[1]) == 1


def test_kv_cache_manager_append():
    alloc = BlockAllocator(32)
    mgr = KVCacheManager(block_size=4, allocator=alloc)

    mgr.allocate_seq(0, 4)
    assert len(mgr.block_tables[0]) == 1

    mgr.append_slots(0, 1)
    assert mgr.context_lens[0] == 5
    assert len(mgr.block_tables[0]) == 2  # needs new block


def test_kv_cache_manager_rollback():
    alloc = BlockAllocator(32)
    mgr = KVCacheManager(block_size=4, allocator=alloc)

    mgr.allocate_seq(0, 12)
    assert len(mgr.block_tables[0]) == 3
    free_before = alloc.num_free

    mgr.rollback(0, 5)
    assert mgr.context_lens[0] == 5
    assert len(mgr.block_tables[0]) == 2  # ceil(5/4) = 2
    assert alloc.num_free == free_before + 1  # freed 1 block


def test_slot_mapping():
    alloc = BlockAllocator(32)
    mgr = KVCacheManager(block_size=4, allocator=alloc)

    mgr.allocate_seq(0, 6)
    slots = mgr.compute_slot_mapping(0, 0, 6, torch.device("cpu"))
    assert slots.shape == (6,)
    assert len(set(slots.tolist())) == 6  # all unique


def test_block_table_tensor():
    alloc = BlockAllocator(32)
    mgr = KVCacheManager(block_size=4, allocator=alloc)

    mgr.allocate_seq(0, 8)
    mgr.allocate_seq(1, 4)

    bt = mgr.get_block_table_tensor([0, 1], torch.device("cpu"))
    assert bt.shape == (2, 2)  # padded to max blocks

    cl = mgr.get_context_lens_tensor([0, 1], torch.device("cpu"))
    assert cl.tolist() == [8, 4]


def test_write_read_kv():
    """End-to-end: allocate, write KV, read back via slot mapping."""
    cache = KVCache(
        num_layers=1, num_kv_heads=2, head_dim=4,
        num_blocks=8, block_size=4, device=torch.device("cpu"),
        dtype=torch.float32,
    )
    alloc = BlockAllocator(8)
    mgr = KVCacheManager(block_size=4, allocator=alloc)

    mgr.allocate_seq(0, 3)
    slots = mgr.compute_slot_mapping(0, 0, 3, torch.device("cpu"))

    k_data = torch.randn(3, 2, 4)
    v_data = torch.randn(3, 2, 4)

    key_cache, value_cache = cache.get_kv(0)
    key_cache[slots] = k_data
    value_cache[slots] = v_data

    assert torch.allclose(key_cache[slots], k_data)
    assert torch.allclose(value_cache[slots], v_data)


if __name__ == "__main__":
    test_block_allocator()
    test_kv_cache_shape()
    test_kv_cache_manager_allocate()
    test_kv_cache_manager_append()
    test_kv_cache_manager_rollback()
    test_slot_mapping()
    test_block_table_tensor()
    test_write_read_kv()
    print("All KV cache tests passed!")
