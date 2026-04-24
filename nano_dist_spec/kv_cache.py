"""Paged KV-Cache management.

Implements the core idea of PagedAttention: the KV cache for each sequence is
stored in fixed-size **blocks** (e.g. 16 tokens each). A block table maps
logical block indices to physical block indices, eliminating fragmentation.

Components:
  BlockAllocator - manages a pool of physical blocks (free-list).
  KVCache        - holds the pre-allocated GPU tensors for all layers.
  KVCacheManager - per-sequence book-keeping (block tables, context lengths,
                   slot mapping computation, and rollback for speculative
                   decoding).
"""

from typing import Dict, List, Optional, Tuple

import torch

from .debug import tracer


class BlockAllocator:
    """Free-list allocator for physical KV cache blocks."""

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_blocks: List[int] = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def allocate(self) -> int:
        if not self.free_blocks:
            raise RuntimeError("KV cache out of blocks")
        block_id = self.free_blocks.pop()
        tracer.on_allocate(block_id, len(self.free_blocks), self.num_blocks)
        return block_id

    def free(self, block_id: int) -> None:
        self.free_blocks.append(block_id)
        tracer.on_free(block_id, len(self.free_blocks), self.num_blocks)


class KVCache:
    """Pre-allocated KV cache tensors on GPU, one pair per layer."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size

        total_slots = num_blocks * block_size
        self.key_caches: List[torch.Tensor] = []
        self.value_caches: List[torch.Tensor] = []
        for _ in range(num_layers):
            self.key_caches.append(
                torch.zeros(total_slots, num_kv_heads, head_dim,
                            device=device, dtype=dtype)
            )
            self.value_caches.append(
                torch.zeros(total_slots, num_kv_heads, head_dim,
                            device=device, dtype=dtype)
            )

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.key_caches[layer_idx], self.value_caches[layer_idx]

    def memory_bytes(self) -> int:
        """Total GPU memory occupied by the KV cache."""
        per_tensor = (
            self.num_blocks * self.block_size * self.num_kv_heads
            * self.head_dim * 2  # bytes per float16
        )
        return per_tensor * 2 * self.num_layers  # K + V, all layers


class KVCacheManager:
    """Per-sequence block-table management on top of BlockAllocator + KVCache.

    Tracks which physical blocks are assigned to each sequence and computes
    the slot_mapping / block_tables tensors needed by the attention kernels.
    """

    def __init__(self, block_size: int, allocator: BlockAllocator):
        self.block_size = block_size
        self.allocator = allocator
        self.block_tables: Dict[int, List[int]] = {}  # seq_id -> [phys_block_id, ...]
        self.context_lens: Dict[int, int] = {}         # seq_id -> current context length

    def allocate_seq(self, seq_id: int, num_tokens: int) -> None:
        """Allocate blocks for a new sequence of *num_tokens* (prompt length)."""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        blocks = [self.allocator.allocate() for _ in range(num_blocks)]
        self.block_tables[seq_id] = blocks
        self.context_lens[seq_id] = num_tokens
        tracer.on_allocate_seq(
            seq_id, num_tokens, blocks,
            self.allocator.num_free, self.allocator.num_blocks,
        )

    def append_slots(self, seq_id: int, num_new: int = 1) -> None:
        """Ensure there are cache slots for *num_new* additional tokens."""
        old_len = self.context_lens[seq_id]
        new_len = old_len + num_new
        old_blocks = (old_len + self.block_size - 1) // self.block_size
        new_blocks = (new_len + self.block_size - 1) // self.block_size
        last_new_block: Optional[int] = None
        for _ in range(new_blocks - old_blocks):
            blk = self.allocator.allocate()
            self.block_tables[seq_id].append(blk)
            last_new_block = blk
        self.context_lens[seq_id] = new_len
        tracer.on_append_slots(
            seq_id, old_len, new_len,
            self.block_tables[seq_id], self.block_size,
            last_new_block,
            self.allocator.num_free, self.allocator.num_blocks,
        )

    def rollback(self, seq_id: int, new_len: int) -> None:
        """Roll back a sequence's context to *new_len*, freeing tail blocks.

        Used after speculative decoding rejection to discard un-accepted KV.
        """
        old_len = self.context_lens[seq_id]
        if new_len >= old_len:
            return
        old_blocks = (old_len + self.block_size - 1) // self.block_size
        new_blocks = (new_len + self.block_size - 1) // self.block_size
        freed: List[int] = []
        for i in range(new_blocks, old_blocks):
            blk = self.block_tables[seq_id][i]
            self.allocator.free(blk)
            freed.append(blk)
        self.block_tables[seq_id] = self.block_tables[seq_id][:new_blocks]
        self.context_lens[seq_id] = new_len
        tracer.on_rollback(
            seq_id, old_len, new_len, freed,
            self.block_tables[seq_id],
            self.allocator.num_free, self.allocator.num_blocks,
        )

    def free_seq(self, seq_id: int) -> None:
        freed: List[int] = []
        for blk in self.block_tables.pop(seq_id, []):
            self.allocator.free(blk)
            freed.append(blk)
        self.context_lens.pop(seq_id, None)
        if freed:
            tracer.on_free_seq(
                seq_id, freed,
                self.allocator.num_free, self.allocator.num_blocks,
            )

    # ------------------------------------------------------------------
    # Compute tensors consumed by attention kernels
    # ------------------------------------------------------------------

    def compute_slot_mapping(
        self, seq_id: int, start_pos: int, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Slot indices for tokens at positions [start_pos .. start_pos+num_tokens)."""
        slots = []
        for pos in range(start_pos, start_pos + num_tokens):
            blk_idx = pos // self.block_size
            blk_off = pos % self.block_size
            phys = self.block_tables[seq_id][blk_idx]
            slots.append(phys * self.block_size + blk_off)
        tracer.on_slot_mapping(
            seq_id, start_pos, num_tokens, slots,
            self.block_tables[seq_id], self.block_size,
        )
        return torch.tensor(slots, dtype=torch.long, device=device)

    def get_block_table_tensor(
        self, seq_ids: List[int], device: torch.device
    ) -> torch.Tensor:
        """Padded block tables for a batch of sequences."""
        tables = [self.block_tables[s] for s in seq_ids]
        max_len = max(len(t) for t in tables)
        padded = [t + [0] * (max_len - len(t)) for t in tables]
        return torch.tensor(padded, dtype=torch.long, device=device)

    def get_context_lens_tensor(
        self, seq_ids: List[int], device: torch.device
    ) -> torch.Tensor:
        return torch.tensor(
            [self.context_lens[s] for s in seq_ids],
            dtype=torch.long, device=device,
        )
