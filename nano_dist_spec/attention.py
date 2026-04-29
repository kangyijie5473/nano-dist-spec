"""Attention utilities: RoPE, prefill attention, and paged decode attention.

Three attention execution paths:
  1. Prefill  - process full prompt, causal mask, no cache read.
  2. Decode   - single new token, paged read from KV cache.
  3. Extend   - K new tokens with prefix in cache (used by speculative verify).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
):
    """Precompute cos/sin tables for RoPE.

    Returns:
        cos, sin: both [max_seq_len, head_dim]
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [max_seq_len, head_dim // 2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
):
    """Apply RoPE to query and key tensors.

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        cos, sin: [max_seq_len, head_dim]
        positions: [batch, seq_len]
    """
    cos = cos[positions].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin[positions].unsqueeze(1)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


# ---------------------------------------------------------------------------
# GQA head expansion
# ---------------------------------------------------------------------------

def expand_kv_for_gqa(
    k: torch.Tensor, v: torch.Tensor, num_kv_groups: int
):
    """Repeat KV heads to match the number of query heads (GQA).

    Args:
        k, v: [batch, num_kv_heads, seq_len, head_dim]
        num_kv_groups: num_query_heads // num_kv_heads
    Returns:
        k, v: [batch, num_heads, seq_len, head_dim]
    """
    if num_kv_groups == 1:
        return k, v
    bsz, nkv, slen, hdim = k.shape
    k = k[:, :, None, :, :].expand(bsz, nkv, num_kv_groups, slen, hdim)
    k = k.reshape(bsz, nkv * num_kv_groups, slen, hdim)
    v = v[:, :, None, :, :].expand(bsz, nkv, num_kv_groups, slen, hdim)
    v = v.reshape(bsz, nkv * num_kv_groups, slen, hdim)
    return k, v


# ---------------------------------------------------------------------------
# Attention metadata (passed through model forward)
# ---------------------------------------------------------------------------

@dataclass
class InputMetadata:
    """Carries KV-cache addressing info through the model forward pass."""
    slot_mapping: torch.Tensor        # [num_tokens] physical slot for each token
    block_tables: Optional[torch.Tensor] = None   # [batch, max_blocks_per_seq]
    context_lens: Optional[torch.Tensor] = None    # [batch] total ctx length incl. new tokens
    block_size: int = 16

    @property
    def is_prefill(self) -> bool:
        return self.block_tables is None


# ---------------------------------------------------------------------------
# Attention execution paths
# ---------------------------------------------------------------------------

def prefill_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_kv_groups: int
) -> torch.Tensor:
    """Standard causal attention for prefill (no KV cache read).

    Args:
        q: [batch, num_heads, seq_len, head_dim]
        k: [batch, num_kv_heads, seq_len, head_dim]
        v: same shape as k
    Returns:
        output: [batch, num_heads, seq_len, head_dim]
    """
    k, v = expand_kv_for_gqa(k, v, num_kv_groups)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def decode_paged_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    block_size: int,
    num_kv_groups: int,
) -> torch.Tensor:
    """Paged attention for decode: gather KV from block-table, compute attention.

    Demonstrates the core concept of PagedAttention — non-contiguous physical
    blocks are gathered via a per-sequence block table, eliminating memory
    fragmentation inherent in contiguous KV allocation.

    Args:
        q: [batch, num_heads, 1, head_dim]
        key_cache, value_cache: [total_slots, num_kv_heads, head_dim]
        block_tables: [batch, max_blocks_per_seq]
        context_lens: [batch]
        block_size: int
        num_kv_groups: num_heads // num_kv_heads
    Returns:
        output: [batch, num_heads, 1, head_dim]
    """
    batch_size, num_heads, _, head_dim = q.shape
    num_kv_heads = num_heads // num_kv_groups
    max_ctx = block_tables.shape[1] * block_size
    device, dtype = q.device, q.dtype

    # Vectorized gather over paged block tables. We intentionally avoid
    # Tensor.item()/Python loops to keep this path CUDA-graph friendly.
    positions = torch.arange(max_ctx, device=device, dtype=torch.long)
    blk_idx = positions // block_size          # [max_ctx]
    blk_off = positions % block_size           # [max_ctx]
    blk_idx = blk_idx.unsqueeze(0).expand(batch_size, -1)  # [batch, max_ctx]
    physical = torch.gather(block_tables, 1, blk_idx)      # [batch, max_ctx]
    slots = physical * block_size + blk_off.unsqueeze(0)   # [batch, max_ctx]

    flat_slots = slots.reshape(-1)
    padded_k = key_cache[flat_slots].reshape(batch_size, max_ctx, num_kv_heads, head_dim)
    padded_v = value_cache[flat_slots].reshape(batch_size, max_ctx, num_kv_heads, head_dim)

    # Transpose -> [batch, num_kv_heads, max_ctx, head_dim]
    k = padded_k.transpose(1, 2)
    v = padded_v.transpose(1, 2)
    k, v = expand_kv_for_gqa(k, v, num_kv_groups)

    # Build attention mask for variable context lengths
    pos_range = torch.arange(max_ctx, device=device).unsqueeze(0)       # [1, max_ctx]
    attn_mask = (pos_range >= context_lens.unsqueeze(1))                 # [batch, max_ctx]
    attn_mask = attn_mask[:, None, None, :]                              # [batch, 1, 1, max_ctx]

    scale = 1.0 / math.sqrt(head_dim)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale         # [batch, heads, 1, max_ctx]
    attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.matmul(attn_weights, v)


def extend_attention(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size: int,
    num_kv_groups: int,
) -> torch.Tensor:
    """Attention for speculative-decoding verify: new tokens attend to cached
    prefix **plus** each other (causally).

    For a verify step with K draft tokens and prefix of length P:
      - Q has K positions (absolute positions [P, P+K)).
      - Full KV has P+K positions (prefix from cache + K new from current step).
      - Query i must attend to keys [0 .. P+i] (entire prefix + its own
        position and all earlier draft positions).

    We MUST NOT use `is_causal=True` here: PyTorch sdpa applies a top-left
    aligned `tril` mask for rectangular Q/K, so query i would only see keys
    [0..i] — i.e. only the first few prefix tokens. That silently destroys
    verification quality (the target model ends up scoring draft tokens
    against a truncated context). Instead we build a bottom-right aligned
    mask explicitly.

    Args:
        q:  [1, num_heads, K, head_dim]
        k_new, v_new:  [1, num_kv_heads, K, head_dim]
        key_cache, value_cache: [total_slots, num_kv_heads, head_dim]
        block_tables: [1, max_blocks_per_seq]
        prefix_lens: [1]
        block_size: int
        num_kv_groups: int
    Returns:
        output: [1, num_heads, K, head_dim]
    """
    if q.is_cuda and torch.cuda.is_current_stream_capturing():
        return _extend_attention_cuda_graph_safe(
            q, k_new, v_new, key_cache, value_cache,
            block_tables, prefix_lens, block_size, num_kv_groups,
        )
    return _extend_attention_eager(
        q, k_new, v_new, key_cache, value_cache,
        block_tables, prefix_lens, block_size, num_kv_groups,
    )


def _extend_attention_eager(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size: int,
    num_kv_groups: int,
) -> torch.Tensor:
    device = q.device
    prefix_len = int(prefix_lens[0].item())
    K = k_new.shape[2]

    if prefix_len > 0:
        positions = torch.arange(prefix_len, device=device)
        blk_idx = positions // block_size
        blk_off = positions % block_size
        physical = block_tables[0][blk_idx]
        slots = physical * block_size + blk_off
        k_prefix = key_cache[slots].transpose(0, 1).unsqueeze(0)
        v_prefix = value_cache[slots].transpose(0, 1).unsqueeze(0)
        k_full = torch.cat([k_prefix, k_new], dim=2)
        v_full = torch.cat([v_prefix, v_new], dim=2)
    else:
        k_full = k_new
        v_full = v_new

    k_full, v_full = expand_kv_for_gqa(k_full, v_full, num_kv_groups)

    kv_len = k_full.shape[2]
    q_idx = torch.arange(K, device=device).unsqueeze(1)
    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    attn_mask = k_idx <= (prefix_len + q_idx)

    return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)


def _extend_attention_cuda_graph_safe(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size: int,
    num_kv_groups: int,
) -> torch.Tensor:
    """Same semantics as eager path but no host sync / Python shape branches."""
    device = q.device
    pl = prefix_lens[0]
    K = k_new.shape[2]
    num_blocks = block_tables.shape[1]
    max_ctx = num_blocks * block_size

    positions = torch.arange(max_ctx, device=device, dtype=torch.long)
    in_prefix = positions < pl
    blk_idx = positions // block_size
    blk_off = positions % block_size
    blk_idx_safe = blk_idx.clamp(max=num_blocks - 1)
    physical = block_tables[0, blk_idx_safe]
    slots = physical * block_size + blk_off
    slots = torch.where(in_prefix, slots, torch.zeros_like(slots))

    k_prefix = key_cache[slots].transpose(0, 1).unsqueeze(0)
    v_prefix = value_cache[slots].transpose(0, 1).unsqueeze(0)
    mask_keep = in_prefix.to(dtype=k_prefix.dtype).view(1, 1, max_ctx, 1)
    k_prefix = k_prefix * mask_keep
    v_prefix = v_prefix * mask_keep

    k_full = torch.cat([k_prefix, k_new], dim=2)
    v_full = torch.cat([v_prefix, v_new], dim=2)
    k_full, v_full = expand_kv_for_gqa(k_full, v_full, num_kv_groups)

    kv_len = k_full.shape[2]
    q_idx = torch.arange(K, device=device).unsqueeze(1)
    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    attend_prefix = (k_idx < max_ctx) & (k_idx < pl)
    attend_draft = (k_idx >= max_ctx) & (k_idx < max_ctx + K) & ((k_idx - max_ctx) <= q_idx)
    attn_mask = attend_prefix | attend_draft

    return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
