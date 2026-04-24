"""Transformer model (Llama / Qwen architecture) with Tensor Parallelism support.

The model uses ColumnParallelLinear and RowParallelLinear so that the same code
runs on 1 GPU (tp_size=1) or N GPUs (tp_size=N) without any change.

Weight loading maps HuggingFace checkpoint names to our module structure and
handles TP-aware splitting automatically.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .debug import tracer
from .parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    all_gather_last_dim,
    get_tp_rank,
    get_tp_world_size,
    tensor_split,
)
from .attention import (
    InputMetadata,
    apply_rotary_emb,
    decode_paged_attention,
    expand_kv_for_gqa,
    extend_attention,
    precompute_rope_cache,
    prefill_attention,
)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(orig_dtype)


class Attention(nn.Module):
    """Multi-head attention with GQA and Tensor Parallelism.

    TP strategy (per sub-layer, only 1 AllReduce total):
      Q, K, V projections -> ColumnParallel (each rank holds a head subset)
      O projection        -> RowParallel    (AllReduce aggregates partials)
    """

    def __init__(self, config: ModelConfig, tp_size: int = 1, layer_idx: int = 0):
        super().__init__()
        self.tp_size = tp_size
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = ColumnParallelLinear(
            config.hidden_size, config.num_attention_heads * config.head_dim,
            bias=config.attention_bias, tp_size=tp_size,
        )
        self.k_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias, tp_size=tp_size,
        )
        self.v_proj = ColumnParallelLinear(
            config.hidden_size, config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias, tp_size=tp_size,
        )
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * config.head_dim, config.hidden_size,
            tp_size=tp_size,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]],
        metadata: InputMetadata,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        q = self.q_proj(hidden).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin, positions)

        # --- Write new KV to cache ---
        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            k_flat = k.transpose(1, 2).contiguous().view(-1, self.num_kv_heads, self.head_dim)
            v_flat = v.transpose(1, 2).contiguous().view(-1, self.num_kv_heads, self.head_dim)
            key_cache[metadata.slot_mapping] = k_flat
            value_cache[metadata.slot_mapping] = v_flat
            if self.layer_idx == 0:
                tracer.on_kv_write(self.layer_idx, metadata.slot_mapping.tolist())

        # --- Compute attention ---
        if metadata.is_prefill:
            out = prefill_attention(q, k, v, self.num_kv_groups)
        elif seq_len == 1:
            out = decode_paged_attention(
                q, key_cache, value_cache,
                metadata.block_tables, metadata.context_lens,
                metadata.block_size, self.num_kv_groups,
            )
        else:
            prefix_lens = metadata.context_lens - seq_len
            out = extend_attention(
                q, k, v,
                key_cache, value_cache,
                metadata.block_tables, prefix_lens,
                metadata.block_size, self.num_kv_groups,
            )

        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """SwiGLU MLP with Tensor Parallelism.

    TP strategy (per sub-layer, only 1 AllReduce total):
      gate_proj, up_proj -> ColumnParallel (each rank handles a shard)
      down_proj          -> RowParallel    (AllReduce aggregates)
    """

    def __init__(self, config: ModelConfig, tp_size: int = 1):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, tp_size=tp_size,
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, tp_size=tp_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, tp_size: int = 1, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Attention(config, tp_size, layer_idx=layer_idx)
        self.mlp = MLP(config, tp_size)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, positions, cos, sin, kv_cache, metadata):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(hidden, positions, cos, sin, kv_cache, metadata)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class TransformerModel(nn.Module):
    def __init__(self, config: ModelConfig, tp_size: int = 1):
        super().__init__()
        self.config = config
        self.tp_size = tp_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, tp_size=tp_size,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config, tp_size, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size, config.vocab_size, tp_size=tp_size,
        )

        self.cos: Optional[torch.Tensor] = None
        self.sin: Optional[torch.Tensor] = None

    def _init_rope(self, device: torch.device):
        if self.cos is None:
            self.cos, self.sin = precompute_rope_cache(
                self.config.head_dim,
                self.config.max_position_embeddings,
                self.config.rope_theta,
                device=device,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        metadata: InputMetadata,
    ) -> torch.Tensor:
        """
        Returns:
            logits: [batch, seq_len, vocab_size] (full vocab, gathered if TP > 1)
        """
        self._init_rope(input_ids.device)
        hidden = self.embed_tokens(input_ids)

        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden = layer(hidden, positions, self.cos, self.sin, kv, metadata)

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        if self.tp_size > 1:
            logits = all_gather_last_dim(logits)

        return logits

    # ------------------------------------------------------------------
    # Weight loading from HuggingFace checkpoints
    # ------------------------------------------------------------------

    def load_weights(self, model_path: str):
        tp_rank = get_tp_rank()
        tp_size = self.tp_size
        weights = _load_safetensors(model_path)

        # --- Embedding ---
        w = weights["model.embed_tokens.weight"]
        if tp_size > 1:
            w = tensor_split(w, tp_rank, tp_size, dim=0)
        self.embed_tokens.embedding.weight.data.copy_(w)

        # --- Transformer layers ---
        for i, layer in enumerate(self.layers):
            p = f"model.layers.{i}"

            for name in ("q_proj", "k_proj", "v_proj"):
                w = weights[f"{p}.self_attn.{name}.weight"]
                if tp_size > 1:
                    w = tensor_split(w, tp_rank, tp_size, dim=0)
                getattr(layer.self_attn, name).linear.weight.data.copy_(w)
                bias_key = f"{p}.self_attn.{name}.bias"
                if bias_key in weights:
                    b = weights[bias_key]
                    if tp_size > 1:
                        b = tensor_split(b, tp_rank, tp_size, dim=0)
                    getattr(layer.self_attn, name).linear.bias.data.copy_(b)

            w = weights[f"{p}.self_attn.o_proj.weight"]
            if tp_size > 1:
                w = tensor_split(w, tp_rank, tp_size, dim=1)
            layer.self_attn.o_proj.linear.weight.data.copy_(w)

            for name in ("gate_proj", "up_proj"):
                w = weights[f"{p}.mlp.{name}.weight"]
                if tp_size > 1:
                    w = tensor_split(w, tp_rank, tp_size, dim=0)
                getattr(layer.mlp, name).linear.weight.data.copy_(w)

            w = weights[f"{p}.mlp.down_proj.weight"]
            if tp_size > 1:
                w = tensor_split(w, tp_rank, tp_size, dim=1)
            layer.mlp.down_proj.linear.weight.data.copy_(w)

            layer.input_layernorm.weight.data.copy_(
                weights[f"{p}.input_layernorm.weight"]
            )
            layer.post_attention_layernorm.weight.data.copy_(
                weights[f"{p}.post_attention_layernorm.weight"]
            )

        # --- Final norm ---
        self.norm.weight.data.copy_(weights["model.norm.weight"])

        # --- LM head ---
        w = weights.get("lm_head.weight", weights["model.embed_tokens.weight"])
        if tp_size > 1:
            w = tensor_split(w, tp_rank, tp_size, dim=0)
        self.lm_head.linear.weight.data.copy_(w)

        del weights


def _load_safetensors(model_path: str) -> dict:
    """Load all tensors from safetensors file(s) at *model_path*."""
    from safetensors import safe_open

    path = Path(model_path)
    tensors: dict[str, torch.Tensor] = {}

    single = path / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        return tensors

    index_file = path / "model.safetensors.index.json"
    if index_file.exists():
        with open(index_file) as f:
            index = json.load(f)
        loaded_files: set[str] = set()
        for filename in index["weight_map"].values():
            if filename not in loaded_files:
                with safe_open(str(path / filename), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
                loaded_files.add(filename)
        return tensors

    raise FileNotFoundError(
        f"No model.safetensors or index found in {model_path}"
    )
