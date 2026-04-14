"""Model, cache, scheduler, and speculative decoding configurations."""

from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration, parsed from HuggingFace config.json."""

    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    vocab_size: int = 32000
    max_position_embeddings: int = 4096
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    head_dim: Optional[int] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        with open(Path(model_path) / "config.json") as f:
            raw = json.load(f)
        return cls(
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            num_hidden_layers=raw["num_hidden_layers"],
            num_attention_heads=raw["num_attention_heads"],
            num_key_value_heads=raw.get(
                "num_key_value_heads", raw["num_attention_heads"]
            ),
            vocab_size=raw["vocab_size"],
            max_position_embeddings=raw.get("max_position_embeddings", 4096),
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", 10000.0),
            head_dim=raw.get("head_dim"),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            attention_bias=raw.get("attention_bias", True),
        )


@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: Optional[int] = None
    gpu_memory_utilization: float = 0.9


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096


@dataclass
class SpeculativeConfig:
    draft_model_path: Optional[str] = None
    num_speculative_tokens: int = 5
