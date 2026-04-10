"""Sampling strategies: temperature scaling, top-k, top-p (nucleus)."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1          # -1 = disabled
    top_p: float = 1.0       # 1.0 = disabled
    max_tokens: int = 256

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be in [0, 1]")


def sample(
    logits: torch.Tensor,
    params: SamplingParams,
) -> torch.Tensor:
    """Sample next token ids from logits.

    Args:
        logits: [batch, vocab_size]
        params: SamplingParams
    Returns:
        token_ids: [batch]
    """
    if params.temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / params.temperature

    if params.top_k > 0:
        logits = _top_k_filter(logits, params.top_k)

    if params.top_p < 1.0:
        logits = _top_p_filter(logits, params.top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert logits to probability distribution (for speculative decoding)."""
    if temperature == 0:
        probs = torch.zeros_like(logits)
        probs.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    return F.softmax(logits / temperature, dim=-1)


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    top_k = min(k, logits.size(-1))
    threshold = logits.topk(top_k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    mask = cumulative_probs - sorted_logits.softmax(dim=-1) > p
    sorted_logits[mask] = float("-inf")
    return sorted_logits.scatter(-1, sorted_idx, sorted_logits)
