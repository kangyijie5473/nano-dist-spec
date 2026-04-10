"""Tests for the speculative decoding rejection sampling algorithm."""

import torch
from nano_dist_spec.speculative import rejection_sample
from nano_dist_spec.sampling import logits_to_probs


def test_greedy_accept():
    """When target argmax matches draft token, greedy should accept."""
    target_probs = torch.tensor([0.1, 0.7, 0.2])
    draft_probs = torch.tensor([0.1, 0.6, 0.3])
    accepted, correction = rejection_sample(
        draft_token=1, draft_prob=0.6,
        target_probs=target_probs, draft_probs_full=draft_probs,
        temperature=0,
    )
    assert accepted is True
    assert correction is None


def test_greedy_reject():
    """When target argmax differs from draft token, greedy should reject."""
    target_probs = torch.tensor([0.1, 0.2, 0.7])
    draft_probs = torch.tensor([0.1, 0.6, 0.3])
    accepted, correction = rejection_sample(
        draft_token=1, draft_prob=0.6,
        target_probs=target_probs, draft_probs_full=draft_probs,
        temperature=0,
    )
    assert accepted is False
    assert correction == 2  # argmax of target


def test_sampling_perfect_match():
    """When draft == target distributions, acceptance rate should be 100%."""
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    n_accept = 0
    n_trials = 1000
    for _ in range(n_trials):
        accepted, _ = rejection_sample(
            draft_token=0, draft_prob=0.25,
            target_probs=probs, draft_probs_full=probs,
            temperature=1.0,
        )
        if accepted:
            n_accept += 1
    assert n_accept / n_trials > 0.95


def test_sampling_with_mismatch():
    """When draft and target differ, some tokens should be rejected."""
    target_probs = torch.tensor([0.9, 0.05, 0.05])
    draft_probs = torch.tensor([0.1, 0.8, 0.1])

    n_accept = 0
    n_trials = 500
    for _ in range(n_trials):
        accepted, correction = rejection_sample(
            draft_token=1, draft_prob=0.8,
            target_probs=target_probs, draft_probs_full=draft_probs,
            temperature=1.0,
        )
        if accepted:
            n_accept += 1
        else:
            assert correction is not None

    accept_rate = n_accept / n_trials
    assert accept_rate < 0.2  # p(1)/q(1) = 0.05/0.8 ≈ 0.0625


def test_residual_distribution():
    """Correction tokens should come from residual = max(0, p - q)."""
    target_probs = torch.tensor([0.6, 0.3, 0.1])
    draft_probs = torch.tensor([0.1, 0.8, 0.1])

    corrections = []
    for _ in range(500):
        accepted, correction = rejection_sample(
            draft_token=1, draft_prob=0.8,
            target_probs=target_probs, draft_probs_full=draft_probs,
            temperature=1.0,
        )
        if not accepted:
            corrections.append(correction)

    if corrections:
        # Token 0 has residual 0.5, token 2 has residual 0.0
        # So corrections should heavily favor token 0
        token_0_count = sum(1 for c in corrections if c == 0)
        assert token_0_count / len(corrections) > 0.8


def test_logits_to_probs_greedy():
    logits = torch.tensor([[1.0, 3.0, 2.0]])
    probs = logits_to_probs(logits, temperature=0)
    assert probs[0, 1] == 1.0  # argmax position
    assert probs.sum().item() == 1.0


def test_logits_to_probs_temperature():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    probs_low = logits_to_probs(logits, temperature=0.1)
    probs_high = logits_to_probs(logits, temperature=10.0)
    # Lower temperature -> more peaked distribution
    assert probs_low[0, 2] > probs_high[0, 2]


if __name__ == "__main__":
    test_greedy_accept()
    test_greedy_reject()
    test_sampling_perfect_match()
    test_sampling_with_mismatch()
    test_residual_distribution()
    test_logits_to_probs_greedy()
    test_logits_to_probs_temperature()
    print("All speculative decoding tests passed!")
