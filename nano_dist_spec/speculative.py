"""Speculative decoding: draft-then-verify with rejection sampling.

Algorithm (Leviathan et al., 2023):
  1. **Draft** — a small, fast model generates K candidate tokens autoregressively.
  2. **Verify** — the large target model scores all K candidates in a single
     forward pass (nearly as fast as scoring 1 token, because LLM decode is
     memory-bandwidth-bound).
  3. **Reject / Accept** — for each candidate position i:
       - Accept with probability  min(1, p_target(x_i) / q_draft(x_i)).
       - If rejected, resample from the *residual* distribution
         norm(max(0, p_target - q_draft)) to maintain the exact target
         distribution guarantee.
  4. If all K candidates are accepted, sample a **bonus** token from the
     target model's distribution at position K+1.

Result: on average more than 1 token per target-model forward pass, with
output distributed identically to the target model.

Key interview insight:
  The acceptance rate depends on how well the draft model approximates the
  target. With K=5 and a well-matched draft, typical acceptance rates are
  60-80%, yielding 2-3x wall-clock speedup.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .attention import InputMetadata
from .debug import tracer
from .kv_cache import KVCache, KVCacheManager
from .model import TransformerModel
from .sampling import SamplingParams, logits_to_probs, sample


@dataclass
class SpeculativeOutput:
    """Result of one speculative decode round."""
    accepted_tokens: List[int]
    num_draft_tokens: int
    num_accepted: int


def rejection_sample(
    draft_token: int,
    draft_prob: float,
    target_probs: torch.Tensor,
    draft_probs_full: torch.Tensor,
    temperature: float,
) -> Tuple[bool, Optional[int]]:
    """Single-position rejection sampling.

    Mathematical guarantee (for sampling-based generation):
      Let q = draft_prob, p = target_probs[draft_token].
      Accept with prob min(1, p/q).
      If rejected, sample from r(x) = max(0, p(x) - q(x)) / Z
      where Z = sum_x max(0, p(x) - q(x)).
      This ensures the marginal distribution equals the target exactly.

    For greedy decoding (temperature=0):
      Accept iff argmax(target) == draft_token.
    """
    if temperature == 0:
        target_choice = target_probs.argmax(dim=-1).item()
        if target_choice == draft_token:
            return True, None
        return False, target_choice

    p = target_probs[draft_token].item()
    q = draft_prob
    accept_prob = min(1.0, p / max(q, 1e-10))

    if torch.rand(1).item() < accept_prob:
        return True, None

    residual = (target_probs - draft_probs_full).clamp(min=0)
    residual_sum = residual.sum()
    if residual_sum < 1e-10:
        correction = target_probs.argmax(dim=-1).item()
    else:
        residual = residual / residual_sum
        correction = torch.multinomial(residual, 1).item()
    return False, correction


class SpeculativeDecoder:
    """Orchestrates draft → verify → reject/accept loop."""

    def __init__(
        self,
        target_model: TransformerModel,
        draft_model: TransformerModel,
        target_kv: KVCache,
        draft_kv: KVCache,
        target_kv_mgr: KVCacheManager,
        draft_kv_mgr: KVCacheManager,
        num_speculative_tokens: int = 5,
        block_size: int = 16,
    ):
        self.target = target_model
        self.draft = draft_model
        self.target_kv = target_kv
        self.draft_kv = draft_kv
        self.target_mgr = target_kv_mgr
        self.draft_mgr = draft_kv_mgr
        self.K = num_speculative_tokens
        self.block_size = block_size
        self.device = next(target_model.parameters()).device

        # Target and draft may have different padded vocab sizes (e.g. Qwen2.5-7B
        # has 152064 while Qwen2.5-1.5B has 151936, both pad-to-multiple-of-N
        # over the same underlying tokenizer). Rejection sampling compares
        # p_target(x) against q_draft(x) element-wise, so both distributions
        # must live on the same support. Truncate to min(target, draft): tokens
        # beyond that index are padding slots never produced by the tokenizer
        # and safe to drop.
        self.shared_vocab_size = min(
            target_model.config.vocab_size, draft_model.config.vocab_size,
        )

    @torch.inference_mode()
    def prefill(
        self, seq_id: int, prompt_ids: torch.Tensor, params: SamplingParams,
    ) -> Tuple[int, torch.Tensor]:
        """Prefill both models, return first token and saved target logits."""
        bsz, seq_len = prompt_ids.shape
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        tracer.on_spec_event("PREFILL", seq_id=seq_id, prompt_len=seq_len)

        # --- Target prefill ---
        tracer.on_spec_event("PREFILL_TARGET", seq_id=seq_id)
        slot_map_t = self.target_mgr.compute_slot_mapping(seq_id, 0, seq_len, self.device)
        meta_t = InputMetadata(slot_mapping=slot_map_t, block_size=self.block_size)
        kv_list_t = [self.target_kv.get_kv(i) for i in range(self.target_kv.num_layers)]
        target_logits = self.target(prompt_ids, positions, kv_list_t, meta_t)

        # --- Draft prefill ---
        tracer.on_spec_event("PREFILL_DRAFT", seq_id=seq_id)
        slot_map_d = self.draft_mgr.compute_slot_mapping(seq_id, 0, seq_len, self.device)
        meta_d = InputMetadata(slot_mapping=slot_map_d, block_size=self.block_size)
        kv_list_d = [self.draft_kv.get_kv(i) for i in range(self.draft_kv.num_layers)]
        self.draft(prompt_ids, positions, kv_list_d, meta_d)

        target_last = target_logits[:, -1, : self.shared_vocab_size]
        first_token = sample(target_last, params).item()
        saved_probs = logits_to_probs(target_last, params.temperature)
        return first_token, saved_probs

    @torch.inference_mode()
    def speculative_step(
        self,
        seq_id: int,
        last_token: int,
        saved_target_probs: torch.Tensor,
        params: SamplingParams,
    ) -> Tuple[SpeculativeOutput, torch.Tensor]:
        """Execute one draft-then-verify round.

        Returns:
            output: SpeculativeOutput with accepted tokens
            new_saved_probs: target probs for verifying the next round's first draft
        """
        K = self.K
        prefix_len = self.target_mgr.context_lens[seq_id]
        tracer.on_spec_event(
            "STEP_BEGIN", seq_id=seq_id, prefix_len=prefix_len, K=K,
            last_token=last_token,
        )

        # =================================================================
        # 1. DRAFT: generate K candidate tokens autoregressively
        # =================================================================
        draft_tokens: List[int] = []
        draft_probs_list: List[torch.Tensor] = []  # full distributions

        current_token = last_token
        for i in range(K):
            pos = prefix_len + i
            tracer.on_spec_event("DRAFT_ITER", iter=i, pos=pos, in_token=current_token)
            self.draft_mgr.append_slots(seq_id, 1)
            slot = self.draft_mgr.compute_slot_mapping(seq_id, pos, 1, self.device)
            bt = self.draft_mgr.get_block_table_tensor([seq_id], self.device)
            cl = self.draft_mgr.get_context_lens_tensor([seq_id], self.device)

            inp = torch.tensor([[current_token]], device=self.device)
            pos_t = torch.tensor([[pos]], device=self.device)
            meta = InputMetadata(
                slot_mapping=slot, block_tables=bt,
                context_lens=cl, block_size=self.block_size,
            )
            kv_list = [self.draft_kv.get_kv(j) for j in range(self.draft_kv.num_layers)]
            draft_logits = self.draft(inp, pos_t, kv_list, meta)

            draft_last = draft_logits[:, -1, : self.shared_vocab_size]
            probs = logits_to_probs(draft_last, params.temperature)
            token = sample(draft_last, params).item()
            draft_tokens.append(token)
            draft_probs_list.append(probs.squeeze(0))
            current_token = token
        tracer.on_spec_event("DRAFT_DONE", tokens=draft_tokens)

        # =================================================================
        # 2. VERIFY: run target model on all K draft tokens at once
        # =================================================================
        tracer.on_spec_event(
            "VERIFY", seq_id=seq_id,
            range=f"[{prefix_len},{prefix_len + K})", num_tokens=K,
        )
        for i in range(K):
            self.target_mgr.append_slots(seq_id, 1)

        verify_start = prefix_len
        slot_map = self.target_mgr.compute_slot_mapping(
            seq_id, verify_start, K, self.device,
        )
        bt = self.target_mgr.get_block_table_tensor([seq_id], self.device)
        cl = self.target_mgr.get_context_lens_tensor([seq_id], self.device)

        inp = torch.tensor([draft_tokens], device=self.device)
        pos_t = torch.arange(verify_start, verify_start + K, device=self.device).unsqueeze(0)
        meta = InputMetadata(
            slot_mapping=slot_map, block_tables=bt,
            context_lens=cl, block_size=self.block_size,
        )
        kv_list = [self.target_kv.get_kv(j) for j in range(self.target_kv.num_layers)]
        target_logits = self.target(inp, pos_t, kv_list, meta)

        # target_probs_from_verify[i] = P(· | prefix, draft[0..i])
        # Used to verify draft[i+1].  Position 0 verifies draft[0] via saved_target_probs.
        target_probs_verify: List[torch.Tensor] = [saved_target_probs.squeeze(0)]
        for i in range(K):
            target_probs_verify.append(
                logits_to_probs(
                    target_logits[:, i, : self.shared_vocab_size],
                    params.temperature,
                ).squeeze(0)
            )

        # =================================================================
        # 3. REJECTION SAMPLING
        # =================================================================
        accepted: List[int] = []
        all_accepted = True

        for i in range(K):
            ok, correction = rejection_sample(
                draft_token=draft_tokens[i],
                draft_prob=draft_probs_list[i][draft_tokens[i]].item(),
                target_probs=target_probs_verify[i],
                draft_probs_full=draft_probs_list[i],
                temperature=params.temperature,
            )
            if ok:
                accepted.append(draft_tokens[i])
                tracer.on_spec_event("ACCEPT", pos=i, token=draft_tokens[i])
            else:
                accepted.append(correction)
                all_accepted = False
                tracer.on_spec_event(
                    "REJECT", pos=i, draft=draft_tokens[i], correction=correction,
                )
                break

        if all_accepted:
            # `target_probs_verify[K]` is already a proper (temperature-scaled)
            # probability distribution. Passing it to `sample()` would softmax
            # it AGAIN, which collapses any probability mass into a near-uniform
            # distribution over the whole vocab — that's how we get random
            # cross-language garbage tokens (Arabic, Russian, weird tags, etc.)
            # leaking into otherwise coherent output.
            bonus_probs = target_probs_verify[K]
            if params.temperature == 0:
                bonus = int(bonus_probs.argmax(dim=-1).item())
            else:
                bonus = int(torch.multinomial(bonus_probs, num_samples=1).item())
            accepted.append(bonus)
            tracer.on_spec_event("BONUS", token=bonus)

        n_accepted = len(accepted)
        n_draft_accepted = K if all_accepted else (n_accepted - 1)

        # =================================================================
        # 4. ROLL BACK rejected KV entries
        # =================================================================
        tracer.on_spec_event(
            "ROLLBACK", seq_id=seq_id,
            target_to=prefix_len + n_draft_accepted,
            draft_to=prefix_len,
            n_draft_accepted=n_draft_accepted,
        )
        # Target: keep prefix + n_draft_accepted tokens' KV
        self.target_mgr.rollback(seq_id, prefix_len + n_draft_accepted)
        # Draft: keep prefix only (we'll re-process accepted tokens next round)
        self.draft_mgr.rollback(seq_id, prefix_len)

        # Resync draft model KV with accepted tokens
        if n_draft_accepted > 0:
            sync_tokens = accepted[:n_draft_accepted]
            tracer.on_spec_event(
                "RESYNC_DRAFT", seq_id=seq_id, n=n_draft_accepted,
                tokens=sync_tokens,
            )
            for i, tk in enumerate(sync_tokens):
                pos = prefix_len + i
                self.draft_mgr.append_slots(seq_id, 1)
                slot = self.draft_mgr.compute_slot_mapping(seq_id, pos, 1, self.device)
                bt_d = self.draft_mgr.get_block_table_tensor([seq_id], self.device)
                cl_d = self.draft_mgr.get_context_lens_tensor([seq_id], self.device)
                inp_d = torch.tensor([[tk]], device=self.device)
                pos_d = torch.tensor([[pos]], device=self.device)
                meta_d = InputMetadata(
                    slot_mapping=slot, block_tables=bt_d,
                    context_lens=cl_d, block_size=self.block_size,
                )
                kv_d = [self.draft_kv.get_kv(j) for j in range(self.draft_kv.num_layers)]
                self.draft(inp_d, pos_d, kv_d, meta_d)

        # Process the last accepted token through target to cache its KV
        # and produce saved probs for the next round
        last_accepted = accepted[-1]
        final_pos = prefix_len + n_draft_accepted
        tracer.on_spec_event(
            "FINAL", seq_id=seq_id, token=last_accepted, pos=final_pos,
            accepted=accepted,
        )
        self.target_mgr.append_slots(seq_id, 1)
        slot_f = self.target_mgr.compute_slot_mapping(seq_id, final_pos, 1, self.device)
        bt_f = self.target_mgr.get_block_table_tensor([seq_id], self.device)
        cl_f = self.target_mgr.get_context_lens_tensor([seq_id], self.device)
        meta_f = InputMetadata(
            slot_mapping=slot_f, block_tables=bt_f,
            context_lens=cl_f, block_size=self.block_size,
        )
        kv_f = [self.target_kv.get_kv(j) for j in range(self.target_kv.num_layers)]
        last_logits = self.target(
            torch.tensor([[last_accepted]], device=self.device),
            torch.tensor([[final_pos]], device=self.device),
            kv_f, meta_f,
        )
        # Also sync draft
        self.draft_mgr.append_slots(seq_id, 1)
        slot_fd = self.draft_mgr.compute_slot_mapping(seq_id, final_pos, 1, self.device)
        bt_fd = self.draft_mgr.get_block_table_tensor([seq_id], self.device)
        cl_fd = self.draft_mgr.get_context_lens_tensor([seq_id], self.device)
        meta_fd = InputMetadata(
            slot_mapping=slot_fd, block_tables=bt_fd,
            context_lens=cl_fd, block_size=self.block_size,
        )
        kv_fd = [self.draft_kv.get_kv(j) for j in range(self.draft_kv.num_layers)]
        self.draft(
            torch.tensor([[last_accepted]], device=self.device),
            torch.tensor([[final_pos]], device=self.device),
            kv_fd, meta_fd,
        )

        new_saved = logits_to_probs(
            last_logits[:, -1, : self.shared_vocab_size], params.temperature,
        )

        output = SpeculativeOutput(
            accepted_tokens=accepted,
            num_draft_tokens=K,
            num_accepted=n_accepted,
        )
        return output, new_saved
