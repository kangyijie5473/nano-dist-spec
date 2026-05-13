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
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.profiler import record_function

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

    if torch.rand(1, device=target_probs.device).item() < accept_prob:
        return True, None

    residual = (target_probs - draft_probs_full).clamp(min=0)
    residual_sum = residual.sum()
    if residual_sum < 1e-10:
        correction = target_probs.argmax(dim=-1).item()
    else:
        residual = residual / residual_sum
        correction = torch.multinomial(residual, 1).item()
    return False, correction


@dataclass(frozen=True)
class _CudaGraphState:
    """Captured CUDA Graph + its mutable input/output buffers."""
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    context_lens: torch.Tensor
    logits: torch.Tensor


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
        use_cuda_graph: bool = True,
        max_seq_len: int = 4096,
    ):
        self.target = target_model
        self.draft = draft_model
        self.target_kv = target_kv
        self.draft_kv = draft_kv
        self.target_mgr = target_kv_mgr
        self.draft_mgr = draft_kv_mgr
        self.K = num_speculative_tokens
        self._max_k_cap = num_speculative_tokens
        self.block_size = block_size
        self.device = next(target_model.parameters()).device
        self.use_cuda_graph = use_cuda_graph
        self._cuda_graph_target_enabled = use_cuda_graph
        self._cuda_graph_draft_enabled = use_cuda_graph
        self.max_seq_len = max_seq_len

        self.shared_vocab_size = min(
            target_model.config.vocab_size, draft_model.config.vocab_size,
        )

        self._max_blocks = (max_seq_len + block_size - 1) // block_size
        mb = self._max_blocks
        cap = self._max_k_cap

        # Target eager buffers — verify processes K+1 tokens ([last_token] + draft),
        # so we need cap+1 slots. There is no longer a separate final target forward.
        self._target_inp = torch.zeros((1, cap + 1), dtype=torch.long, device=self.device)
        self._target_pos = torch.zeros((1, cap + 1), dtype=torch.long, device=self.device)
        self._target_slots = torch.zeros((cap + 1,), dtype=torch.long, device=self.device)
        self._target_bt = torch.zeros((1, mb), dtype=torch.long, device=self.device)
        self._target_cl = torch.zeros((1,), dtype=torch.long, device=self.device)

        # Draft eager buffers — only single-token forwards now (loop iter + post-verify
        # write of d_{K-1} when all accepted). `_resync_*` retained at cap to support
        # legacy 1-token forwards through the same code path.
        self._draft_inp_1 = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._draft_pos_1 = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self._draft_slot_1 = torch.zeros((1,), dtype=torch.long, device=self.device)
        self._draft_bt = torch.zeros((1, mb), dtype=torch.long, device=self.device)
        self._draft_cl = torch.zeros((1,), dtype=torch.long, device=self.device)
        self._resync_inp = torch.zeros((1, cap), dtype=torch.long, device=self.device)
        self._resync_pos = torch.zeros((1, cap), dtype=torch.long, device=self.device)
        self._resync_slots = torch.zeros((cap,), dtype=torch.long, device=self.device)

        # CUDA Graph caches — keyed by seq_len.
        self._target_graphs: Dict[int, _CudaGraphState] = {}
        self._draft_graphs: Dict[int, _CudaGraphState] = {}

        self._kv_list_target = [self.target_kv.get_kv(i) for i in range(self.target_kv.num_layers)]
        self._kv_list_draft = [self.draft_kv.get_kv(i) for i in range(self.draft_kv.num_layers)]

    # ------------------------------------------------------------------
    # CUDA Graph helpers
    # ------------------------------------------------------------------

    def reset_cuda_graph_runtime(self) -> None:
        """Drop captured graphs (e.g. after KV `free_seq` / re-`allocate_seq` changes block tables)."""
        self._cuda_graph_target_enabled = self.use_cuda_graph
        self._cuda_graph_draft_enabled = self.use_cuda_graph
        self._target_graphs.clear()
        self._draft_graphs.clear()

    def _can_cuda_graph(self, params: SamplingParams, *, draft: bool = False) -> bool:
        enabled = self._cuda_graph_draft_enabled if draft else self._cuda_graph_target_enabled
        if not enabled:
            return False
        if not self.use_cuda_graph or self.device.type != "cuda" or not torch.cuda.is_available():
            return False
        if params.temperature != 0 or params.top_k > 0 or params.top_p < 1.0:
            return False
        return True

    def _build_graph(
        self,
        model: TransformerModel,
        kv_list: List[Tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
        warmup_extra: Optional[Callable] = None,
        fill_inputs: Optional[Callable[..., None]] = None,
    ) -> _CudaGraphState:
        """Capture a CUDA Graph for a single-sequence forward of *seq_len* tokens.

        Args:
            model: target or draft model.
            kv_list: pre-built KV layer list.
            seq_len: number of tokens in the forward pass.
            warmup_extra: optional callable(logits) run during warmup (e.g. sample).
            fill_inputs: if set, called before warmup and again before capture to
                populate graph input buffers with a **valid** decode/extend step.
                Warmup/capture with all-zero ``context_lens`` / ``slot_mapping`` runs
                extend_attention with prefix_len <= 0 and wrong KV writes, producing
                a captured graph that does not match real verify semantics.
        """
        input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)
        positions = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)
        slot_mapping = torch.zeros((seq_len,), dtype=torch.long, device=self.device)
        block_tables = torch.zeros(
            (1, self._max_blocks), dtype=torch.long, device=self.device,
        )
        context_lens = torch.zeros((1,), dtype=torch.long, device=self.device)
        meta = InputMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=self.block_size,
        )
        if fill_inputs is not None:
            fill_inputs(input_ids, positions, slot_mapping, block_tables, context_lens)
        warmup = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(warmup):
            for _ in range(2):
                logits = model(input_ids, positions, kv_list, meta)
                if warmup_extra is not None:
                    warmup_extra(logits)
        torch.cuda.current_stream(self.device).wait_stream(warmup)
        torch.cuda.synchronize(self.device)

        if fill_inputs is not None:
            fill_inputs(input_ids, positions, slot_mapping, block_tables, context_lens)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = model(input_ids, positions, kv_list, meta)
        return _CudaGraphState(
            graph=graph,
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            logits=logits,
        )

    def _get_target_graph(
        self,
        k: int,
        fill_inputs: Callable[..., None],
    ) -> _CudaGraphState:
        st = self._target_graphs.get(k)
        if st is not None:
            return st
        st = self._build_graph(
            self.target, self._kv_list_target, k, fill_inputs=fill_inputs,
        )
        self._target_graphs[k] = st
        return st

    def _get_draft_graph(
        self,
        seq_len: int,
        params: SamplingParams,
        fill_inputs: Callable[..., None],
    ) -> _CudaGraphState:
        st = self._draft_graphs.get(seq_len)
        if st is not None:
            return st
        warmup_extra = None
        if seq_len == 1:
            warmup_extra = lambda logits: sample(
                logits[:, -1, : self.shared_vocab_size], params,
            )
        st = self._build_graph(
            self.draft, self._kv_list_draft, seq_len,
            warmup_extra=warmup_extra, fill_inputs=fill_inputs,
        )
        self._draft_graphs[seq_len] = st
        return st

    # ------------------------------------------------------------------
    # Target forward (verify + final unified)
    # ------------------------------------------------------------------

    def _run_target_forward(
        self,
        seq_id: int,
        start_pos: int,
        tokens: List[int],
        params: SamplingParams,
    ) -> torch.Tensor:
        """Run target model forward on *tokens* starting at *start_pos*.

        Works for both verify (K tokens) and final (1 token) steps.  The k=1
        CUDA Graph is shared — a single-token verify is identical to a final
        step (both route through decode_paged_attention).
        """
        k = len(tokens)
        if self._can_cuda_graph(params):
            try:
                def _fill_target_buffers(
                    inp: torch.Tensor,
                    pos: torch.Tensor,
                    sm: torch.Tensor,
                    bt: torch.Tensor,
                    cl: torch.Tensor,
                ) -> None:
                    for i, t in enumerate(tokens):
                        inp[0, i] = t
                    pos[0, :k].copy_(
                        torch.arange(
                            start_pos, start_pos + k,
                            device=self.device, dtype=torch.long,
                        ),
                    )
                    self.target_mgr.compute_slot_mapping_into(
                        seq_id, start_pos, k, sm,
                    )
                    self.target_mgr.fill_block_table_padded(
                        seq_id, bt, self._max_blocks,
                    )
                    cl[0] = self.target_mgr.context_lens[seq_id]

                state = self._get_target_graph(k, _fill_target_buffers)
                _fill_target_buffers(
                    state.input_ids, state.positions,
                    state.slot_mapping, state.block_tables, state.context_lens,
                )
                state.graph.replay()
                return state.logits
            except Exception as exc:
                self._cuda_graph_target_enabled = False
                self._target_graphs.clear()
                print(f"[spec] target cuda graph disabled: {exc}")

        self._target_inp[0, :k].copy_(
            torch.tensor(tokens, device=self.device, dtype=torch.long),
        )
        self._target_pos[0, :k].copy_(
            torch.arange(
                start_pos, start_pos + k,
                device=self.device, dtype=torch.long,
            ),
        )
        self.target_mgr.compute_slot_mapping_into(
            seq_id, start_pos, k, self._target_slots,
        )
        self.target_mgr.fill_block_table_padded(
            seq_id, self._target_bt, self._max_blocks,
        )
        self._target_cl[0] = self.target_mgr.context_lens[seq_id]
        meta = InputMetadata(
            slot_mapping=self._target_slots[:k],
            block_tables=self._target_bt,
            context_lens=self._target_cl,
            block_size=self.block_size,
        )
        return self.target(
            self._target_inp[:, :k], self._target_pos[:, :k],
            self._kv_list_target, meta,
        )

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prefill(
        self, seq_id: int, prompt_ids: torch.Tensor, params: SamplingParams,
    ) -> int:
        """Prefill both models and return the first generated token.

        The first token is sampled from the target's last-position logits but
        is NOT yet written to either KV cache. It is treated as a "pending"
        token: the first call to :meth:`speculative_step` will receive it as
        ``last_token`` and write it into both caches as the first input of
        target's verify forward (and as the seed of the draft loop).
        """
        bsz, seq_len = prompt_ids.shape
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        tracer.on_spec_event("PREFILL", seq_id=seq_id, prompt_len=seq_len)

        tracer.on_spec_event("PREFILL_TARGET", seq_id=seq_id)
        slot_map_t = self.target_mgr.compute_slot_mapping(seq_id, 0, seq_len, self.device)
        meta_t = InputMetadata(slot_mapping=slot_map_t, block_size=self.block_size)
        target_logits = self.target(prompt_ids, positions, self._kv_list_target, meta_t)

        tracer.on_spec_event("PREFILL_DRAFT", seq_id=seq_id)
        slot_map_d = self.draft_mgr.compute_slot_mapping(seq_id, 0, seq_len, self.device)
        meta_d = InputMetadata(slot_mapping=slot_map_d, block_size=self.block_size)
        self.draft(prompt_ids, positions, self._kv_list_draft, meta_d)

        target_last = target_logits[:, -1, : self.shared_vocab_size]
        first_token = sample(target_last, params).item()
        return first_token

    # ------------------------------------------------------------------
    # Draft helpers
    # ------------------------------------------------------------------

    def _draft_step_one(
        self, seq_id: int, pos: int, current_token: int, params: SamplingParams,
    ) -> Tuple[int, torch.Tensor]:
        self.draft_mgr.append_slots(seq_id, 1)
        pos_actual = self.draft_mgr.context_lens[seq_id] - 1
        if pos_actual != pos:
            raise RuntimeError(f"draft slot position mismatch: expected {pos}, got {pos_actual}")

        draft_logits = self._run_draft_forward(
            seq_id=seq_id,
            start_pos=pos,
            tokens=[current_token],
            params=params,
        )
        draft_last = draft_logits[:, -1, : self.shared_vocab_size]
        probs = logits_to_probs(draft_last, params.temperature)
        token = sample(draft_last, params).item()
        return token, probs.squeeze(0)

    def _run_draft_forward(
        self,
        seq_id: int,
        start_pos: int,
        tokens: List[int],
        params: SamplingParams,
    ) -> torch.Tensor:
        """Run draft model forward for step/resync/final with shared buffers."""
        n = len(tokens)
        if n <= 0:
            raise RuntimeError("draft forward requires at least one token")

        if self._can_cuda_graph(params, draft=True):
            try:
                def _fill_draft_buffers(
                    inp: torch.Tensor,
                    pos: torch.Tensor,
                    sm: torch.Tensor,
                    bt: torch.Tensor,
                    cl: torch.Tensor,
                ) -> None:
                    for i, t in enumerate(tokens):
                        inp[0, i] = t
                    pos[0, :n].copy_(
                        torch.arange(
                            start_pos, start_pos + n,
                            device=self.device, dtype=torch.long,
                        ),
                    )
                    self.draft_mgr.compute_slot_mapping_into(
                        seq_id, start_pos, n, sm,
                    )
                    self.draft_mgr.fill_block_table_padded(
                        seq_id, bt, self._max_blocks,
                    )
                    cl[0] = self.draft_mgr.context_lens[seq_id]

                st = self._get_draft_graph(n, params, _fill_draft_buffers)
                _fill_draft_buffers(
                    st.input_ids, st.positions,
                    st.slot_mapping, st.block_tables, st.context_lens,
                )
                st.graph.replay()
                return st.logits
            except Exception as exc:
                self._cuda_graph_draft_enabled = False
                self._draft_graphs.clear()
                print(f"[spec] draft cuda graph disabled: {exc}")

        for i, t in enumerate(tokens):
            self._resync_inp[0, i] = t
        self._resync_pos[0, :n].copy_(
            torch.arange(
                start_pos, start_pos + n,
                device=self.device, dtype=torch.long,
            ),
        )
        self.draft_mgr.compute_slot_mapping_into(seq_id, start_pos, n, self._resync_slots)
        self.draft_mgr.fill_block_table_padded(seq_id, self._draft_bt, self._max_blocks)
        self._draft_cl[0] = self.draft_mgr.context_lens[seq_id]
        meta_d = InputMetadata(
            slot_mapping=self._resync_slots[:n],
            block_tables=self._draft_bt,
            context_lens=self._draft_cl,
            block_size=self.block_size,
        )
        return self.draft(
            self._resync_inp[:, :n], self._resync_pos[:, :n],
            self._kv_list_draft, meta_d,
        )

    # ------------------------------------------------------------------
    # Main speculative step
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def speculative_step(
        self,
        seq_id: int,
        last_token: int,
        params: SamplingParams,
    ) -> SpeculativeOutput:
        """Execute one draft-then-verify round.

        ``last_token`` is the token PENDING write to KV cache — either the
        ``first_token`` returned by :meth:`prefill`, or the bonus/correction
        accepted at the end of the previous step. It will be written to both
        target (via verify) and draft (via draft-loop iter 0) caches at
        position ``prefix_len`` during this step.

        The target verify forward processes ``[last_token, d_0, ..., d_{K-1}]``
        at positions ``[P, P+1, ..., P+K]``. ``target_logits[i]`` for
        ``i in [0, K-1]`` verifies ``d_i``; ``target_logits[K]`` is the bonus
        prediction. This is the standard "one-shot" verification convention
        used by vLLM and the Leviathan et al. paper.
        """
        K = self.K
        prefix_len = self.target_mgr.context_lens[seq_id]
        if K > self._max_k_cap:
            raise RuntimeError("K exceeds decoder capacity; recreate SpeculativeDecoder with larger K")

        tracer.on_spec_event(
            "STEP_BEGIN", seq_id=seq_id, prefix_len=prefix_len, K=K,
            last_token=last_token,
        )

        # --- Draft ---
        # Iter 0 processes ``last_token`` at pos ``prefix_len`` and predicts ``d_0``;
        # iter i (i>0) processes ``d_{i-1}`` at pos ``prefix_len+i`` and predicts ``d_i``.
        # After the loop, draft cache contains [..., last_token, d_0..d_{K-2}] at
        # positions [..., P, P+1, ..., P+K-1]. ``d_{K-1}`` is sampled but NOT yet
        # in draft cache.
        draft_tokens: List[int] = []
        draft_probs_list: List[torch.Tensor] = []

        current_token = last_token
        with record_function("draft_loop"):
            for i in range(K):
                pos = prefix_len + i
                tracer.on_spec_event("DRAFT_ITER", iter=i, pos=pos, in_token=current_token)
                token, probs = self._draft_step_one(seq_id, pos, current_token, params)
                draft_tokens.append(token)
                draft_probs_list.append(probs)
                current_token = token
        tracer.on_spec_event("DRAFT_DONE", tokens=draft_tokens)

        # --- Verify ---
        # Target processes K+1 tokens (last_token plus K draft tokens) at positions
        # [prefix_len, prefix_len+K]. This writes last_token at pos prefix_len —
        # the key fix: previously last_token's KV was missing, causing the cache
        # to drift one position relative to the actual generated sequence.
        tracer.on_spec_event(
            "VERIFY", seq_id=seq_id,
            range=f"[{prefix_len},{prefix_len + K}]", num_tokens=K + 1,
        )
        for _ in range(K + 1):
            self.target_mgr.append_slots(seq_id, 1)

        with record_function("verify_target_batch"):
            target_logits = self._run_target_forward(
                seq_id, prefix_len, [last_token] + draft_tokens, params,
            )

        # target_logits[i] = pred at pos prefix_len+i+1 given [last_token, d_0..d_{i-1}].
        #   - i in [0, K-1]: verifies draft_tokens[i]
        #   - i == K       : bonus distribution
        target_probs_verify: List[torch.Tensor] = []
        for i in range(K + 1):
            target_probs_verify.append(
                logits_to_probs(
                    target_logits[:, i, : self.shared_vocab_size],
                    params.temperature,
                ).squeeze(0)
            )

        # --- Reject / Accept ---
        accepted: List[int] = []
        all_accepted = True

        with record_function("rejection_sampling"):
            if params.temperature == 0 and K > 0:
                # Greedy: choices[i] = target's greedy prediction at pos P+i+1
                # given last_token + d_0..d_{i-1}. Accept iff equal to d_i.
                choices = target_logits[0, :K, : self.shared_vocab_size].argmax(dim=-1)
                draft_t = torch.tensor(draft_tokens, device=self.device, dtype=torch.long)
                eq = choices == draft_t
                if bool(eq.all().item()):
                    accepted = list(draft_tokens)
                else:
                    all_accepted = False
                    bad = int((~eq).nonzero(as_tuple=False)[0, 0].item())
                    accepted = draft_tokens[:bad] + [int(choices[bad].item())]
                    for i in range(bad):
                        tracer.on_spec_event("ACCEPT", pos=i, token=draft_tokens[i])
                    tracer.on_spec_event(
                        "REJECT", pos=bad, draft=draft_tokens[bad],
                        correction=int(choices[bad].item()),
                    )
            else:
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

        if params.temperature == 0 and all_accepted:
            for i in range(K):
                tracer.on_spec_event("ACCEPT", pos=i, token=draft_tokens[i])

        if all_accepted:
            bonus_probs = target_probs_verify[K]
            if params.temperature == 0:
                bonus = int(bonus_probs.argmax(dim=-1).item())
            else:
                bonus = int(torch.multinomial(bonus_probs, num_samples=1).item())
            accepted.append(bonus)
            tracer.on_spec_event("BONUS", token=bonus)

        n_accepted = len(accepted)
        n_draft_accepted = K if all_accepted else (n_accepted - 1)

        # --- Cache management ---
        # Target cache after verify holds [..., last_token, d_0..d_{K-1}] at
        # [..., P..P+K] (context_lens = P+K+1). The new pending token
        # (bonus or correction) is NOT written — it'll be written in the next
        # step's verify as the new last_token.
        #   - all accepted (n_draft_accepted == K): no rollback needed.
        #   - partial (j < K): rollback to P+j+1, keeping last_token + d_0..d_{j-1}.
        target_keep = prefix_len + n_draft_accepted + 1
        with record_function("rollback_kv"):
            tracer.on_spec_event(
                "ROLLBACK", seq_id=seq_id,
                target_to=target_keep,
                draft_to=target_keep,
                n_draft_accepted=n_draft_accepted,
            )
            self.target_mgr.rollback(seq_id, target_keep)

        # Draft cache after draft loop has [..., last_token, d_0..d_{K-2}] at
        # [..., P..P+K-1] (context_lens = P+K). To match target:
        #   - all accepted: append d_{K-1} at pos P+K (one extra 1-token forward).
        #   - partial (j < K): rollback to P+j+1, keeping last_token + d_0..d_{j-1}.
        if all_accepted and K > 0:
            self.draft_mgr.append_slots(seq_id, 1)
            with record_function("draft_write_last"):
                self._run_draft_forward(
                    seq_id=seq_id,
                    start_pos=prefix_len + K,
                    tokens=[draft_tokens[K - 1]],
                    params=params,
                )
        else:
            self.draft_mgr.rollback(seq_id, target_keep)

        output = SpeculativeOutput(
            accepted_tokens=accepted,
            num_draft_tokens=K,
            num_accepted=n_accepted,
        )
        return output

