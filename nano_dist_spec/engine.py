"""Inference engine: the core loop that drives generation.

Two modes:
  1. Standard — autoregressive decode, one token per forward pass.
  2. Speculative — draft K tokens with a small model, verify with the large
     model, accept/reject via rejection sampling.

The engine coordinates model, KV cache, scheduler, and sampling.
"""

from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer

from .attention import InputMetadata
from .config import CacheConfig, ModelConfig, SchedulerConfig, SpeculativeConfig
from .debug import tracer
from .kv_cache import BlockAllocator, KVCache, KVCacheManager
from .model import TransformerModel
from .sampling import SamplingParams, sample
from .scheduler import Scheduler, Sequence
from .speculative import SpeculativeDecoder


class GenerationOutput:
    def __init__(self, prompt: str, text: str, token_ids: List[int]):
        self.prompt = prompt
        self.text = text
        self.token_ids = token_ids

    def __repr__(self):
        return f"GenerationOutput(text={self.text!r})"


class LLMEngine:
    """Core inference engine — manages model, KV cache, and generation loop."""

    def __init__(
        self,
        model_path: str,
        tp_size: int = 1,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        cache_config: Optional[CacheConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.tp_size = tp_size

        self.model_config = ModelConfig.from_pretrained(model_path)
        self.cache_config = cache_config or CacheConfig()
        self.scheduler_config = scheduler_config or SchedulerConfig()

        self.model = TransformerModel(self.model_config, tp_size=tp_size)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.load_weights(model_path)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )

        num_kv_heads = self.model_config.num_key_value_heads // tp_size
        block_size = self.cache_config.block_size
        num_blocks = self.cache_config.num_gpu_blocks or self._estimate_num_blocks()

        self.kv_cache = KVCache(
            num_layers=self.model_config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=self.model_config.head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.allocator = BlockAllocator(num_blocks)
        self.kv_mgr = KVCacheManager(block_size, self.allocator)
        self.scheduler = Scheduler(
            self.kv_mgr, max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        self.block_size = block_size

    def _estimate_num_blocks(self) -> int:
        """Estimate available KV cache blocks from free GPU memory."""
        if not torch.cuda.is_available():
            return 256
        free_mem, _ = torch.cuda.mem_get_info(self.device)
        usable = int(free_mem * self.cache_config.gpu_memory_utilization)
        cfg = self.model_config
        num_kv_heads = cfg.num_key_value_heads // self.tp_size
        bytes_per_block = (
            self.cache_config.block_size
            * num_kv_heads
            * cfg.head_dim
            * 2  # float16
            * 2  # K + V
            * cfg.num_hidden_layers
        )
        num = max(usable // max(bytes_per_block, 1), 32)
        return num

    @torch.inference_mode()
    def generate(
        self,
        prompts: List[str],
        sampling_params: SamplingParams,
    ) -> List[GenerationOutput]:
        """Generate completions for a list of prompts (offline batch)."""
        eos_id = self.tokenizer.eos_token_id
        all_prompt_ids = [
            self.tokenizer.encode(p, add_special_tokens=True) for p in prompts
        ]

        for ids in all_prompt_ids:
            self.scheduler.add_request(ids, max_tokens=sampling_params.max_tokens)

        results: dict[int, GenerationOutput] = {}

        while self.scheduler.has_pending:
            sched_out = self.scheduler.schedule()

            # --- Prefill ---
            for seq in sched_out.prefill_seqs:
                self._prefill_seq(seq, sampling_params)

            # --- Decode ---
            if sched_out.decode_seqs:
                self._decode_batch(sched_out.decode_seqs, sampling_params)

            # --- Collect finished ---
            finished_ids = []
            for seq in list(self.scheduler.running.values()):
                done = (
                    len(seq.generated_token_ids) >= seq.max_tokens
                    or (seq.generated_token_ids and seq.generated_token_ids[-1] == eos_id)
                )
                if done:
                    finished_ids.append(seq.seq_id)

            self.scheduler.after_step(sched_out.prefill_seqs, finished_ids)

        for seq in self.scheduler.finished.values():
            idx = seq.seq_id
            text = self.tokenizer.decode(seq.generated_token_ids, skip_special_tokens=True)
            results[idx] = GenerationOutput(
                prompt=prompts[idx],
                text=text,
                token_ids=seq.generated_token_ids,
            )

        return [results[i] for i in range(len(prompts))]

    def _prefill_seq(self, seq: Sequence, params: SamplingParams):
        ids = seq.prompt_token_ids
        seq_len = len(ids)
        tracer.on_step("PREFILL", seq_id=seq.seq_id, prompt_len=seq_len)

        input_ids = torch.tensor([ids], device=self.device)
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        slot_mapping = self.kv_mgr.compute_slot_mapping(
            seq.seq_id, 0, seq_len, self.device,
        )
        metadata = InputMetadata(slot_mapping=slot_mapping, block_size=self.block_size)
        kv_list = [self.kv_cache.get_kv(i) for i in range(self.kv_cache.num_layers)]

        logits = self.model(input_ids, positions, kv_list, metadata)
        token = sample(logits[:, -1, :], params).item()
        seq.generated_token_ids.append(token)

    def _decode_batch(self, seqs: List[Sequence], params: SamplingParams):
        seq_ids = [s.seq_id for s in seqs]
        tokens = [s.generated_token_ids[-1] for s in seqs]
        tracer.on_step("DECODE", seq_ids=seq_ids, batch=len(seq_ids))

        for sid in seq_ids:
            self.kv_mgr.append_slots(sid, 1)

        input_ids = torch.tensor(tokens, device=self.device).unsqueeze(1)
        positions_list = [self.kv_mgr.context_lens[s] - 1 for s in seq_ids]
        positions = torch.tensor(positions_list, device=self.device).unsqueeze(1)

        slot_list = []
        for sid in seq_ids:
            pos = self.kv_mgr.context_lens[sid] - 1
            slot_list.append(
                self.kv_mgr.compute_slot_mapping(sid, pos, 1, self.device)
            )
        slot_mapping = torch.cat(slot_list)

        block_tables = self.kv_mgr.get_block_table_tensor(seq_ids, self.device)
        context_lens = self.kv_mgr.get_context_lens_tensor(seq_ids, self.device)

        metadata = InputMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=self.block_size,
        )
        kv_list = [self.kv_cache.get_kv(i) for i in range(self.kv_cache.num_layers)]
        logits = self.model(input_ids, positions, kv_list, metadata)

        new_tokens = sample(logits[:, -1, :], params)
        for i, seq in enumerate(seqs):
            seq.generated_token_ids.append(new_tokens[i].item())


# ---------------------------------------------------------------------------
# User-facing API
# ---------------------------------------------------------------------------

class LLM:
    """High-level API for LLM inference (mirrors vLLM's LLM interface).

    Usage:
        llm = LLM("/path/to/model")
        outputs = llm.generate(["Hello, world!"], SamplingParams(temperature=0.7))
        print(outputs[0].text)
    """

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        dtype: str = "float16",
        device: str = "cuda",
        num_gpu_blocks: Optional[int] = None,
        block_size: int = 16,
        draft_model_path: Optional[str] = None,
        num_speculative_tokens: int = 5,
    ):
        dt = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        cache_cfg = CacheConfig(block_size=block_size, num_gpu_blocks=num_gpu_blocks)
        self.engine = LLMEngine(
            model_path=model_path,
            tp_size=tensor_parallel_size,
            dtype=dt,
            device=device,
            cache_config=cache_cfg,
        )
        self.draft_model_path = draft_model_path
        self.num_speculative_tokens = num_speculative_tokens
        self._spec_decoder: Optional[SpeculativeDecoder] = None

        if draft_model_path:
            self._init_speculative(draft_model_path, num_speculative_tokens, dt, device)

    def _init_speculative(self, draft_path: str, K: int, dtype: torch.dtype, device: str):
        draft_config = ModelConfig.from_pretrained(draft_path)
        draft_model = TransformerModel(draft_config, tp_size=self.engine.tp_size)
        draft_model.to(device=self.engine.device, dtype=dtype)
        draft_model.load_weights(draft_path)
        draft_model.eval()

        num_kv_heads = draft_config.num_key_value_heads // self.engine.tp_size
        block_size = self.engine.block_size
        num_blocks = self.engine.allocator.num_blocks // 2  # share memory budget

        draft_kv = KVCache(
            num_layers=draft_config.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=draft_config.head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            device=self.engine.device,
            dtype=dtype,
        )
        draft_allocator = BlockAllocator(num_blocks)
        draft_mgr = KVCacheManager(block_size, draft_allocator)

        self._spec_decoder = SpeculativeDecoder(
            target_model=self.engine.model,
            draft_model=draft_model,
            target_kv=self.engine.kv_cache,
            draft_kv=draft_kv,
            target_kv_mgr=self.engine.kv_mgr,
            draft_kv_mgr=draft_mgr,
            num_speculative_tokens=K,
            block_size=block_size,
        )

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[GenerationOutput]:
        params = sampling_params or SamplingParams()

        if self._spec_decoder is not None:
            return self._generate_speculative(prompts, params)

        return self.engine.generate(prompts, params)

    @torch.inference_mode()
    def _generate_speculative(
        self, prompts: List[str], params: SamplingParams,
    ) -> List[GenerationOutput]:
        """Speculative generation — process one prompt at a time."""
        results = []
        eos_id = self.engine.tokenizer.eos_token_id

        for prompt in prompts:
            prompt_ids = self.engine.tokenizer.encode(prompt, add_special_tokens=True)
            seq_id = 0

            self.engine.kv_mgr.allocate_seq(seq_id, len(prompt_ids))
            self._spec_decoder.draft_mgr.allocate_seq(seq_id, len(prompt_ids))

            prompt_tensor = torch.tensor([prompt_ids], device=self.engine.device)
            first_token, saved_probs = self._spec_decoder.prefill(
                seq_id, prompt_tensor, params,
            )

            generated = [first_token]
            total_accepted = 0
            total_draft = 0

            while len(generated) < params.max_tokens:
                if generated[-1] == eos_id:
                    break

                output, saved_probs = self._spec_decoder.speculative_step(
                    seq_id, generated[-1], saved_probs, params,
                )
                generated.extend(output.accepted_tokens)
                total_accepted += output.num_accepted
                total_draft += output.num_draft_tokens

            generated = generated[:params.max_tokens]
            text = self.engine.tokenizer.decode(generated, skip_special_tokens=True)

            self.engine.kv_mgr.free_seq(seq_id)
            self._spec_decoder.draft_mgr.free_seq(seq_id)

            results.append(GenerationOutput(prompt=prompt, text=text, token_ids=generated))

        return results
