from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

import torch
from transformers import AutoTokenizer

from nano_dist_spec import LLM, LLMEngine, SamplingParams
from nano_dist_spec.config import CacheConfig, SchedulerConfig


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_cuda_mem_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def make_token_ids(tokenizer, length: int) -> List[int]:
    """Create a deterministic token list with exact length (fixed benchmark text)."""
    seed = (
        "Once upon a time, in a land far beyond the mountains, there lived a "
        "curious young scribe who spent every evening copying ancient texts "
        "by candlelight, hoping to one day uncover the secrets that the old "
        "wizards had hidden inside their poems. "
    )
    ids = tokenizer.encode(seed, add_special_tokens=False)
    if not ids:
        ids = [0]
    while len(ids) < length:
        ids = ids + ids
    ids = ids[:length]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and length >= 1:
        ids[0] = bos
    return ids


def make_random_token_ids(tokenizer, length: int, rng: random.Random) -> List[int]:
    """Synthetic prompt aligned with vLLM `bench throughput --random-input-len L --random-range-ratio 0`.

    Each token id is sampled uniformly from the tokenizer vocabulary, excluding
    BOS/EOS/PAD if present so prompts rarely terminate during prefill.
    """
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab <= 0:
        raise ValueError("tokenizer.vocab_size missing for random prompts")

    bad_ids = {
        x
        for x in (
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
        )
        if x is not None
    }

    out: List[int] = []
    for _ in range(length):
        tid = rng.randrange(vocab)
        guard = 0
        while tid in bad_ids and guard < 32:
            tid = rng.randrange(vocab)
            guard += 1
        out.append(tid)
    return out


def reset_scheduler(engine: LLMEngine) -> None:
    for sid in list(engine.scheduler.running.keys()):
        engine.kv_mgr.free_seq(sid)
    engine.scheduler.waiting.clear()
    engine.scheduler.running.clear()
    engine.scheduler.finished.clear()


def throughput_metrics(
    elapsed_s: float,
    num_prompts: int,
    total_output_tokens: int,
) -> Dict[str, float]:
    return {
        "elapsed_s": elapsed_s,
        "num_prompts": float(num_prompts),
        "total_output_tokens": float(total_output_tokens),
        "request_throughput_rps": (num_prompts / elapsed_s) if elapsed_s > 0 else 0.0,
        "output_token_throughput_tps": (
            total_output_tokens / elapsed_s if elapsed_s > 0 else 0.0
        ),
    }


@dataclass
class SharedArgs:
    input_len: int
    output_len: int
    num_prompts: int
    max_num_seqs: int
    max_model_len: int
    tensor_parallel_size: int
    num_gpu_blocks: Optional[int] = None
    #: `random`: i.i.d. token ids per prompt (vLLM-style synthetic); `fixed`: legacy seed text.
    prompt_mode: Literal["fixed", "random"] = "random"
    bench_seed: int = 42

    def validate(self, max_k: int = 0) -> None:
        if self.max_num_seqs != 1:
            raise ValueError("当前 benchmark 仅支持 --max-num-seqs 1")
        required = self.input_len + self.output_len + max_k + 2
        if required > self.max_model_len:
            raise ValueError(
                f"max_model_len 太小: 需要 >= {required}, 当前 {self.max_model_len}",
            )


def _make_prompt_list(
    tokenizer,
    shared: SharedArgs,
    rng: random.Random,
) -> List[List[int]]:
    """Build one prompt per request: identical repeats for `fixed`, i.i.d. for `random`."""
    if shared.prompt_mode == "fixed":
        base = make_token_ids(tokenizer, shared.input_len)
        return [list(base) for _ in range(shared.num_prompts)]

    return [
        make_random_token_ids(tokenizer, shared.input_len, rng)
        for _ in range(shared.num_prompts)
    ]


def _build_target_engine(
    model_path: str,
    shared: SharedArgs,
    use_cuda_graph: bool = True,
) -> LLMEngine:
    return LLMEngine(
        model_path=model_path,
        tp_size=shared.tensor_parallel_size,
        dtype=torch.bfloat16,
        device="cuda",
        cache_config=CacheConfig(num_gpu_blocks=shared.num_gpu_blocks),
        scheduler_config=SchedulerConfig(max_num_seqs=shared.max_num_seqs),
        use_cuda_graph=use_cuda_graph,
    )


def _bench_basic_single(
    engine: LLMEngine,
    prompt_ids: List[int],
    output_len: int,
) -> Dict[str, float]:
    reset_scheduler(engine)
    params = SamplingParams(temperature=0.0, max_tokens=output_len)
    seq_id = engine.scheduler.add_request(prompt_ids, max_tokens=output_len)

    sched_out = engine.scheduler.schedule()
    if len(sched_out.prefill_seqs) != 1:
        raise RuntimeError("expected exactly one prefill sequence")
    seq = sched_out.prefill_seqs[0]

    cuda_sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        engine._prefill_seq(seq, params)
    cuda_sync()
    ttft_s = time.perf_counter() - t0
    engine.scheduler.after_step([seq], [])

    decode_steps = 0
    cuda_sync()
    t1 = time.perf_counter()
    with torch.inference_mode():
        while len(seq.generated_token_ids) < output_len:
            engine._decode_batch([seq], params)
            decode_steps += 1
    cuda_sync()
    decode_s = time.perf_counter() - t1

    engine.kv_mgr.free_seq(seq_id)
    engine.scheduler.running.pop(seq_id, None)

    return {
        "ttft_s": ttft_s,
        "decode_tps": decode_steps / decode_s if decode_s > 0 else 0.0,
        "total_tokens": len(seq.generated_token_ids),
    }


def _bench_basic_prompt_set(
    engine: LLMEngine,
    prompts: Sequence[Sequence[int]],
    output_len: int,
) -> Dict[str, Any]:
    reset_cuda_mem_stats()
    reset_scheduler(engine)

    per_request: List[Dict[str, float]] = []
    total_output_tokens = 0

    cuda_sync()
    t0 = time.perf_counter()
    for prompt_ids in prompts:
        r = _bench_basic_single(engine, list(prompt_ids), output_len)
        per_request.append(r)
        total_output_tokens += int(r["total_tokens"])
    cuda_sync()
    elapsed = time.perf_counter() - t0

    num_prompts = len(prompts)
    n = max(len(per_request), 1)
    return {
        "throughput": throughput_metrics(elapsed, num_prompts, total_output_tokens),
        "per_request_mean_ttft_s": sum(x["ttft_s"] for x in per_request) / n,
        "per_request_mean_decode_tps": sum(x["decode_tps"] for x in per_request) / n,
        "peak_mem_gb": peak_mem_gb(),
    }


def run_basic(model: str, shared: SharedArgs) -> Dict[str, Any]:
    shared.validate(max_k=0)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    rng = random.Random(shared.bench_seed)
    prompts = _make_prompt_list(tokenizer, shared, rng)

    engine = _build_target_engine(model, shared)
    result = _bench_basic_prompt_set(
        engine=engine,
        prompts=prompts,
        output_len=shared.output_len,
    )
    return {
        "mode": "basic",
        "model": model,
        "config": asdict(shared),
        "summary": result,
    }


def _bench_spec_single(
    llm: LLM,
    prompt_ids: List[int],
    output_len: int,
) -> Dict[str, Any]:
    spec = llm._spec_decoder
    if spec is None:
        raise RuntimeError("spec decoder not initialized")

    spec.reset_cuda_graph_runtime()
    seq_id = 0
    params = SamplingParams(temperature=0.0, max_tokens=output_len)

    llm.engine.kv_mgr.allocate_seq(seq_id, len(prompt_ids))
    spec.draft_mgr.allocate_seq(seq_id, len(prompt_ids))
    prompt_tensor = torch.tensor([prompt_ids], device=llm.engine.device)

    cuda_sync()
    t0 = time.perf_counter()
    first_token, saved_probs = spec.prefill(seq_id, prompt_tensor, params)
    cuda_sync()
    ttft_s = time.perf_counter() - t0

    generated = [first_token]
    total_accepted = 0
    total_draft = 0
    total_draft_accepted = 0
    num_rounds = 0
    accept_by_pos = [0] * spec.K
    #: drafted_by_pos[i] = number of rounds where the draft loop reached position i
    #: (i.e. steps 0..i-1 were all accepted in that round, so step i was actually evaluated).
    #: This is the denominator for the per-position *conditional* acceptance rate,
    #: matching vLLM's "Per-position acceptance rate".
    drafted_by_pos = [0] * spec.K

    cuda_sync()
    t1 = time.perf_counter()
    while len(generated) < output_len:
        out, saved_probs = spec.speculative_step(
            seq_id, generated[-1], saved_probs, params,
        )
        generated.extend(out.accepted_tokens)
        total_accepted += out.num_accepted
        total_draft += out.num_draft_tokens
        if out.num_accepted > out.num_draft_tokens:
            n_draft_accepted = out.num_draft_tokens
        else:
            n_draft_accepted = max(out.num_accepted - 1, 0)
        total_draft_accepted += n_draft_accepted
        for pos in range(min(n_draft_accepted, spec.K)):
            accept_by_pos[pos] += 1
        for pos in range(n_draft_accepted + 1):
            if pos < spec.K:
                drafted_by_pos[pos] += 1
        num_rounds += 1
    cuda_sync()
    decode_s = time.perf_counter() - t1

    generated = generated[:output_len]
    decode_tokens = max(len(generated) - 1, 0)

    llm.engine.kv_mgr.free_seq(seq_id)
    spec.draft_mgr.free_seq(seq_id)

    # Per-position conditional acceptance rate: accept_by_pos[i] / drafted_by_pos[i]
    # This matches vLLM's "Per-position acceptance rate":
    #   P(accept at pos i | draft loop actually reached pos i)
    per_pos_accept_rate: List[Optional[float]] = []
    for pos in range(spec.K):
        denom = drafted_by_pos[pos]
        if denom > 0:
            per_pos_accept_rate.append(accept_by_pos[pos] / denom)
        else:
            per_pos_accept_rate.append(None)

    # Mean acceptance length (vLLM: "Mean acceptance length")
    mean_acceptance_length = total_draft_accepted / num_rounds if num_rounds else 0.0

    # Accepted / Drafted throughput (vLLM semantics over decode phase)
    accepted_throughput = total_draft_accepted / decode_s if decode_s > 0 else 0.0
    drafted_throughput = total_draft / decode_s if decode_s > 0 else 0.0

    return {
        "ttft_s": ttft_s,
        "decode_tps": (decode_tokens / decode_s) if decode_s > 0 else 0.0,
        "total_tokens": len(generated),
        "total_accepted": total_accepted,
        "total_draft": total_draft,
        "total_draft_accepted": total_draft_accepted,
        "draft_accept_counts_by_pos": accept_by_pos,
        "drafted_counts_by_pos": drafted_by_pos,
        "draft_accept_rate_by_pos": [
            (cnt / num_rounds) if num_rounds else 0.0 for cnt in accept_by_pos
        ],
        "per_pos_accept_rate": per_pos_accept_rate,
        "num_rounds": num_rounds,
        "tokens_per_round": (total_accepted / num_rounds) if num_rounds else 0.0,
        "draft_accept_rate": (
            total_draft_accepted / total_draft if total_draft else 0.0
        ),
        "mean_acceptance_length": mean_acceptance_length,
        "accepted_throughput": accepted_throughput,
        "drafted_throughput": drafted_throughput,
    }


def _bench_spec_prompt_set(
    llm: LLM,
    prompts: Sequence[Sequence[int]],
    output_len: int,
) -> Dict[str, Any]:
    reset_cuda_mem_stats()
    spec = llm._spec_decoder
    if spec is None:
        raise RuntimeError("spec decoder not initialized")

    per_request: List[Dict[str, Any]] = []
    total_output_tokens = 0
    total_accepted = 0
    total_draft = 0
    total_draft_accepted = 0
    total_rounds = 0
    accept_by_pos = [0] * spec.K
    drafted_by_pos = [0] * spec.K

    num_prompts = len(prompts)

    cuda_sync()
    t0 = time.perf_counter()
    for prompt_ids in prompts:
        r = _bench_spec_single(llm, list(prompt_ids), output_len)
        per_request.append(r)
        total_output_tokens += int(r["total_tokens"])
        total_accepted += int(r["total_accepted"])
        total_draft += int(r["total_draft"])
        total_draft_accepted += int(r["total_draft_accepted"])
        total_rounds += int(r["num_rounds"])
        for i, cnt in enumerate(r["draft_accept_counts_by_pos"]):
            accept_by_pos[i] += int(cnt)
        for i, cnt in enumerate(r.get("drafted_counts_by_pos", [])):
            if i < spec.K:
                drafted_by_pos[i] += int(cnt)
    cuda_sync()
    elapsed = time.perf_counter() - t0

    n = max(len(per_request), 1)

    # Per-position conditional acceptance rate (vLLM: "Per-position acceptance rate")
    per_pos_accept_rate: List[Optional[float]] = []
    for pos in range(spec.K):
        denom = drafted_by_pos[pos]
        if denom > 0:
            per_pos_accept_rate.append(accept_by_pos[pos] / denom)
        else:
            per_pos_accept_rate.append(None)

    # Mean acceptance length (vLLM: "Mean acceptance length")
    mean_acceptance_length = total_draft_accepted / total_rounds if total_rounds else 0.0

    # Accepted / Drafted throughput (vLLM semantics, computed over wall-clock decode phase)
    total_decode_s = elapsed  # elapsed covers the full decode loop including prefill per-request
    accepted_throughput = total_draft_accepted / total_decode_s if total_decode_s > 0 else 0.0
    drafted_throughput = total_draft / total_decode_s if total_decode_s > 0 else 0.0

    return {
        "K": spec.K,
        "throughput": throughput_metrics(elapsed, num_prompts, total_output_tokens),
        "per_request_mean_ttft_s": sum(x["ttft_s"] for x in per_request) / n,
        "per_request_mean_decode_tps": sum(x["decode_tps"] for x in per_request) / n,
        "total_accepted": total_accepted,
        "total_draft": total_draft,
        "total_draft_accepted": total_draft_accepted,
        "draft_accept_counts_by_pos": accept_by_pos,
        "drafted_counts_by_pos": drafted_by_pos,
        "draft_accept_rate_by_pos": [
            (cnt / total_rounds) if total_rounds else 0.0 for cnt in accept_by_pos
        ],
        "per_pos_accept_rate": per_pos_accept_rate,
        "num_rounds": total_rounds,
        "tokens_per_round": (total_accepted / total_rounds) if total_rounds else 0.0,
        "draft_accept_rate": (
            total_draft_accepted / total_draft if total_draft else 0.0
        ),
        "mean_acceptance_length": mean_acceptance_length,
        "accepted_throughput": accepted_throughput,
        "drafted_throughput": drafted_throughput,
        "peak_mem_gb": peak_mem_gb(),
    }


def parse_k_values(raw: str) -> List[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("--k-values 不能为空")
    if any(k <= 0 for k in values):
        raise ValueError("--k-values 需要全部为正整数")
    return values


def run_spec(
    target_model: str,
    draft_model: str,
    shared: SharedArgs,
    k_values: List[int],
) -> Dict[str, Any]:
    max_k = max(k_values)
    shared.validate(max_k=max_k)
    tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
    rng = random.Random(shared.bench_seed)
    prompts = _make_prompt_list(tokenizer, shared, rng)

    llm = LLM(
        model_path=target_model,
        tensor_parallel_size=shared.tensor_parallel_size,
        dtype="bfloat16",
        device="cuda",
        num_gpu_blocks=shared.num_gpu_blocks,
        draft_model_path=draft_model,
        num_speculative_tokens=max_k,
        max_seq_len=shared.max_model_len,
    )

    sweep: List[Dict[str, Any]] = []
    for k in k_values:
        llm._spec_decoder.K = k
        result = _bench_spec_prompt_set(
            llm=llm,
            prompts=prompts,
            output_len=shared.output_len,
        )
        sweep.append(result)

    return {
        "mode": "spec",
        "target_model": target_model,
        "draft_model": draft_model,
        "config": {
            **asdict(shared),
            "k_values": k_values,
        },
        "sweep": sweep,
    }
