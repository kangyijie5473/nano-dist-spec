"""Benchmark harness for nano-dist-spec.

Four modes, each emitting structured JSON to ``bench_results/`` for later
post-processing into the resume tables:

    python bench.py basic --model <path> --prompt-len 128 --max-tokens 256
    python bench.py spec  --target <7B> --draft <1.5B> --K-sweep 1,2,3,4,5,6,7,8
    python bench.py batch --model <path> --batch-sizes 1,2,4,8,16,32
    python bench.py kv-utilization --model <path>

Why bypass ``LLM.generate``? The user-facing API conflates prefill and decode
into a single wall-clock figure. To split TTFT (= prefill + first sample) from
the steady-state decode tokens/s we drive ``LLMEngine`` primitives directly.
This also lets us count speculative draft / accept events per step without
patching the engine.

Each ``bench_*`` helper performs a single run; the outer loop adds a warmup
run plus N measurement runs and reports mean / std.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from nano_dist_spec import LLM, LLMEngine, SamplingParams
from nano_dist_spec.config import CacheConfig

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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
    """Build a token id list of EXACTLY ``length`` tokens.

    We don't care about content fidelity — benchmarks measure throughput, not
    output quality — so we encode a long fairy-tale snippet, tile it, and
    truncate. A BOS token (if the tokenizer has one) is prepended so the
    model isn't fed completely raw text.
    """
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


def reset_scheduler(engine: LLMEngine) -> None:
    """Free any leftover KV blocks and clear scheduler queues between runs."""
    for sid in list(engine.scheduler.running.keys()):
        engine.kv_mgr.free_seq(sid)
    for sid in list(engine.scheduler.waiting):
        pass
    engine.scheduler.waiting.clear()
    engine.scheduler.running.clear()
    engine.scheduler.finished.clear()


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def write_results(mode: str, payload: Dict[str, Any], out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"{mode}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return str(path)


# ---------------------------------------------------------------------------
# Mode: basic — single prompt, manual prefill + decode timing
# ---------------------------------------------------------------------------


def bench_basic_one_run(
    engine: LLMEngine, prompt_ids: List[int], max_tokens: int,
) -> Dict[str, Any]:
    """One pass of greedy prefill+decode, with TTFT split out from decode tps.

    TTFT here = prefill forward pass + the first sample. Decode tps is then
    measured over the next (max_tokens - 1) decode steps. We disable EOS
    by using a sentinel id so the run always reaches ``max_tokens`` — that
    keeps the figure stable across temperatures and prompts.
    """
    reset_cuda_mem_stats()
    reset_scheduler(engine)

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    seq_id = engine.scheduler.add_request(prompt_ids, max_tokens=max_tokens)

    sched_out = engine.scheduler.schedule()
    assert len(sched_out.prefill_seqs) == 1, "expected exactly one prefill seq"
    seq = sched_out.prefill_seqs[0]

    cuda_sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        engine._prefill_seq(seq, params)
    cuda_sync()
    ttft_s = time.perf_counter() - t0
    engine.scheduler.after_step([seq], [])

    n_decode = 0
    cuda_sync()
    t1 = time.perf_counter()
    with torch.inference_mode():
        while len(seq.generated_token_ids) < max_tokens:
            engine._decode_batch([seq], params)
            n_decode += 1
    cuda_sync()
    decode_time = time.perf_counter() - t1
    decode_tps = n_decode / decode_time if decode_time > 0 else 0.0

    mem_gb = peak_mem_gb()

    engine.kv_mgr.free_seq(seq_id)
    engine.scheduler.running.pop(seq_id, None)

    return {
        "ttft_s": ttft_s,
        "decode_tps": decode_tps,
        "decode_tokens": n_decode,
        "decode_time_s": decode_time,
        "total_tokens": n_decode + 1,
        "peak_mem_gb": mem_gb,
    }


def run_basic(args) -> Dict[str, Any]:
    print(f"[basic] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    dt_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    engine = LLMEngine(
        model_path=args.model,
        tp_size=1,
        dtype=dt_map[args.dtype],
        device="cuda",
        cache_config=CacheConfig(num_gpu_blocks=args.num_gpu_blocks),
    )

    prompt_ids = make_token_ids(tokenizer, args.prompt_len)
    print(f"[basic] prompt_len={len(prompt_ids)}, max_tokens={args.max_tokens}")

    runs: List[Dict[str, Any]] = []
    for i in range(args.warmup):
        print(f"[basic] warmup {i + 1}/{args.warmup}")
        bench_basic_one_run(engine, prompt_ids, args.max_tokens)

    for i in range(args.runs):
        r = bench_basic_one_run(engine, prompt_ids, args.max_tokens)
        r["run"] = i
        runs.append(r)
        print(
            f"[basic] run {i + 1}/{args.runs}  "
            f"ttft={r['ttft_s'] * 1000:.1f}ms  "
            f"decode={r['decode_tps']:.1f} tok/s  "
            f"peak_mem={r['peak_mem_gb']:.2f}GB"
        )

    summary = {
        "ttft_s": summarize([r["ttft_s"] for r in runs]),
        "decode_tps": summarize([r["decode_tps"] for r in runs]),
        "peak_mem_gb": summarize([r["peak_mem_gb"] for r in runs]),
    }

    return {
        "mode": "basic",
        "model": args.model,
        "config": {
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "results": runs,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Mode: spec — speculative decoding K-sweep with acceptance + speedup
# ---------------------------------------------------------------------------


def bench_spec_one_run(
    llm: LLM, prompt_ids: List[int], max_tokens: int, temperature: float,
) -> Dict[str, Any]:
    """One speculative generation, exposing accept counts + timings.

    We mirror ``LLM._generate_speculative`` but instrument the inner loop so
    we can return ``total_accepted`` / ``total_draft`` and split TTFT from
    decode.
    """
    reset_cuda_mem_stats()

    spec = llm._spec_decoder
    assert spec is not None, "spec decoder not initialized"
    eos_id = -1  # disable early stop
    seq_id = 0
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

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
    total_draft_accepted = 0  # excluding bonus, used for "draft acceptance rate"
    num_rounds = 0
    K = spec.K

    cuda_sync()
    t1 = time.perf_counter()
    while len(generated) < max_tokens:
        if generated[-1] == eos_id:
            break
        out, saved_probs = spec.speculative_step(
            seq_id, generated[-1], saved_probs, params,
        )
        generated.extend(out.accepted_tokens)
        total_accepted += out.num_accepted
        total_draft += out.num_draft_tokens
        # When all K drafts accepted: num_accepted == K + 1 (bonus token).
        # Otherwise the last token in `accepted` is a correction (not from draft).
        if out.num_accepted > out.num_draft_tokens:
            total_draft_accepted += out.num_draft_tokens
        else:
            total_draft_accepted += max(out.num_accepted - 1, 0)
        num_rounds += 1
    cuda_sync()
    decode_time = time.perf_counter() - t1

    generated = generated[:max_tokens]
    n_decode = len(generated) - 1
    decode_tps = n_decode / decode_time if decode_time > 0 else 0.0
    mem_gb = peak_mem_gb()

    llm.engine.kv_mgr.free_seq(seq_id)
    spec.draft_mgr.free_seq(seq_id)

    return {
        "K": K,
        "ttft_s": ttft_s,
        "decode_tps": decode_tps,
        "decode_tokens": n_decode,
        "decode_time_s": decode_time,
        "total_tokens": len(generated),
        "total_accepted": total_accepted,
        "total_draft": total_draft,
        "total_draft_accepted": total_draft_accepted,
        "num_rounds": num_rounds,
        "tokens_per_round": (total_accepted / num_rounds) if num_rounds else 0.0,
        "draft_accept_rate": (
            total_draft_accepted / total_draft if total_draft else 0.0
        ),
        "peak_mem_gb": mem_gb,
    }


def run_spec(args) -> Dict[str, Any]:
    print(f"[spec] target={args.target}  draft={args.draft}")
    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    prompt_ids = make_token_ids(tokenizer, args.prompt_len)
    print(f"[spec] prompt_len={len(prompt_ids)}  max_tokens={args.max_tokens}")

    K_list = [int(k) for k in args.K_sweep.split(",") if k.strip()]
    temps = [float(t) for t in args.temperatures.split(",") if t.strip()]

    # ---- Baseline: target-only autoregressive decode (for speedup ratio) ----
    baseline: Dict[str, Any] = {}
    if args.baseline:
        print("[spec] running target-only baseline")
        baseline_engine = LLMEngine(
            model_path=args.target,
            tp_size=1,
            dtype=torch.bfloat16,
            device="cuda",
            cache_config=CacheConfig(num_gpu_blocks=args.num_gpu_blocks),
        )
        for i in range(args.warmup):
            print(f"[spec] baseline warmup {i + 1}/{args.warmup}")
            bench_basic_one_run(baseline_engine, prompt_ids, args.max_tokens)
        bruns = []
        for i in range(args.runs):
            r = bench_basic_one_run(baseline_engine, prompt_ids, args.max_tokens)
            r["run"] = i
            bruns.append(r)
            print(
                f"[spec] baseline run {i + 1}/{args.runs}  "
                f"decode={r['decode_tps']:.1f} tok/s  "
                f"ttft={r['ttft_s'] * 1000:.1f}ms"
            )
        baseline = {
            "results": bruns,
            "summary": {
                "ttft_s": summarize([r["ttft_s"] for r in bruns]),
                "decode_tps": summarize([r["decode_tps"] for r in bruns]),
                "peak_mem_gb": summarize([r["peak_mem_gb"] for r in bruns]),
            },
        }
        del baseline_engine
        torch.cuda.empty_cache()

    # Reuse a single LLM across the full sweep — only K changes between runs,
    # and K lives on `_spec_decoder.K`. Temperature is purely a sampling param.
    print("[spec] loading target+draft once for the full sweep")
    llm = LLM(
        args.target,
        dtype="bfloat16",
        num_gpu_blocks=args.num_gpu_blocks,
        draft_model_path=args.draft,
        num_speculative_tokens=max(K_list),
    )

    sweep: List[Dict[str, Any]] = []
    for temperature in temps:
        for K in K_list:
            print(f"[spec] === K={K}  temperature={temperature} ===")
            llm._spec_decoder.K = K

            for i in range(args.warmup):
                print(f"[spec]   warmup {i + 1}/{args.warmup}")
                bench_spec_one_run(llm, prompt_ids, args.max_tokens, temperature)

            runs: List[Dict[str, Any]] = []
            for i in range(args.runs):
                r = bench_spec_one_run(llm, prompt_ids, args.max_tokens, temperature)
                r["run"] = i
                runs.append(r)
                print(
                    f"[spec]   run {i + 1}/{args.runs}  "
                    f"decode={r['decode_tps']:.1f} tok/s  "
                    f"draft_accept={r['draft_accept_rate']:.2%}  "
                    f"tok/round={r['tokens_per_round']:.2f}  "
                    f"peak_mem={r['peak_mem_gb']:.2f}GB"
                )

            entry = {
                "K": K,
                "temperature": temperature,
                "results": runs,
                "summary": {
                    "ttft_s": summarize([r["ttft_s"] for r in runs]),
                    "decode_tps": summarize([r["decode_tps"] for r in runs]),
                    "draft_accept_rate": summarize(
                        [r["draft_accept_rate"] for r in runs]
                    ),
                    "tokens_per_round": summarize(
                        [r["tokens_per_round"] for r in runs]
                    ),
                    "peak_mem_gb": summarize([r["peak_mem_gb"] for r in runs]),
                },
            }
            if baseline:
                entry["speedup_vs_baseline"] = (
                    entry["summary"]["decode_tps"]["mean"]
                    / max(baseline["summary"]["decode_tps"]["mean"], 1e-6)
                )
            sweep.append(entry)

    del llm
    torch.cuda.empty_cache()

    return {
        "mode": "spec",
        "target": args.target,
        "draft": args.draft,
        "config": {
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "K_sweep": K_list,
            "temperatures": temps,
            "warmup": args.warmup,
            "runs": args.runs,
            "num_gpu_blocks": args.num_gpu_blocks,
        },
        "baseline": baseline,
        "sweep": sweep,
    }


# ---------------------------------------------------------------------------
# Mode: batch — continuous-batching throughput at varying batch sizes
# ---------------------------------------------------------------------------


def bench_batch_one_run(
    engine: LLMEngine,
    prompts_ids_list: List[List[int]],
    max_tokens: int,
) -> Dict[str, Any]:
    reset_cuda_mem_stats()
    reset_scheduler(engine)

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    seq_ids: List[int] = []
    for ids in prompts_ids_list:
        sid = engine.scheduler.add_request(ids, max_tokens=max_tokens)
        seq_ids.append(sid)

    cuda_sync()
    t_total = time.perf_counter()
    t_first_decode: Optional[float] = None
    n_prefill_steps = 0
    n_decode_steps = 0

    with torch.inference_mode():
        while engine.scheduler.has_pending:
            sched_out = engine.scheduler.schedule()
            for seq in sched_out.prefill_seqs:
                engine._prefill_seq(seq, params)
                n_prefill_steps += 1
            if sched_out.decode_seqs:
                if t_first_decode is None:
                    cuda_sync()
                    t_first_decode = time.perf_counter()
                engine._decode_batch(sched_out.decode_seqs, params)
                n_decode_steps += 1
            finished = []
            for seq in list(engine.scheduler.running.values()):
                if len(seq.generated_token_ids) >= seq.max_tokens:
                    finished.append(seq.seq_id)
            engine.scheduler.after_step(sched_out.prefill_seqs, finished)

    cuda_sync()
    elapsed = time.perf_counter() - t_total

    total_gen_tokens = sum(
        len(s.generated_token_ids) for s in engine.scheduler.finished.values()
    )
    aggregate_tps = total_gen_tokens / elapsed if elapsed > 0 else 0.0
    mem_gb = peak_mem_gb()

    engine.scheduler.finished.clear()

    return {
        "batch_size": len(prompts_ids_list),
        "elapsed_s": elapsed,
        "total_tokens": total_gen_tokens,
        "aggregate_tps": aggregate_tps,
        "n_prefill_steps": n_prefill_steps,
        "n_decode_steps": n_decode_steps,
        "peak_mem_gb": mem_gb,
    }


def run_batch(args) -> Dict[str, Any]:
    print(f"[batch] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine = LLMEngine(
        model_path=args.model,
        tp_size=1,
        dtype=torch.bfloat16,
        device="cuda",
        cache_config=CacheConfig(num_gpu_blocks=args.num_gpu_blocks),
    )

    prompt_ids = make_token_ids(tokenizer, args.prompt_len)
    batch_sizes = [int(b) for b in args.batch_sizes.split(",") if b.strip()]

    print(
        f"[batch] prompt_len={len(prompt_ids)}  max_tokens={args.max_tokens}  "
        f"batch_sizes={batch_sizes}"
    )

    sweep: List[Dict[str, Any]] = []
    for bs in batch_sizes:
        prompts = [list(prompt_ids) for _ in range(bs)]
        for i in range(args.warmup):
            print(f"[batch] bs={bs} warmup {i + 1}/{args.warmup}")
            bench_batch_one_run(engine, prompts, args.max_tokens)
        runs = []
        for i in range(args.runs):
            r = bench_batch_one_run(engine, prompts, args.max_tokens)
            r["run"] = i
            runs.append(r)
            print(
                f"[batch] bs={bs} run {i + 1}/{args.runs}  "
                f"throughput={r['aggregate_tps']:.1f} tok/s  "
                f"elapsed={r['elapsed_s']:.2f}s  "
                f"peak_mem={r['peak_mem_gb']:.2f}GB"
            )
        sweep.append({
            "batch_size": bs,
            "results": runs,
            "summary": {
                "aggregate_tps": summarize([r["aggregate_tps"] for r in runs]),
                "elapsed_s": summarize([r["elapsed_s"] for r in runs]),
                "peak_mem_gb": summarize([r["peak_mem_gb"] for r in runs]),
            },
        })

    return {
        "mode": "batch",
        "model": args.model,
        "config": {
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "batch_sizes": batch_sizes,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "sweep": sweep,
    }


# ---------------------------------------------------------------------------
# Mode: kv-utilization — paged vs naive contiguous allocation
# ---------------------------------------------------------------------------


def run_kv_utilization(args) -> Dict[str, Any]:
    """Measure KV cache memory savings from paging vs naive allocation.

    Method:
      1. Submit a batch of prompts of varied lengths.
      2. At every scheduler step, snapshot ``allocator.num_blocks - num_free``
         (peak blocks actually held in flight by the paged allocator).
      3. Naive baseline = num_seqs * ceil(max_seq_len / block_size), where
         max_seq_len = max(prompt_len) + max_tokens. This is what a naive
         contiguous allocator would have to reserve for every sequence (the
         worst case across the batch) to avoid copies.
      4. Report ratio paged / naive.
    """
    print(f"[kv-utilization] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine = LLMEngine(
        model_path=args.model,
        tp_size=1,
        dtype=torch.bfloat16,
        device="cuda",
        cache_config=CacheConfig(num_gpu_blocks=args.num_gpu_blocks),
    )

    prompt_lens = [int(x) for x in args.prompt_lens.split(",") if x.strip()]
    prompts = [make_token_ids(tokenizer, L) for L in prompt_lens]
    max_tokens = args.max_tokens
    block_size = engine.block_size

    reset_cuda_mem_stats()
    reset_scheduler(engine)

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    for ids in prompts:
        engine.scheduler.add_request(ids, max_tokens=max_tokens)

    peak_blocks_used = 0
    sample_trace: List[int] = []
    total_blocks = engine.allocator.num_blocks

    with torch.inference_mode():
        while engine.scheduler.has_pending:
            cur = total_blocks - engine.allocator.num_free
            peak_blocks_used = max(peak_blocks_used, cur)
            sample_trace.append(cur)

            sched_out = engine.scheduler.schedule()
            for seq in sched_out.prefill_seqs:
                engine._prefill_seq(seq, params)
            if sched_out.decode_seqs:
                engine._decode_batch(sched_out.decode_seqs, params)
            finished = []
            for seq in list(engine.scheduler.running.values()):
                if len(seq.generated_token_ids) >= seq.max_tokens:
                    finished.append(seq.seq_id)
            engine.scheduler.after_step(sched_out.prefill_seqs, finished)

    final_used = total_blocks - engine.allocator.num_free
    peak_blocks_used = max(peak_blocks_used, final_used)

    max_seq_len = max(prompt_lens) + max_tokens
    naive_blocks_per_seq = math.ceil(max_seq_len / block_size)
    naive_blocks_total = naive_blocks_per_seq * len(prompts)

    # Per-token bytes: K and V tensors, num_layers, num_kv_heads, head_dim, dtype
    cfg = engine.model_config
    bytes_per_dtype = 2  # bfloat16
    bytes_per_token = (
        2  # K + V
        * cfg.num_hidden_layers
        * cfg.num_key_value_heads
        * cfg.head_dim
        * bytes_per_dtype
    )
    paged_bytes = peak_blocks_used * block_size * bytes_per_token
    naive_bytes = naive_blocks_total * block_size * bytes_per_token

    return {
        "mode": "kv-utilization",
        "model": args.model,
        "config": {
            "prompt_lens": prompt_lens,
            "max_tokens": max_tokens,
            "block_size": block_size,
            "num_gpu_blocks": total_blocks,
            "num_seqs": len(prompts),
        },
        "results": {
            "peak_blocks_used_paged": peak_blocks_used,
            "naive_blocks_required": naive_blocks_total,
            "paged_naive_block_ratio": peak_blocks_used / naive_blocks_total
            if naive_blocks_total else 0.0,
            "memory_savings_pct": (
                (1.0 - peak_blocks_used / naive_blocks_total) * 100.0
                if naive_blocks_total else 0.0
            ),
            "paged_bytes": paged_bytes,
            "naive_bytes": naive_bytes,
            "bytes_per_token": bytes_per_token,
        },
        "trace": sample_trace[::max(1, len(sample_trace) // 64)],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="nano-dist-spec benchmark harness")
    p.add_argument("--out-dir", default="bench_results")
    sub = p.add_subparsers(dest="mode", required=True)

    pb = sub.add_parser("basic", help="single-prompt TTFT + decode tps")
    pb.add_argument("--model", required=True)
    pb.add_argument("--prompt-len", type=int, default=128)
    pb.add_argument("--max-tokens", type=int, default=256)
    pb.add_argument("--runs", type=int, default=3)
    pb.add_argument("--warmup", type=int, default=1)
    pb.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    pb.add_argument("--num-gpu-blocks", type=int, default=None)

    ps = sub.add_parser("spec", help="speculative decoding K-sweep")
    ps.add_argument("--target", required=True)
    ps.add_argument("--draft", required=True)
    ps.add_argument("--K-sweep", default="5",
                    help="comma-separated list of K values")
    ps.add_argument("--temperatures", default="0.0",
                    help="comma-separated list of temperatures to sweep")
    ps.add_argument("--prompt-len", type=int, default=128)
    ps.add_argument("--max-tokens", type=int, default=128)
    ps.add_argument("--runs", type=int, default=3)
    ps.add_argument("--warmup", type=int, default=1)
    ps.add_argument("--num-gpu-blocks", type=int, default=4000)
    ps.add_argument("--baseline", action="store_true",
                    help="also run target-only baseline for speedup")

    pba = sub.add_parser("batch", help="continuous-batching throughput sweep")
    pba.add_argument("--model", required=True)
    pba.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    pba.add_argument("--prompt-len", type=int, default=128)
    pba.add_argument("--max-tokens", type=int, default=128)
    pba.add_argument("--runs", type=int, default=2)
    pba.add_argument("--warmup", type=int, default=1)
    pba.add_argument("--num-gpu-blocks", type=int, default=None)

    pkv = sub.add_parser("kv-utilization", help="paged-vs-naive KV memory ratio")
    pkv.add_argument("--model", required=True)
    pkv.add_argument(
        "--prompt-lens",
        default="32,128,512,2048,32,128,256,768",
        help="comma-separated prompt lengths in the same batch",
    )
    pkv.add_argument("--max-tokens", type=int, default=64)
    pkv.add_argument("--num-gpu-blocks", type=int, default=None)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "basic":
        payload = run_basic(args)
    elif args.mode == "spec":
        payload = run_spec(args)
    elif args.mode == "batch":
        payload = run_batch(args)
    elif args.mode == "kv-utilization":
        payload = run_kv_utilization(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    out_path = write_results(args.mode, payload, args.out_dir)
    print(f"\nResults written to {out_path}")
    print(json.dumps(payload.get("summary") or payload.get("sweep")
                     or payload.get("results"), indent=2, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
