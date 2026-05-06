"""Match ``profiler/bench.py basic`` metrics against vLLM (streaming).

- Same prompt construction as ``bench.make_token_ids`` (128 tokens by default).
- Greedy ``temperature=0``, ``ignore_eos=True``, fixed ``max_tokens``.
- **ttft_s**: wall time from request start until first generated token is observed.
- **decode_tps**: ``(max_tokens - 1) / (t_last - t_first)`` where ``t_first`` is the
  timestamp when cumulative output length first reaches 1 and ``t_last`` when it
  reaches ``max_tokens`` — aligned with nano's 127 decode steps after the first token.

Run (vLLM conda env)::

    conda activate vllm
    python profiler/vllm_basic_match_bench.py \\
        --model /model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


def make_token_ids(tokenizer, length: int) -> List[int]:
    """Same construction as ``profiler/bench.make_token_ids``."""
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


async def one_streaming_run(
    llm: AsyncLLM,
    prompt_ids: List[int],
    max_tokens: int,
) -> Dict[str, Any]:
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)

    t_start = time.perf_counter()
    t_first: float | None = None
    t_at_target: float | None = None
    prev_n = 0

    rid = f"bench-{uuid.uuid4().hex}"
    async for out in llm.generate(prompt, sp, request_id=rid):
        if not out.outputs:
            continue
        n = len(out.outputs[0].token_ids)
        if n <= prev_n:
            continue
        now = time.perf_counter()
        if t_first is None and n >= 1:
            t_first = now
        if n >= max_tokens:
            t_at_target = now
        prev_n = n

    if t_first is None or t_at_target is None:
        raise RuntimeError(
            f"incomplete stream: first={t_first}, at_max={t_at_target}, last_n={prev_n}",
        )

    ttft_s = t_first - t_start
    decode_time = t_at_target - t_first
    n_decode = max_tokens - 1
    decode_tps = n_decode / decode_time if decode_time > 0 else 0.0

    return {
        "ttft_s": ttft_s,
        "decode_tps": decode_tps,
        "decode_tokens": n_decode,
        "decode_time_s": decode_time,
        "total_tokens": max_tokens,
    }


async def async_main(args: argparse.Namespace) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_ids = make_token_ids(tokenizer, args.prompt_len)

    engine_args = AsyncEngineArgs(
        model=args.model,
        tokenizer=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
    )
    llm = AsyncLLM.from_engine_args(engine_args)

    runs: List[Dict[str, Any]] = []
    try:
        for i in range(args.warmup):
            print(f"[vllm-basic] warmup {i + 1}/{args.warmup}")
            await one_streaming_run(llm, prompt_ids, args.max_tokens)

        for i in range(args.runs):
            r = await one_streaming_run(llm, prompt_ids, args.max_tokens)
            r["run"] = i
            runs.append(r)
            print(
                f"[vllm-basic] run {i + 1}/{args.runs}  "
                f"ttft={r['ttft_s'] * 1000:.1f}ms  "
                f"decode={r['decode_tps']:.1f} tok/s",
            )
    finally:
        llm.shutdown()

    summary = {
        "ttft_s": summarize([r["ttft_s"] for r in runs]),
        "decode_tps": summarize([r["decode_tps"] for r in runs]),
    }
    return {
        "mode": "vllm-basic-match",
        "model": args.model,
        "config": {
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "results": runs,
        "summary": summary,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM basic metrics aligned with nano bench.py")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    p.add_argument("--out-dir", default="bench_results")
    args = p.parse_args()

    payload = asyncio.run(async_main(args))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"vllm_basic_match_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nResults written to {path}")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
