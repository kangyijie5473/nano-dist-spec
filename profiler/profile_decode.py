"""Localize the decode-tps bottleneck via torch.profiler.

What this answers
-----------------
Given that single-model decode tps (~28 tok/s) is much lower than the back-of-
the-envelope estimate (80-150 tok/s on a 4090 + bf16 + 1.5B model), we want
runtime evidence for the claim "Python overhead dominates" rather than just
benching wall-clock numbers.

torch.profiler is the right tool: it tracks **CPU and CUDA on two timelines
simultaneously**, so the question "which one is the bottleneck" gets a direct
answer. We further split each decode step into named segments via
``record_function`` so the table maps cleanly onto the body of
``LLMEngine._decode_batch``.

How to read the output
----------------------
1. Two summary tables are printed to stdout:
   - sorted by ``self_cpu_time_total`` (CPU hot spots)
   - sorted by ``self_cuda_time_total`` (CUDA hot spots)
   Compare the column totals at the bottom: if total CPU >> total CUDA,
   Python orchestration dominates wall-clock and you can't speed it up by
   making the GPU "faster".

2. A Chrome trace is written to ``profiler/traces/decode_<ts>.json``.
   Open it in either:
     - Local:  chrome://tracing/  (drag-drop the JSON)
     - Online: https://ui.perfetto.dev/  (drag-drop the JSON)
   Look at the GPU-stream row (usually the bottom track):
     - **dense, no white gaps** -> kernels back-to-back, GPU is saturated.
       Optimize the kernels themselves (FlashAttention, torch.compile, ...).
     - **frequent white gaps**  -> CPU is still preparing the next kernel.
       Optimize Python overhead (CUDA Graph capture, pre-built slot tensors,
       remove ``.item()`` syncs from the hot path, batched H2D, ...).

3. Compare segment durations: the per-step ``record_function`` blocks
   (``compute_slot_mapping``, ``build_input_tensors``, ``model_forward``,
   ``sample``, ``item_sync``) appear in both the table and the trace. Watch
   for ``compute_slot_mapping + build_input_tensors`` rivalling or exceeding
   ``model_forward`` -- that is the smoking gun for the Python-overhead-dominates
   hypothesis logged in docs/DEV_LOG.md (#9).

Caveats
-------
- The profiler itself adds 5-10% overhead. Wall-clock numbers from this script
  are NOT comparable to bench.py output -- only the *ratios* between segments
  are meaningful.
- Always warm up first. The first few forwards trigger cuDNN autotune,
  CUDA context init, and kernel JIT, all of which would skew the captured
  segment that happens to land first.

Usage
-----
    python profiler/profile_decode.py
    python profiler/profile_decode.py --model <path> --steps 64
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import List

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer

from nano_dist_spec import LLMEngine, SamplingParams
from nano_dist_spec.attention import InputMetadata
from nano_dist_spec.config import CacheConfig
from nano_dist_spec.sampling import sample
from nano_dist_spec.scheduler import Sequence

DEFAULT_MODEL = "/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# ---------------------------------------------------------------------------
# Utilities (mirrors bench.py for consistency)
# ---------------------------------------------------------------------------


def make_token_ids(tokenizer, length: int) -> List[int]:
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


# ---------------------------------------------------------------------------
# Instrumented decode step
# ---------------------------------------------------------------------------
#
# This mirrors LLMEngine._decode_batch (nano_dist_spec/engine.py:172-206) for
# the single-sequence case, but wraps each substep in a record_function so
# the segments show up as named bands in the profiler table and Chrome trace.
#
# We keep this in the profiler script (rather than instrumenting the engine
# directly) so production code stays clean.
# ---------------------------------------------------------------------------


def instrumented_decode_step(
    engine: LLMEngine, seq: Sequence, params: SamplingParams,
) -> None:
    """Run one decode step with named segments. Mutates ``seq.generated_token_ids``."""
    seq_id = seq.seq_id

    # 1) Bookkeeping: extend the block table to fit one more token.
    with record_function("kv_append_slots"):
        engine.kv_mgr.append_slots(seq_id, 1)

    # 2) Compute the physical KV slot for the new token. Pure Python loop +
    #    list -> tensor + H2D copy. Suspect #1 for Python overhead.
    with record_function("compute_slot_mapping"):
        pos = engine.kv_mgr.context_lens[seq_id] - 1
        slot_mapping = engine.kv_mgr.compute_slot_mapping(
            seq_id, pos, 1, engine.device,
        )

    # 3) Build all the per-step input tensors. Several small H2D transfers,
    #    each with its own kernel launch overhead. Suspect #2.
    with record_function("build_input_tensors"):
        last_token = seq.generated_token_ids[-1]
        input_ids = torch.tensor([[last_token]], device=engine.device)
        positions = torch.tensor([[pos]], device=engine.device)
        block_tables = engine.kv_mgr.get_block_table_tensor([seq_id], engine.device)
        context_lens = engine.kv_mgr.get_context_lens_tensor([seq_id], engine.device)
        metadata = InputMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=engine.block_size,
        )
        kv_list = [
            engine.kv_cache.get_kv(i) for i in range(engine.kv_cache.num_layers)
        ]

    # 4) Actual transformer forward. The GPU's main job.
    with record_function("model_forward"):
        logits = engine.model(input_ids, positions, kv_list, metadata)

    # 5) Sampling. argmax / softmax / multinomial -- still on GPU.
    with record_function("sample"):
        new_token = sample(logits[:, -1, :], params)

    # 6) .item() forces a CUDA sync (D2H copy of a single int). This is the
    #    one unavoidable sync per step in the current design.
    with record_function("item_sync"):
        seq.generated_token_ids.append(new_token.item())


# ---------------------------------------------------------------------------
# Profile driver
# ---------------------------------------------------------------------------


def run_profile(args) -> Path:
    print(f"[profile] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    engine = LLMEngine(
        model_path=args.model,
        tp_size=1,
        dtype=torch.bfloat16,
        device="cuda",
        cache_config=CacheConfig(num_gpu_blocks=args.num_gpu_blocks),
    )

    prompt_ids = make_token_ids(tokenizer, args.prompt_len)
    print(
        f"[profile] prompt_len={len(prompt_ids)}  "
        f"warmup_steps={args.warmup_steps}  steps={args.steps}"
    )

    params = SamplingParams(temperature=0.0, max_tokens=args.steps + args.warmup_steps + 8)
    seq_id = engine.scheduler.add_request(prompt_ids, max_tokens=params.max_tokens)
    sched_out = engine.scheduler.schedule()
    assert len(sched_out.prefill_seqs) == 1
    seq = sched_out.prefill_seqs[0]

    # ---- Prefill once (NOT profiled; we only care about decode) ----
    with torch.inference_mode():
        engine._prefill_seq(seq, params)
    engine.scheduler.after_step([seq], [])

    # ---- Warmup decode steps (NOT profiled). Triggers cuDNN autotune,
    #      kernel JIT, and stabilizes CUDA context so the measured segments
    #      reflect steady-state behavior, not first-iteration overhead.
    print(f"[profile] warmup ({args.warmup_steps} decode steps)")
    with torch.inference_mode():
        for _ in range(args.warmup_steps):
            instrumented_decode_step(engine, seq, params)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ---- Profile measured decode steps ----
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    print(f"[profile] capturing {args.steps} decode steps via torch.profiler")
    t0 = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        with_stack=args.with_stack,
        profile_memory=False,
    ) as prof:
        with torch.inference_mode():
            with record_function("profile_decode_loop"):
                for _ in range(args.steps):
                    with record_function("decode_step"):
                        instrumented_decode_step(engine, seq, params)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(
        f"[profile] captured {args.steps} steps in {elapsed:.2f}s  "
        f"(profile-on tps ~ {args.steps / elapsed:.1f}, NOT comparable to bench.py)"
    )

    # ---- Print summary tables ----
    print()
    print("=" * 100)
    print("TOP 25 by self CPU time  (where the CPU spends its cycles)")
    print("=" * 100)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=25,
    ))

    print()
    print("=" * 100)
    print("TOP 25 by self CUDA time  (where the GPU spends its cycles)")
    print("=" * 100)
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=25,
    ))

    # ---- Decode-step segment summary -------------------------------------
    # Pull only the rows for our custom record_function segments and print a
    # compact table that is easy to read at a glance: each segment's total
    # CPU time, total CUDA time, and call count over `args.steps` decodes.
    segment_names = {
        "decode_step",
        "kv_append_slots",
        "compute_slot_mapping",
        "build_input_tensors",
        "model_forward",
        "sample",
        "item_sync",
    }
    print()
    print("=" * 100)
    print(f"Decode-step segments  ({args.steps} steps captured)")
    print("=" * 100)
    print(
        f"  {'segment':<24} {'count':>6}  "
        f"{'CPU total (ms)':>16}  {'CPU avg (us)':>14}  "
        f"{'CUDA total (ms)':>17}  {'CUDA avg (us)':>15}"
    )
    print("  " + "-" * 96)
    # In recent PyTorch, the cuda total attribute is ``device_time_total``
    # (was ``cuda_time_total`` in older versions). Fall back gracefully.
    def _cuda_total_us(evt) -> float:
        return float(getattr(evt, "device_time_total",
                             getattr(evt, "cuda_time_total", 0.0)))

    for evt in prof.key_averages():
        if evt.key not in segment_names:
            continue
        n = evt.count
        cpu_total_ms = evt.cpu_time_total / 1000.0
        cpu_avg_us = evt.cpu_time_total / max(n, 1)
        cuda_total_us = _cuda_total_us(evt)
        cuda_total_ms = cuda_total_us / 1000.0
        cuda_avg_us = cuda_total_us / max(n, 1)
        print(
            f"  {evt.key:<24} {n:>6}  "
            f"{cpu_total_ms:>16.2f}  {cpu_avg_us:>14.1f}  "
            f"{cuda_total_ms:>17.2f}  {cuda_avg_us:>15.1f}"
        )

    # ---- Export Chrome trace ---------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_path = out_dir / f"decode_{ts}.json"
    prof.export_chrome_trace(str(trace_path))
    print()
    print(f"Chrome trace -> {trace_path}")
    print("Open with: chrome://tracing/  (drag-drop)  or  https://ui.perfetto.dev/")

    # Cleanup
    engine.kv_mgr.free_seq(seq_id)
    engine.scheduler.running.pop(seq_id, None)

    return trace_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile the decode loop with torch.profiler"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=32,
                        help="number of decode steps captured by profiler")
    parser.add_argument("--num-gpu-blocks", type=int, default=2000)
    parser.add_argument("--out-dir", default="profiler/traces")
    parser.add_argument("--record-shapes", action="store_true",
                        help="record input shapes (small overhead, useful detail)")
    parser.add_argument("--with-stack", action="store_true",
                        help="capture Python call stacks (large overhead)")
    args = parser.parse_args()

    run_profile(args)


if __name__ == "__main__":
    main()
