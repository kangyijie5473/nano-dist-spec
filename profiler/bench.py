"""Minimal benchmark CLI aligned with vLLM throughput script.

Only two modes are kept:
  - basic: target-only throughput
  - spec:  target + draft speculative throughput with K sweep
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from bench_core import SharedArgs, parse_k_values, run_basic, run_spec


def write_results(mode: str, payload: Dict[str, Any], out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"{mode}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return str(path)


def add_shared_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-gpu-blocks", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=("random", "fixed"),
        default="random",
        help="random: vLLM-style synthetic prompts (--random-range-ratio 0); fixed: legacy repeating text",
    )
    parser.add_argument(
        "--bench-seed",
        type=int,
        default=42,
        help="RNG seed for random prompts (same seed → same prompt list across runs/spec K sweep)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nano-dist-spec benchmark harness")
    parser.add_argument("--out-dir", default="bench_results")
    sub = parser.add_subparsers(dest="mode", required=True)

    basic = sub.add_parser("basic", help="target-only throughput")
    basic.add_argument("--model", required=True)
    add_shared_cli_args(basic)

    spec = sub.add_parser("spec", help="speculative decoding throughput sweep")
    spec.add_argument("--target-model", required=True)
    spec.add_argument("--draft-model", required=True)
    spec.add_argument("--k-values", default="1,2,3,4,5,6,7")
    add_shared_cli_args(spec)

    return parser


def to_shared_args(args: argparse.Namespace) -> SharedArgs:
    return SharedArgs(
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        num_gpu_blocks=args.num_gpu_blocks,
        prompt_mode=args.prompt_mode,
        bench_seed=args.bench_seed,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    shared = to_shared_args(args)

    if args.mode == "basic":
        payload = run_basic(model=args.model, shared=shared)
    elif args.mode == "spec":
        payload = run_spec(
            target_model=args.target_model,
            draft_model=args.draft_model,
            shared=shared,
            k_values=parse_k_values(args.k_values),
        )
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    out_path = write_results(args.mode, payload, args.out_dir)
    print(f"\nResults written to {out_path}")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
