"""Speculative decoding example: small draft model + large target model.

Usage:
    python examples/speculative_decode.py \
        --target /path/to/Qwen3-1.7B \
        --draft  /path/to/Qwen3-0.6B \
        --prompt "用一句话介绍你自己"

The draft model generates K candidate tokens autoregressively, the target
model verifies them in a single forward pass, and rejection sampling ensures
the output distribution matches the target model exactly.

The target model's tokenizer is used to apply the chat template — target and
draft are expected to share the same tokenizer (this is true for model pairs
within the same family, e.g. DeepSeek-R1-Distill-Qwen 7B + 1.5B).
"""

import argparse
import time

from transformers import AutoTokenizer

from nano_dist_spec import LLM, SamplingParams


def build_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
    """Wrap raw user text in the model's chat template when available.

    DeepSeek-R1 series additionally requires `<think>\\n` to be prefilled
    after `<｜Assistant｜>` to reliably enter the reasoning chain.
    """
    if tokenizer.chat_template is None:
        return user_text

    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if prompt.endswith("<｜Assistant｜>"):
        prompt += "<think>\n"

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="Large target model")
    parser.add_argument("--draft", type=str, required=True, help="Small draft model")
    parser.add_argument(
        "--prompt", type=str, default="用一句话介绍你自己",
        help="Raw user question; chat template will be applied automatically.",
    )
    parser.add_argument("--K", type=int, default=5, help="Speculative tokens per round")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Target model: {args.target}")
    print(f"Draft model:  {args.draft}")
    print(f"Speculative tokens per round: {args.K}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
    prompt = build_prompt(tokenizer, args.prompt)

    llm = LLM(
        args.target,
        dtype="bfloat16",
        num_gpu_blocks=4000,  # 避免OOM 限制KVcache
        draft_model_path=args.draft,
        num_speculative_tokens=args.K,
    )

    params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

    start = time.perf_counter()
    outputs = llm.generate([prompt], params)
    elapsed = time.perf_counter() - start

    for out in outputs:
        n_tokens = len(out.token_ids)
        print(f"Prompt:  {out.prompt}")
        print(f"Output:  {out.text}")
        print(f"Tokens:  {n_tokens}")
        print(f"Time:    {elapsed:.2f}s")
        print(f"Speed:   {n_tokens / elapsed:.1f} tok/s")


if __name__ == "__main__":
    main()
