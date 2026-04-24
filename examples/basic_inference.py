"""Basic single-GPU inference example.

Usage:
    python examples/basic_inference.py --model /path/to/Qwen3-0.6B
    python examples/basic_inference.py --model /path/to/model --prompt "你好"

Chat-tuned models (DeepSeek-R1-Distill-*, Qwen-*-Chat/-Instruct, etc.) require
their prompts to be wrapped in a model-specific chat template (e.g. with
`<｜User｜>` / `<｜Assistant｜>` or `<|im_start|>user` markers). Without the
template, Chinese prompts in particular tend to collapse to degenerate output
(see `docs/DEV_LOG.md` finding #8). We let `AutoTokenizer.apply_chat_template`
inject those markers automatically.
"""

import argparse

from transformers import AutoTokenizer

from nano_dist_spec import LLM, SamplingParams


def build_prompt(tokenizer: AutoTokenizer, user_text: str) -> str:
    """Wrap raw user text in the model's chat template when available.

    DeepSeek-R1 series additionally requires `<think>\\n` to be prefilled
    after `<｜Assistant｜>` (see the model card on HuggingFace) so the
    model reliably enters its reasoning chain. The stock chat template
    stops at `<｜Assistant｜>`, so we patch it here when detected.
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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--prompt", type=str, default="用一句话介绍你自己",
        help="Raw user question; chat template will be applied automatically.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = build_prompt(tokenizer, args.prompt)

    llm = LLM(args.model, dtype="bfloat16")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate([prompt], sampling_params)

    for output in outputs:
        print(f"Prompt:  {output.prompt}")
        print(f"Output:  {output.text}")
        print("-" * 60)


if __name__ == "__main__":
    main()
