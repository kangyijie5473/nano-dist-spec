"""Compare nano-dist-spec model against HuggingFace reference implementation.

Usage:
    pytest tests/test_model_vs_hf.py -v --model-path ~/models/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/
    
    Or run directly:
    python tests/test_model_vs_hf.py --model-path ~/models/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/
"""

import argparse
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nano_dist_spec.config import ModelConfig
from nano_dist_spec.model import TransformerModel, _load_safetensors
from nano_dist_spec.attention import InputMetadata
from nano_dist_spec.kv_cache import BlockAllocator, KVCache, KVCacheManager
from nano_dist_spec.sampling import SamplingParams, sample

MODEL_PATH = None  # set by conftest / CLI


def get_model_path():
    if MODEL_PATH is not None:
        return MODEL_PATH
    default = Path.home() / "models/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if default.exists():
        return str(default)
    pytest.skip("Model path not available")


# ---------------------------------------------------------------
# Test 1: Weight completeness
# ---------------------------------------------------------------

def test_weight_completeness():
    """Every weight key in safetensors should be loaded by our model."""
    model_path = get_model_path()
    raw_weights = _load_safetensors(model_path)
    raw_keys = set(raw_weights.keys())

    config = ModelConfig.from_pretrained(model_path)
    model = TransformerModel(config, tp_size=1)

    our_keys_map = {}
    for name, _ in model.named_parameters():
        our_keys_map[name] = True

    expected_hf_keys = set()
    expected_hf_keys.add("model.embed_tokens.weight")
    expected_hf_keys.add("model.norm.weight")
    expected_hf_keys.add("lm_head.weight")

    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            expected_hf_keys.add(f"{p}.self_attn.{proj}.weight")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expected_hf_keys.add(f"{p}.mlp.{proj}.weight")
        expected_hf_keys.add(f"{p}.input_layernorm.weight")
        expected_hf_keys.add(f"{p}.post_attention_layernorm.weight")

    missing = expected_hf_keys - raw_keys
    assert not missing, f"Expected keys missing from safetensors: {missing}"

    extra_in_safetensors = raw_keys - expected_hf_keys
    print(f"\nExtra keys in safetensors (not in minimal expected set): {extra_in_safetensors}")
    if extra_in_safetensors:
        print("  → These are likely bias weights that should also be loaded!")

    del raw_weights


# ---------------------------------------------------------------
# Test 2: Prefill logits comparison
# ---------------------------------------------------------------

@torch.no_grad()
def test_prefill_logits():
    """Compare prefill logits between our model and HuggingFace reference."""
    model_path = get_model_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_text = "The capital of France is"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]

    # --- HuggingFace reference ---
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device,
    )
    hf_model.eval()
    hf_out = hf_model(input_ids)
    hf_logits = hf_out.logits  # [1, seq_len, vocab]

    # --- Our model ---
    config = ModelConfig.from_pretrained(model_path)
    our_model = TransformerModel(config, tp_size=1)
    our_model.to(device=device, dtype=dtype)
    our_model.load_weights(model_path)
    our_model.eval()

    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    slot_mapping = torch.arange(seq_len, device=device)
    metadata = InputMetadata(slot_mapping=slot_mapping, block_size=16)

    block_size = 16
    num_kv_heads = config.num_key_value_heads
    num_blocks = (seq_len + block_size - 1) // block_size + 1
    kv_cache = KVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=config.head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )
    kv_list = [kv_cache.get_kv(i) for i in range(kv_cache.num_layers)]

    our_logits = our_model(input_ids, positions, kv_list, metadata)

    # --- Compare ---
    hf_last = hf_logits[0, -1, :].float()
    our_last = our_logits[0, -1, :].float()

    hf_top5 = hf_last.topk(5)
    our_top5 = our_last.topk(5)

    print(f"\nHF  top-5 tokens: {hf_top5.indices.tolist()} logits: {hf_top5.values.tolist()}")
    print(f"Our top-5 tokens: {our_top5.indices.tolist()} logits: {our_top5.values.tolist()}")

    hf_pred = hf_last.argmax().item()
    our_pred = our_last.argmax().item()
    print(f"HF  greedy next token: {hf_pred} = '{tokenizer.decode([hf_pred])}'")
    print(f"Our greedy next token: {our_pred} = '{tokenizer.decode([our_pred])}'")

    cos_sim = torch.nn.functional.cosine_similarity(hf_last.unsqueeze(0), our_last.unsqueeze(0))
    print(f"Cosine similarity of last-position logits: {cos_sim.item():.6f}")

    assert cos_sim.item() > 0.99, f"Logits too different: cosine_sim={cos_sim.item():.4f}"
    assert hf_pred == our_pred, f"Greedy prediction mismatch: HF={hf_pred}, ours={our_pred}"

    del hf_model, our_model, kv_cache


# ---------------------------------------------------------------
# Test 3: Greedy decode comparison
# ---------------------------------------------------------------

@torch.no_grad()
def test_greedy_decode():
    """Compare greedy decode output (20 tokens) between our model and HF."""
    model_path = get_model_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = "1 + 1 = 2, 2 + 2 ="
    input_ids_list = tokenizer.encode(prompt, add_special_tokens=True)
    num_gen = 20

    # --- HuggingFace greedy ---
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device,
    )
    hf_model.eval()
    hf_input = torch.tensor([input_ids_list], device=device)
    hf_generated = hf_model.generate(
        hf_input, max_new_tokens=num_gen, do_sample=False,
        temperature=None, top_p=None,
    )
    hf_new_tokens = hf_generated[0, len(input_ids_list):].tolist()

    # --- Our greedy decode ---
    config = ModelConfig.from_pretrained(model_path)
    our_model = TransformerModel(config, tp_size=1)
    our_model.to(device=device, dtype=dtype)
    our_model.load_weights(model_path)
    our_model.eval()

    block_size = 16
    num_kv_heads = config.num_key_value_heads
    max_len = len(input_ids_list) + num_gen
    num_blocks = (max_len + block_size - 1) // block_size + 2
    kv_cache = KVCache(
        num_layers=config.num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=config.head_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )
    allocator = BlockAllocator(num_blocks)
    kv_mgr = KVCacheManager(block_size, allocator)

    seq_id = 0
    kv_mgr.allocate_seq(seq_id, len(input_ids_list))
    prompt_len = len(input_ids_list)

    # Prefill
    inp = torch.tensor([input_ids_list], device=device)
    positions = torch.arange(prompt_len, device=device).unsqueeze(0)
    slot_mapping = kv_mgr.compute_slot_mapping(seq_id, 0, prompt_len, device)
    metadata = InputMetadata(slot_mapping=slot_mapping, block_size=block_size)
    kv_list = [kv_cache.get_kv(i) for i in range(kv_cache.num_layers)]
    logits = our_model(inp, positions, kv_list, metadata)
    next_token = logits[0, -1, :].argmax().item()

    our_new_tokens = [next_token]

    # Decode loop
    for step in range(num_gen - 1):
        kv_mgr.append_slots(seq_id, 1)
        pos = kv_mgr.context_lens[seq_id] - 1
        inp = torch.tensor([[next_token]], device=device)
        positions = torch.tensor([[pos]], device=device)
        slot_mapping = kv_mgr.compute_slot_mapping(seq_id, pos, 1, device)
        block_tables = kv_mgr.get_block_table_tensor([seq_id], device)
        context_lens = kv_mgr.get_context_lens_tensor([seq_id], device)
        metadata = InputMetadata(
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            context_lens=context_lens,
            block_size=block_size,
        )
        kv_list = [kv_cache.get_kv(i) for i in range(kv_cache.num_layers)]
        logits = our_model(inp, positions, kv_list, metadata)
        next_token = logits[0, -1, :].argmax().item()
        our_new_tokens.append(next_token)

    hf_text = tokenizer.decode(hf_new_tokens, skip_special_tokens=True)
    our_text = tokenizer.decode(our_new_tokens, skip_special_tokens=True)

    print(f"\nPrompt: {prompt}")
    print(f"HF  output: {hf_text!r}")
    print(f"Our output: {our_text!r}")
    print(f"HF  tokens: {hf_new_tokens}")
    print(f"Our tokens: {our_new_tokens}")

    match_count = sum(a == b for a, b in zip(hf_new_tokens, our_new_tokens))
    print(f"Token match: {match_count}/{min(len(hf_new_tokens), len(our_new_tokens))}")

    assert match_count >= len(our_new_tokens) * 0.8, (
        f"Too many mismatches: {match_count}/{len(our_new_tokens)}"
    )

    del hf_model, our_model, kv_cache


# ---------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    args = parser.parse_args()
    MODEL_PATH = args.model_path

    print("=" * 60)
    print("Test 1: Weight completeness")
    print("=" * 60)
    test_weight_completeness()
    print("PASSED")

    print("\n" + "=" * 60)
    print("Test 2: Prefill logits comparison")
    print("=" * 60)
    test_prefill_logits()
    print("PASSED")

    print("\n" + "=" * 60)
    print("Test 3: Greedy decode comparison")
    print("=" * 60)
    test_greedy_decode()
    print("PASSED")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
