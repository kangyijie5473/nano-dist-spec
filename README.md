# nano-dist-spec

**Minimal distributed inference + speculative decoding framework (~2000 lines of Python)**

A from-scratch implementation designed for learning and demonstrating deep understanding of LLM inference infrastructure. Covers the two most interview-relevant topics in modern LLM serving: **Tensor Parallelism** and **Speculative Decoding**.

## Features

| Feature | Description |
|---------|-------------|
| **Tensor Parallelism** | Hand-written `ColumnParallelLinear` / `RowParallelLinear` with NCCL AllReduce |
| **Speculative Decoding** | Draft-then-verify with mathematically exact rejection sampling |
| **Paged KV Cache** | Block-based allocation eliminating memory fragmentation (PagedAttention) |
| **Continuous Batching** | Dynamic request scheduling with prefill/decode separation |
| **HuggingFace Compatible** | Load any Llama/Qwen model directly from safetensors checkpoints |
| **GQA Support** | Grouped Query Attention for efficient KV heads |

## Architecture

```
                    ┌─────────────────────────┐
                    │       LLM (API)         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │      LLMEngine          │
                    │  ┌───────┐ ┌─────────┐  │
                    │  │Scheduler│ │KVCacheMgr│ │
                    │  └───┬───┘ └────┬────┘  │
                    └──────┼──────────┼───────┘
                           │          │
              ┌────────────▼──────────▼────────────┐
              │        TransformerModel             │
              │  ┌──────────────────────────────┐   │
              │  │ VocabParallelEmbedding        │   │
              │  │ N × TransformerBlock          │   │
              │  │   ├─ RMSNorm                  │   │
              │  │   ├─ Attention (ColumnParallel)│  │
              │  │   │   Q,K,V → ColumnParallel  │   │
              │  │   │   O     → RowParallel     │   │
              │  │   ├─ RMSNorm                  │   │
              │  │   └─ MLP (SwiGLU)             │   │
              │  │       gate,up → ColumnParallel │   │
              │  │       down   → RowParallel    │   │
              │  │ RMSNorm                       │   │
              │  │ LM Head (ColumnParallel)      │   │
              │  └──────────────────────────────┘   │
              └─────────────────────────────────────┘
```

## Quick Start

### Single GPU Inference

```python
from nano_dist_spec import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-0.6B")
outputs = llm.generate(
    ["Explain distributed inference:"],
    SamplingParams(temperature=0.7, max_tokens=128),
)
print(outputs[0].text)
```

### Tensor Parallel (Multi-GPU)

```bash
torchrun --nproc_per_node=2 examples/tensor_parallel.py --model /path/to/model
```

### Speculative Decoding

```python
from nano_dist_spec import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-1.7B",         # target (large)
    draft_model_path="/path/to/Qwen3-0.6B",  # draft (small)
    num_speculative_tokens=5,
)
outputs = llm.generate(
    ["What is speculative decoding?"],
    SamplingParams(temperature=0.7, max_tokens=256),
)
```

## Core Concepts Explained

### Tensor Parallelism (TP)

Split model weights across GPUs so each GPU handles a subset of attention heads and MLP neurons. Communication happens via **AllReduce** at two points per layer:

```
GPU 0:  x → [Q₀,K₀,V₀] → Attn₀ → O₀ ─┐
                                          ├─ AllReduce → residual
GPU 1:  x → [Q₁,K₁,V₁] → Attn₁ → O₁ ─┘

GPU 0:  h → gate₀,up₀ → SiLU·mul → down₀ ─┐
                                              ├─ AllReduce → residual
GPU 1:  h → gate₁,up₁ → SiLU·mul → down₁ ─┘
```

**Why this split?** Q/K/V projections use `ColumnParallel` (split output dim — each GPU gets different heads). O/down projections use `RowParallel` (split input dim — each GPU holds partial results, AllReduce sums them). This minimizes communication to just 2 AllReduce ops per transformer block.

### Speculative Decoding

Accelerate inference by having a small **draft** model guess K future tokens, then verifying all K at once with the large **target** model:

```
Step 1 (Draft):    Small model generates K=5 tokens autoregressively
                   t₁ → t₂ → t₃ → t₄ → t₅   (fast, 5 sequential steps)

Step 2 (Verify):   Large model scores all 5 in ONE forward pass
                   [t₁, t₂, t₃, t₄, t₅] → [p₁, p₂, p₃, p₄, p₅, p₆]

Step 3 (Accept):   Rejection sampling: accept t₁✓ t₂✓ t₃✗ → resample t₃'
                   Result: 3 tokens from ~1 large-model forward pass
```

**Rejection sampling** guarantees the output distribution is identical to the target model — no approximation. Accept token `x` with probability `min(1, p_target(x) / q_draft(x))`. If rejected, sample from `max(0, p_target - q_draft)` (the residual distribution).

### Paged KV Cache

Instead of allocating contiguous memory per sequence (fragmentation!), use fixed-size **blocks** (e.g., 16 tokens). A **block table** maps logical positions to physical blocks:

```
Sequence A: [Block 3] [Block 7] [Block 1]  ← 48 tokens in 3 blocks
Sequence B: [Block 0] [Block 5]            ← 25 tokens in 2 blocks

Physical:   [B:0-15] [A:32-47] [free] [A:0-15] [free] [B:16-25] [free] [A:16-31]
Block ID:      0         1       2        3       4       5        6        7
```

Benefits: no fragmentation, efficient memory utilization, easy rollback for speculative decoding (just free tail blocks).

## Project Structure

```
nano_dist_spec/
├── config.py          # Model/cache/scheduler configuration       (~60 lines)
├── parallel.py        # TP primitives: Column/Row/VocabParallel  (~150 lines)
├── attention.py       # RoPE, prefill/decode/extend attention    (~200 lines)
├── model.py           # Transformer + HuggingFace weight loading (~250 lines)
├── kv_cache.py        # Paged KV cache + block allocator         (~170 lines)
├── sampling.py        # Temperature, top-k, top-p sampling       (~80 lines)
├── scheduler.py       # Continuous batching scheduler            (~120 lines)
├── speculative.py     # Speculative decoding + rejection sampling(~250 lines)
├── engine.py          # Inference engine + LLM API               (~250 lines)
└── worker.py          # Distributed worker for torchrun          (~80 lines)
```

## Interview-Relevant Design Decisions

1. **Why ColumnParallel for Q/K/V but RowParallel for O?**
   Each GPU needs complete heads for attention computation. ColumnParallel splits heads across GPUs. The O projection recombines, requiring AllReduce.

2. **Why does speculative decoding use rejection sampling instead of just argmax?**
   Rejection sampling preserves the exact target distribution for any temperature, not just greedy. The residual distribution `max(0, p-q)/Z` corrects for draft model errors.

3. **Why paged KV cache instead of contiguous allocation?**
   Contiguous allocation wastes memory (must pre-allocate for max sequence length). Paged allocation grows dynamically, shares blocks across sequences, and supports efficient rollback.

4. **How many AllReduce ops per transformer block in TP?**
   Exactly 2: one in the attention O projection (RowParallel) and one in the MLP down projection (RowParallel). This is the theoretical minimum.

5. **What determines speculative decoding speedup?**
   Speedup ≈ `E[accepted + 1] / (K × cost_draft + cost_target)`. Higher acceptance rate (better draft-target alignment) and lower `cost_draft / cost_target` ratio give better speedup.

## Running Tests

```bash
python -m pytest tests/ -v

# Or run individual test files
python tests/test_parallel.py
python tests/test_kv_cache.py
python tests/test_speculative.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Hugging Face Transformers
- safetensors
- NVIDIA GPU with CUDA (for inference; tests run on CPU)

## License

MIT
