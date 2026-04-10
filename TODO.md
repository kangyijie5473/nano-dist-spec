# nano-dist-spec TODO

## Done

- [x] **config.py** — ModelConfig (from HuggingFace config.json), CacheConfig, SchedulerConfig, SpeculativeConfig
- [x] **parallel.py** — ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding, AllReduce/AllGather
- [x] **attention.py** — RoPE precomputation, prefill_attention (causal sdpa), decode_paged_attention (block-table gather), extend_attention (verify path)
- [x] **model.py** — RMSNorm, Attention (GQA + TP), MLP (SwiGLU + TP), TransformerBlock, TransformerModel, safetensors weight loading with TP split
- [x] **kv_cache.py** — BlockAllocator (free-list), KVCache (pre-allocated GPU tensors), KVCacheManager (block tables, slot mapping, append, rollback)
- [x] **sampling.py** — temperature scaling, top-k filter, top-p (nucleus) filter, logits_to_probs
- [x] **scheduler.py** — Sequence lifecycle (WAITING/RUNNING/FINISHED), continuous batching, memory-aware admission
- [x] **speculative.py** — SpeculativeDecoder (draft→verify→reject loop), rejection_sample (greedy + stochastic), KV rollback & resync
- [x] **engine.py** — LLMEngine (prefill + batched decode loop), LLM user API, speculative generation integration
- [x] **worker.py** — Distributed worker with `torchrun`, NCCL init, rank-aware model sharding
- [x] **examples/** — basic_inference, tensor_parallel, speculative_decode
- [x] **tests/** — test_parallel (5 tests), test_kv_cache (8 tests), test_speculative (7 tests) — all 20 passing
- [x] **README.md** — Architecture diagram, quick start, core concepts explanation, interview Q&A

## Future Improvements

- [ ] **CUDA Graph** — Capture decode step as CUDA graph for reduced kernel launch overhead
- [ ] **torch.compile** — Apply `torch.compile` to model forward for operator fusion
- [ ] **Prefix Caching** — Hash-based KV block deduplication for shared prompt prefixes
- [ ] **INT8 KV Quantization** — Quantize KV cache to int8 for ~50% memory reduction
- [ ] **Chunked Prefill** — Mix prefill and decode tokens in a single batch for better GPU utilization
- [ ] **Streaming Output** — Yield tokens as they are generated (async generator API)
- [ ] **Pipeline Parallelism** — Split model layers across GPUs (complement to tensor parallelism)
- [ ] **Multi-node Support** — Extend worker.py to support multi-node via `torchrun --nnodes`
- [ ] **FlashAttention Integration** — Optional FlashAttention backend for prefill attention
- [ ] **Benchmark Script** — `bench.py` with tokens/s, TTFT, acceptance rate, memory usage metrics
- [ ] **Medusa/EAGLE-style Draft** — Single-model speculative decoding with extra prediction heads
- [ ] **Request Preemption** — Swap out low-priority sequences when KV memory is exhausted
