# 开发日志

## Bug #1: RMSNorm dtype 提升导致 dtype 不匹配

**现象**: `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half`

**根因**: `RMSNorm.forward` 中 `x.float()` 将 variance 提升为 float32，随后 `x = x * torch.rsqrt(variance + self.eps)` 隐式将 `x` 也提升为 float32。最后 `.to(x.dtype)` 引用的是已经被提升后的 dtype（float32），等于没转换，导致后续 Linear 层收到 float32 输入但权重是 float16。

**修复**: 在计算前保存 `orig_dtype = x.dtype`，返回时 `.to(orig_dtype)`。

```python
# model.py RMSNorm.forward
def forward(self, x):
    orig_dtype = x.dtype  # 保存原始精度
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.eps)
    return (self.weight * x).to(orig_dtype)  # 转回原始精度
```

---

## Bug #2: Qwen2 attention bias 未加载导致输出乱码

**现象**: 模型推理输出全是 `0` 和乱码字符（如 `额`、`�`），完全不可读。

**根因**: DeepSeek-R1-Distill-Qwen-1.5B 基于 Qwen2ForCausalLM 架构。Qwen2 的 Q/K/V 投影默认 `attention_bias=True`（HuggingFace 默认值，config.json 中未显式写出）。但我们代码中 `ColumnParallelLinear` 全部使用 `bias=False`，导致 safetensors 中的 84 个 bias 权重被静默跳过，模型计算完全错误。

**修复**:

1. `config.py` 新增 `attention_bias` 字段：

```python
@dataclass
class ModelConfig:
    ...
    attention_bias: bool = False

@classmethod
def from_pretrained(cls, model_path):
    ...
    attention_bias=raw.get("attention_bias", True),  # Qwen2 HF 默认 True
```

2. `model.py` Attention 类传入 bias 参数：

```python
self.q_proj = ColumnParallelLinear(..., bias=config.attention_bias, ...)
self.k_proj = ColumnParallelLinear(..., bias=config.attention_bias, ...)
self.v_proj = ColumnParallelLinear(..., bias=config.attention_bias, ...)
self.o_proj = RowParallelLinear(..., bias=False, ...)  # Qwen2 O 投影无 bias
```

3. `model.py` load_weights 增加 bias 加载：

```python
bias_key = f"{p}.self_attn.{name}.bias"
if bias_key in weights:
    b = weights[bias_key]
    if tp_size > 1:
        b = tensor_split(b, tp_rank, tp_size, dim=0)
    getattr(layer.self_attn, name).linear.bias.data.copy_(b)
```

**验证**: 编写 `tests/test_model_vs_hf.py`，对照 HuggingFace 参考实现：
- Prefill logits cosine similarity = 0.999895
- Greedy decode 20 tokens 完全匹配（20/20）

---

## Bug #3: float16 精度溢出导致 NaN 崩溃

**现象**: `Assertion 'probability tensor contains either inf, nan or element < 0' failed`，发生在 `torch.multinomial` 调用处。

**根因**: 模型原生 dtype 为 `bfloat16`（config.json 中 `"torch_dtype": "bfloat16"`），但 `basic_inference.py` 使用 `dtype="float16"`。bfloat16 和 float32 共享相同的指数位（8 位，最大值 ~3.4e38），而 float16 只有 5 位指数（最大值 ~65504）。attention score（QK^T）在 float16 下容易溢出为 inf，softmax 后变成 NaN，最终 multinomial 崩溃。

**修复**:

1. `examples/basic_inference.py` 默认 dtype 改为 `bfloat16`：

```python
llm = LLM(args.model, dtype="bfloat16")
```

2. `sampling.py` 增加防护——采样前转 float32 并 clamp：

```python
logits = logits.float() / params.temperature  # float32 避免溢出
...
probs = F.softmax(logits, dim=-1)
probs = probs.clamp(min=0.0)
probs = probs / probs.sum(dim=-1, keepdim=True)  # renorm 防 NaN
```

---

## Bug #4: 投机解码初始化 KV cache 抢占全部显存导致 draft 模型 OOM

**现象**: 在 24GB RTX 4090 上同时加载 7B target + 1.5B draft 模型时，target 加载完成后开始分配 KV cache，直接报 `CUDA out of memory`，错误发生在分配 target KV cache 的阶段，draft 模型的权重还没来得及加载。

**根因**: `LLMEngine.__init__` 里的 KV cache 初始化流程是：

```python
# engine.py 简化示意
self.target_model = load_target(...)         # 7B 权重占显存
num_blocks = self._estimate_num_blocks(...)  # 按「当前剩余显存 * util」算
self.target_kv = KVCache(num_blocks, ...)    # 吃掉几乎所有剩余显存
self.draft_model = load_draft(...)           # 此时没显存了 → OOM
```

`_estimate_num_blocks` 用的是**调用时的剩余显存**作为基数，它不知道后面还要加载 1.5B 的 draft 权重 + draft 自己的 KV cache。单模型推理时这个逻辑没问题，投机解码多了 draft 模型就会打架。

**修复**: 在 `examples/speculative_decode.py` 里显式指定 `num_gpu_blocks=4000` 做workaround，绕过自动估算：

```python
llm = LLM(
    args.target,
    dtype="bfloat16",
    num_gpu_blocks=4000,   # 手动限制，给 draft 留出显存
    draft_model_path=args.draft,
    num_speculative_tokens=args.K,
)
```

**更根本的修复方向**（尚未实施）: `LLMEngine` 应该在 draft 模型和其 KV cache 都加载完成**之后**再给 target KV cache 估算 block 数，或者引入 `gpu_memory_utilization` 参数让两个模型的 KV 共同分配一个预算。

---

## Bug #5: Target / Draft 模型 padded vocab size 不一致导致形状不匹配

**现象**:

```
RuntimeError: The size of tensor a (152064) must match the size of tensor b (151936)
  at non-singleton dimension 2
```

发生在 `speculative.py` 的 `rejection_sample` 里 `target_probs - draft_probs_full` 这一行。

**根因**: Target (DeepSeek-R1-Distill-Qwen-7B) 和 Draft (DeepSeek-R1-Distill-Qwen-1.5B) 虽然共用同一套 tokenizer（实际词表 ~151665），但它们的 `config.json` 里 `vocab_size` 字段做了不同倍数的 padding：

| 模型 | 实际 tokens | padded vocab_size |
|------|-------------|-------------------|
| 7B   | ~151665     | **152064**（pad 到 256 倍数） |
| 1.5B | ~151665     | **151936**（pad 到 128 倍数） |

两边 LM head 输出 logits 的最后一维就此不一致。rejection sampling 要逐元素比较 `p_target(x)` 和 `q_draft(x)`，形状必须对齐。

**修复**: 把两者共同的、实际会被 tokenizer 产出的前缀部分作为「共享词表」，超出的 padding 槽位直接丢弃——它们本就不会被采样到。

```python
# speculative.py SpeculativeDecoder.__init__
self.shared_vocab_size = min(
    target_model.config.vocab_size,
    draft_model.config.vocab_size,
)
```

然后**所有**进入概率空间的 logits 都要先截断到这个维度：prefill 的 `target_last`、draft 循环的 `draft_last`、verify 阶段的 `target_logits[:, i, :]`、以及 round 结束后保存的 `new_saved`：

```python
target_last = target_logits[:, -1, : self.shared_vocab_size]
first_token = sample(target_last, params).item()
saved_probs = logits_to_probs(target_last, params.temperature)
```

**验证**: RuntimeError 消失，投机解码可以跑完整个 round，但输出还有问题（见 Bug #6、#7）。

---

## Bug #6: `extend_attention` 使用 `is_causal=True` 对矩形 Q/K 应用了错误的顶端对齐掩码

**现象**: 修完 Bug #5 之后，投机解码对英文 prompt `"Introduce you by 10 words"` 的输出是：

```
Introduce you by 10 words
 in.

1.,1: 1., , 2: , : : respond
2, , : ,:副校长：,:张老师<think><think>
)。嗯，,好的,我,我,是,ioneer,, 介绍,介绍,)。,
:,
```

**基础推理同一个 1.5B 模型单独跑完全正常**，所以不是模型问题，是投机解码特有的。

**根因**: `extend_attention` 是投机解码 verify 阶段专用的 attention 路径——Q 是 K 个 draft token（shape `[1, h, K, d]`），KV 是「cached prefix（P 个）+ 当前 K 个」一共 `P+K` 个。原来的实现直接用了：

```python
return F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)
```

**PyTorch 的 `is_causal=True` 对矩形 Q/K 默认应用「顶端对齐」（top-left aligned）的 tril 掩码**。即当 `Q_len=K`、`KV_len=P+K` 时，生成的掩码形状是：

```
     k0  k1  k2  ... kP-1  kP  kP+1  ...
q0:   1   0   0       0     0    0
q1:   1   1   0       0     0    0
q2:   1   1   1       0     0    0
...
```

也就是说 query `i`（代表第 `P+i` 个位置上的 draft token）**只能看到 KV 的前 `i+1` 个**——那是 prompt 最前面几个 token，**根本不是它自己所在位置的 prefix**。整个 target 模型是在用被截断得不成样子的上下文给 draft token 打分。

正确的"右下对齐"（bottom-right aligned）掩码应该是：

```
     k0  k1  k2  ... kP-1  kP  kP+1  kP+2
q0:   1   1   1       1     1    0     0
q1:   1   1   1       1     1    1     0
q2:   1   1   1       1     1    1     1
```

**修复**: 手动构造掩码：

```python
kv_len = k_full.shape[2]  # == prefix_len + K
q_idx = torch.arange(K, device=device).unsqueeze(1)       # [K, 1]
k_idx = torch.arange(kv_len, device=device).unsqueeze(0)  # [1, kv_len]
attn_mask = k_idx <= (prefix_len + q_idx)                 # True = attend

return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=attn_mask)
```

**验证**: 新增 `tests/test_attention.py` 防回归：

- `test_extend_attention_matches_reference_with_prefix`: 对照手写参考实现逐元素 `allclose`
- `test_extend_attention_zero_prefix_is_causal`: `prefix_len=0` 时必须和 `prefill_attention` 完全一致

全部 25 个测试通过。

---

## Bug #7: 投机解码 bonus token 采样时把 probs 当 logits 做了二次 softmax

**现象**: Bug #6 修完后，投机解码输出大部分是连贯的，但**偶尔**会冒出完全无关的 token，跨语言、跨字符集：

```
好嗯，用户让我用一句话 appending �介绍自己。首先，我<Vertex>是一个AI助手，
由深度求索开发... ᵈ可能是在寻找... الحرب的回应... 一目了然地了解我_MT。
```

`appending`、`<Vertex>`、`ᵈ`、`الحرب`（阿拉伯语）、`_MT`、`дополнительный`（俄语）—— 典型的**从均匀分布里瞎采**的特征。

**根因**: 当 K 个 draft token **全部被接受**时，要再额外采一个 "bonus token"：

```python
# 错误代码
if all_accepted:
    bonus = sample(target_probs_verify[K].unsqueeze(0), params).item()
```

这里 `target_probs_verify[K]` 已经是 `logits_to_probs(target_logits[..., :shared_vocab_size], T)` 的结果——**一个合法的概率分布**。而 `sample()` 的约定输入是 **raw logits**，它内部会再 softmax 一次：

```python
# sampling.py sample()
logits = logits.float() / params.temperature
...
probs = F.softmax(logits, dim=-1)
```

对一个概率向量（所有元素在 `[0, 1]` 且极小，约 `1/152064`）做 `softmax(p / 0.7)`：因为 p 本身各元素之间的数值差异也很小（分布的峰值可能是 0.3，其他几万个 token 都是接近 0 的小数），除以 0.7 后差异仍小，softmax 出来**几乎是词表上的均匀分布**。于是 bonus 实际上是从全词表里随机抽一个 token。

为什么这个 bug 只表现为"偶尔乱码"而非"全崩"？数学估算：

- 典型接受率约 60%，K=5 时全接受的概率 ≈ `0.6⁵ ≈ 7.8%`
- 每 round 平均产出 3–4 个 accepted + 仅当全接受时 1 个随机 bonus
- 所以 bonus 在总输出里占比约 **2–5%**，和实测吻合

更糟糕的是随机 bonus 会写进 **target KV cache** 并成为下一轮的 prefix，污染后续所有生成。

**修复**: bonus 应该直接从已有的 probs 多项式采样，别再过一次 sampler 的 softmax：

```python
# speculative.py
if all_accepted:
    bonus_probs = target_probs_verify[K]
    if params.temperature == 0:
        bonus = int(bonus_probs.argmax(dim=-1).item())
    else:
        bonus = int(torch.multinomial(bonus_probs, num_samples=1).item())
    accepted.append(bonus)
```

**验证**: 所有 25 个测试继续通过；投机解码中文输出不再出现跨语言垃圾 token。

---

## 发现 #8: DeepSeek-R1-Distill-Qwen 系列必须应用 chat template 才能正常生成

**现象**: 把 `basic_inference.py` 的 prompt 换成中文 `"用一句话介绍你自己"` 后，输出退化成无限重复 `**` 或半截乱码；换成英文 prompt 又正常。

**根因**: 不是 bug，是使用姿势问题。DeepSeek-R1-Distill-Qwen 是**对话精调模型**（chat-tuned），需要特定的对话模板：

```
<｜begin▁of▁sentence｜><｜User｜>用一句话介绍你自己<｜Assistant｜><think>\n
```

原始 `engine.py` 里只调用了 `tokenizer.encode(p, add_special_tokens=True)`——只会加 BOS，**不会套模板**。没有模板时，中文 prompt 对这类模型来说是"裸喂"，模型没有 user/assistant 分隔符，就会进入退化模式。英文因为在 base pre-train 里占比大，即使没有模板也能勉强续写，所以更难暴露这个问题。

**当前处理方式**（临时）: 在 example 脚本里手动把模板拼到 prompt 字符串里：

```python
prompts = ["<｜begin▁of▁sentence｜><｜User｜>用一句话介绍你自己<｜Assistant｜><think>\n"]
```

**更系统的方案**（TODO）: 在 `LLM.generate` 里增加参数支持 `apply_chat_template=True`，自动调用 `tokenizer.apply_chat_template(...)`。

---

## 进展 #9: 基准测试 — Baseline & 投机解码 K-sweep 实测数据

**目标**: 编写 `bench.py` 并跑出简历需要的数据点（tokens/s、TTFT、acceptance rate、speedup）。

### 测试环境

- 硬件: RTX 4090 24GB，单卡
- 模型: DeepSeek-R1-Distill-Qwen-1.5B / 7B（bf16 原生 dtype）
- 测量方法: bypass `LLM.generate()`，直接驱动 `LLMEngine._prefill_seq` / `_decode_batch`，把 prefill+第一次 sample（=TTFT）和稳态 decode 拆开计时；warmup 1 次 + 测量 N 次取均值 / std；每次 run 前 `torch.cuda.empty_cache() + reset_peak_memory_stats()`
- 输出: `bench_results/<mode>_<timestamp>.json`

### bench.py 四个 mode

| mode | 用途 | 关键指标 |
|------|------|----------|
| `basic` | 单 prompt baseline | TTFT、decode_tps、peak_mem |
| `spec` | target+draft 在 K × temperature 矩阵上扫描 | acceptance_rate、tokens/round、speedup_vs_baseline |
| `batch` | 连续批处理吞吐曲线 | aggregate_tps |
| `kv-utilization` | paged 实际占用 vs naive 连续分配 | memory_savings_pct |

实现要点：spec 模式下整个 sweep 只加载一次 target+draft（K 是 `_spec_decoder.K` 的运行时属性，温度只影响采样），把 16 个 (K, T) 配置的总耗时从 ~30 分钟压到 ~15 分钟。

### 单模型 Baseline（prompt_len=128, max_tokens=256, runs=3）

| 模型 | TTFT | decode tps | peak mem |
|------|------|------------|----------|
| 1.5B | 29.3 ± 4.0 ms | **28.0 ± 0.3 tok/s** | 4.36 GB |
| 7B   | 29.1 ± 1.7 ms | **26.2 ± 0.4 tok/s** | 15.71 GB |

### 投机解码 K-sweep（target=7B, draft=1.5B, prompt_len=128, max_tokens=128, runs=2）

Target-only baseline: **26.6 tok/s**，TTFT 29.6 ms

| K | T=0 decode_tps | T=0 accept_rate | T=0 tok/round | T=0 speedup | T=0.7 decode_tps | T=0.7 accept_rate |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 10.04 | 85.5% | 1.86 | 0.38× | 8.84 | 44.9% |
| 2 | 10.64 | 81.6% | 2.63 | 0.40× | 9.02 | 52.2% |
| 3 | 10.94 | 78.9% | 3.37 | **0.41×** | 7.64 | 35.8% |
| 4 | 10.62 | 71.2% | 3.85 | 0.40× | 7.05 | 28.8% |
| 5 | 10.40 | 69.0% | 4.45 | 0.39× | 7.17 | 33.4% |
| 6 |  9.98 | 61.7% | 4.70 | 0.38× | 5.88 | 21.2% |
| 7 | 10.45 | 65.2% | 5.57 | 0.39× | 8.40 | 47.8% |
| 8 |  8.93 | 57.8% | 5.62 | 0.34× | 5.17 | 18.0% |

### 反直觉发现：spec speedup < 1

按 plan 预期 spec 加速比应在 1.5–2.5×，但实测**全部 K × T 组合都比 target-only baseline 慢**（最好的 K=3/T=0 也只有 0.41×）。
**接受率本身完全正常**：T=0、K=1 时 86%，符合 Leviathan 论文 0.6–0.9 的范围；T 升高 → 接受率显著下降；K 增大 → 接受率单调下降。这说明算法实现是正确的（rejection sampling 数学等价性已经在 `tests/test_speculative.py` 验证，K 扫描曲线进一步背书）。

**根因是 per-step Python 编排开销**。观察 baseline 端：

- 1.5B 和 7B 的 decode_tps 几乎相等（28 vs 26 tok/s），说明 **decode 时 GPU 没饱和**——Python 端的 per-step 开销已经压过 4.6× 模型规模差。
- 每个 decode step 都要执行：
  1. `compute_slot_mapping` 在 Python `for pos in range(...)` 里逐个算 slot → 构造 list → `torch.tensor(slots)` 一次 H2D
  2. `get_block_table_tensor` / `get_context_lens_tensor` 各一次 H2D
  3. `torch.tensor([tokens], device='cuda')` 输入 H2D
  4. forward → `sample(...)` → `.item()`（强制 GPU→CPU 同步）
- 每步净开销 ~30 ms，里面 forward 实际耗时只占小部分。

**精测修正（见进展 #10）**：`torch.profiler` 按 `_decode_batch` 分段后，`compute_slot_mapping` + `build_input_tensors` 合计约 **0.4%/step**；瓶颈在 **`model_forward` 内大量 `cudaLaunchKernel`（CPU launch-bound）**，而非 slot 列表推导本身。

**spec 把这个开销 ×（K + verify + resync + final）放大了**：每 round 要做 K（draft）+ 1（verify，唯一被 batched 的）+ N_accepted（resync）+ 2（final target/draft 各一次）次 forward 调用，每次都带完整的 H2D + slot 计算 + `.item()` 同步。round 内 batch 节省的 GPU 时间被周边 Python overhead 吃掉还倒亏。

### 写进简历时的话术（关键加分点）

不能假装 spec 拿到了 2× speedup。要把"算法正确性已验证、Python 编排是当前瓶颈"作为故事讲：

> 在 4090 单卡上跑通 7B+1.5B 投机解码，**接受率与论文吻合**（K=1/T=0 达 86%，K=5 达 69%）。**当前实现 speedup 0.4×**（< 1）：`torch.profiler`（进展 #10）显示 decode 步 **Self CPU ≫ Self CUDA**，瓶颈在 **`model_forward` 内海量 `cudaLaunchKernel`**（launch-bound）；`slot_mapping`/小 H2D 仅占一步 CPU 的千分之几。spec 把每 round 的 forward 次数放大后，host 侧开销进一步恶化。**下一步**：CUDA Graph 或 `torch.compile` 压扁 launch，再视情况做融合算子；而非优先抠 slot 列表推导。

这个结论比"我做到了 2× 加速"更可信，也更体现真实的性能调优思维（profile → root cause → 下一步该做什么）。

### 输出文件

- `bench_results/basic_20260427_081636.json` — 1.5B baseline
- `bench_results/basic_20260427_081838.json` — 7B baseline
- `bench_results/spec_20260427_083700.json` — 完整 K-sweep × T 数据 + target-only baseline

### 下一步（plan 任务 3+）

- [ ] `--mode batch` 在 1.5B 上跑 batch_sizes=1,2,4,8,16,32（KV cache `num_gpu_blocks` 调到能容纳）
- [ ] `--mode kv-utilization` 跑长度方差大的 prompt 集合，记录 paged/naive 比例
- [ ] TP=2 本地数值等价性测试（gloo backend on CPU）
- [ ] 把以上数据填进 `docs/RESUME.md`，并修正 `docs/INTERVIEW_QA.md` Q16 的 sdpa 对齐描述（已经在 Bug #6 里发现旧描述错了）

---

## 进展 #10: `torch.profiler` — decode 路径 CPU vs GPU 精测

**目标**: 用 `torch.profiler` + `record_function` 回答「decode tps 偏低是 Python 编排慢还是 GPU kernel 慢」，并与 `_decode_batch` 子步骤一一对应。

### 脚本与产物

- 脚本: `profiler/profile_decode.py`（镜像 `engine._decode_batch` 单序列逻辑，不改动 `engine.py`）
- 默认: 1.5B、`prompt_len=128`、`warmup_steps=10`、`steps=32`、`num_gpu_blocks=2000`
- Chrome trace: `profiler/traces/decode_20260427_100326.json`（Perfetto / `chrome://tracing/`）
- 同次 stdout 存档: `profiler/traces/decode_20260427_100326.stdout.log`

### 总结论：**CPU / host 侧主导，GPU 严重欠饱和（launch-bound）**

32 步累计（PyTorch 2.9.1，`key_averages` 表底栏）：

| 指标 | 数值 |
|------|------|
| Self CPU time total | **2.742 s** |
| Self CUDA time total | **0.203 s** |

比例约 **13.5 : 1**。均摊到单步 decode：CPU ~**86 ms** / CUDA ~**6.4 ms**。若仅看 GPU 算子时间，理论上限约 **~150+ tok/s** 量级；bench 实测 ~28 tok/s 的差距主要来自 **host 派发与编排**，而非 GEMM 本身太慢。

### 与 `_decode_batch` 分段对应（每步均摊，record_function）

| 段 | 约 CPU avg | 约 CUDA avg | 占一步 CPU |
|----|------------|-------------|------------|
| `kv_append_slots` | ~23 µs | ~0 | 可忽略 |
| `compute_slot_mapping` | ~137 µs | <1 µs | ~0.16% |
| `build_input_tensors` | ~222 µs | ~1.4 µs | ~0.26% |
| **`model_forward`** | **~84.8 ms** | **~6.35 ms** | **~99%** |
| `sample` | ~148 µs | ~5.6 µs | ~0.17% |
| `item_sync` | ~232 µs | ~1 µs | ~0.27% |

**纠正进展 #9 的粗判**：`compute_slot_mapping` / 小 tensor H2D **不是**主要矛盾；矛盾在 **`model_forward` 内成千上万次小 kernel 的 CPU launch**。

### CPU 热点（self CPU Top，定性）

- `model_forward`（自定义段）与下层 `aten::*` 合计占绝大部分 CPU。
- **`cudaLaunchKernel`**：32 步内 **~55k 次** launch（约 **~1.7k / step**），self CPU 约 **18%** —— 典型「many tiny kernels → CPU 喂不饱 GPU」。
- GPU 侧 self CUDA 以 `aten::mm` / cuBLAS GEMV 为主；单次 kernel 很短，trace 上 GPU stream **白缝多**、与 CPU 线程密集小格形成对照。

### 下一步（与简历话术一致）

1. **CUDA Graph** 或 **`torch.compile(..., mode="reduce-overhead")`**：压扁 per-step launch 次数，优先于继续抠 slot mapping 微优化。
2. **Kernel 融合**（FlashAttention、融合 Norm/MLP）：在 Graph 之后仍有收益时再上。
3. **py-spy**：若需定位到具体 Python 函数栈，可在 torch.profiler 确认 launch-bound 后再开。

---

## 进展 #11: `bench spec` 复测（对比进展 #9）

**目标**: 在完成 basic 路径解耦 + CUDA Graph 提速后，复跑 `bench.py --mode spec`，验证投机解码 `speedup_vs_baseline` 是否已经 > 1。

### 复测配置（与进展 #9 对齐）

- target: `DeepSeek-R1-Distill-Qwen-7B`
- draft: `DeepSeek-R1-Distill-Qwen-1.5B`
- `prompt_len=128`、`max_tokens=128`、`warmup=1`、`runs=2`
- `K=1..8`、`temperature in {0.0, 0.7}`
- 含 target-only baseline（`--baseline`）
- 输出: `bench_results/spec_20260428_075828.json`

### 新结果摘要

- target-only baseline: **29.23 tok/s**（进展 #9 是 26.6 tok/s）
- T=0 下最好点：**K=3, decode_tps=12.41, speedup=0.425x**
- 全部配置里 `speedup_vs_baseline` 范围约 **0.24x ~ 0.43x**，**没有任何一组 > 1**

### 与进展 #9 的对比结论

- 结论不变：**spec speedup 仍然 < 1**。
- 数值上，spec decode_tps 从 #9 的约 9~11 tok/s 抬升到约 10~12 tok/s，但 target-only baseline 同时也提升了，因此比值依旧小于 1。
- 接受率曲线形态仍合理（T=0 高、T=0.7 下降，K 增大整体下降），说明算法正确性没有回退。

### 为什么 basic 提升了但 spec 仍没翻正

- 当前 CUDA Graph 提速路径主要落在 `LLMEngine._decode_batch`（basic 直接走这条路径）。
- `spec` 模式走的是 `SpeculativeDecoder` 自己的循环（`speculative.py` 中 draft/verify/resync/final 多段前向），没有复用 engine 的 graph decode runner。
- 因此 spec 仍承受多次小 forward + 频繁 Python 编排与同步，虽有局部改进，但不足以把 `speedup_vs_baseline` 推到 >1。

### 下一步（若要让 spec > 1）

1. 给 `SpeculativeDecoder` 的 draft/verify 主循环做 graph-friendly 缓冲预分配与 capture/replay（至少覆盖 verify + final 热段）。
2. 合并 round 内重复的小张量构造（`torch.tensor(...)`、`block_table/context_len`）到长期 device buffer。
3. 在 spec 路径上单独做 profiler，确认 `cudaLaunchKernel` 与 CPU 空转是否明显下降，再复跑 K-sweep。

---

## 经验总结

| 问题类型 | 关键教训 |
|----------|----------|
| dtype 管理 | 中间计算（RMSNorm、softmax）转 float32 后必须显式转回原始 dtype |
| 权重加载 | 不能假设所有模型都无 bias；应检查 safetensors 中的 key 与模型参数是否完全对应 |
| 精度选择 | 优先使用模型原生 dtype（bfloat16）；float16 在大模型中容易溢出 |
| 测试策略 | 对照 HuggingFace 参考实现做 logits 对比是最有效的正确性验证手段 |
| 显存预算 | 多模型协作（投机解码）时 KV cache 自动估算会"吃独食"，必须手动切预算或延后估算 |
| 词表对齐 | 同 tokenizer 不等于同 `vocab_size`——不同规模的模型可能 pad 到不同倍数，跨模型概率比较必须 `min(vocab)` 截断 |
| 注意力掩码 | `F.scaled_dot_product_attention(is_causal=True)` 对矩形 Q/K 是**顶端对齐**；做 speculative verify / prefix + extend 这类场景必须**手动构造右下对齐掩码** |
| 采样 API 契约 | `sample(logits)` 吃 raw logits 并内部 softmax；若误传 probs 会二次 softmax 退化为均匀分布，bug 表现为"偶发乱码 token"（非必现，很难抓） |
| 调试方法论 | "基础路径正常但组合路径异常"时，先隔离组合路径特有的代码（如 `extend_attention`、bonus 分支），不要怀疑底层模型 |
| 模型使用 | 对话精调模型（`-Chat` / `-Distill` / `-Instruct`）必须套 chat template，否则中文等长尾语种会严重退化 |
| 性能测量 | TTFT 和 decode tps 必须分开计时；用 `torch.cuda.synchronize()` 切两段，否则 prefill 的耗时会污染稳态 decode 数字 |
| 性能瓶颈定位 | "1.5B 和 7B decode tps 几乎相等" = GPU 没饱和 = CPU/Python overhead 主导。投机解码 speedup 公式只在 GPU 饱和时成立，否则 K-batched verify 节省的 GPU 时间会被 K 倍 Python 编排开销吃掉 |
| 基准测量姿势 | bench 必须 bypass `LLM.generate()` 直接驱动 `LLMEngine` 原语，才能精确分离 TTFT / decode_tps；warmup 不可省（首次 run 因 CUDA context lazy init 会偏慢一截） |
| Profiler | `torch.profiler` 的 Self CPU vs Self CUDA 总时长比能直接判断 host/GPU 谁拖后腿；Chrome trace 里 GPU stream **白缝多** = launch-bound。自定义段要镜像真实循环（如 `_decode_batch`），否则容易误判「slot 推导很慢」——实测瓶颈常在 `model_forward` 内海量 `cudaLaunchKernel`。PyTorch 2.9+ 段汇总里 GPU 时间用 `device_time_total`，旧版可能是 `cuda_time_total`，脚本里需兼容 |
| 与 vLLM 对比 | 固定 ISL/OSL 时 vLLM `bench throughput` 须设 `--random-input-len/--random-output-len`；JSON 的 `tokens_per_second` 含 prompt，与 nano `aggregate_tps`（仅生成）对齐要用输出 token 数 / `elapsed_time`。nano decode graph 仅 `len(seqs)==1`，batch 模式 B≥2 无 graph 红利 |
| CUDA Graph 构图 | **Warmup/capture 的输入必须与 `replay` 一致**：全零 `context_lens`/`slot_mapping` 会让 verify 的 `extend_attention` 在 eager warmup 与 capture 阶段处于错误 prefix 语义，捕获的图与生产前向不等价，可表现为接受率归零（见 Bug #21） |

---

## 进展 #12: 投机解码优化复盘（一）— `speedup < 1` 的定位与修复

**问题现象**:

- `bench.py --mode spec` 中，`speedup_vs_baseline` 长期小于 1（最优仅约 0.4x），与预期不符。
- 同时 basic 路径表现正常，说明不是模型权重或采样正确性整体失效。

**定位过程**:

1. 先排除算法错误：acceptance 曲线形态合理（`T=0` 高、`T=0.7` 下降，K 增大整体下降），`rejection sampling` 逻辑与单测均正常。
2. 结合 `torch.profiler` 与代码审查，锁定 `SpeculativeDecoder` 中 block table 维度异常：
   - 修复前按 `target_kv.num_blocks` 分配（例如 2048）
   - 导致 paged attention 实际按 `2048 * block_size` 计算 `max_ctx`
   - 远大于真实上下文，造成大量无效 KV gather/attention 计算
3. 对照 `LLMEngine` 的 basic decode 实现，发现 basic 按真实序列长度推导 `max_blocks`，而 spec 路径使用了 allocator 容量级维度，二者口径不一致。

**修改内容**:

- `SpeculativeDecoder` 增加 `max_seq_len`，统一以 `(max_seq_len + block_size - 1) // block_size` 作为 graph/eager 路径的 block table 上限。
- `engine.py` 中 `LLM` 初始化链路把 `max_seq_len` 透传到 `SpeculativeDecoder`。
- `profiler/bench.py` 构造 `LLM` 时显式传入与 spec 运行范围匹配的 `max_seq_len`（包含 `prompt_len / max_tokens / K` 余量）。
- `kv_cache.py::fill_block_table_padded` 做了配套优化，减少不必要的全量清零与逐元素写入开销。

**修改结果**:

- 关键拐点从 `K>=4` 开始出现 `speedup > 1`（相对 eager baseline）。
- `K=8, T=0` 可达到约 `1.5x`（相对 eager baseline），说明此前主瓶颈确实在 block table 维度错误引发的无效计算。
- acceptance/tokens-per-round 曲线保持合理，说明修复提升了性能但未破坏采样行为。

---

## 进展 #13: 投机解码优化复盘（二）— CUDA Graph 覆盖与代码结构精简

**问题现象**:

- 修复 #12 后，spec 路径虽然显著变快，但与 target 的 decode CUDA Graph 基线相比仍有差距。
- 同时 `speculative.py` 内部存在明显重复逻辑，维护成本高，后续优化风险大。

**定位过程**:

1. 审查 spec 路径 CUDA Graph 覆盖面：
   - 只覆盖了部分 target/draft forward 场景
   - draft 路径中采样和同步仍有 eager 开销
   - `T != 0` 时 graph 路径天然受限
2. 审查代码结构：
   - graph state / graph build / can-use-graph 判断存在多处重复实现
   - target verify/final 与 draft step/final 在输入缓冲与执行流程上高度相似
   - 重复代码使参数变更（如 `max_blocks`）需要多点同步修改，容易漏改

**修改方向与结果**:

- 对 `speculative.py` 做结构性精简：合并相似状态结构、统一公共构图/前向辅助路径、减少重复 buffer/重复分支。
- 在不改变算法语义的前提下，降低了实现复杂度，后续针对 spec 路径做 graph 扩展和 profiler 定位更直接。
- 结论上，第二次修改的核心收益是**可维护性与可持续优化能力提升**：把“能跑通”推进到“可持续调优”。

---

## 进展 #14: profiler 与 vLLM 对比测试（basic / batch）及性能结论

**环境**: NVIDIA RTX 4090；模型 `DeepSeek-R1-Distill-Qwen-7B`（本地路径 `/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/`）。nano 使用仓库内 `profiler/bench.py`；vLLM 使用 conda 环境 `vllm`（v0.20.x 量级）。

### 14.1 Batch 吞吐（`bench.py batch` vs `vllm bench throughput`）

**对齐 workload**: `prompt_len=128`，每条生成 `max_tokens=128`；nano 侧 `batch_sizes=1,2,4,8,16`，`warmup=1`，`runs=2`。vLLM 侧 `num-prompts == max-num-seqs == B`，并**必须**显式 `--random-input-len 128 --random-output-len 128`（否则默认 `random-input-len=1024` 会覆盖 `--input-len`）；`--max-model-len 8192`、`--gpu-memory-utilization 0.92`、`--trust-remote-code`。

**vLLM 等价命令模板**（对每个 `BS` 执行一次）:

```bash
conda activate vllm
vllm bench throughput \
  --model "$MODEL" --tokenizer "$MODEL" \
  --max-model-len 8192 --gpu-memory-utilization 0.92 --trust-remote-code \
  --random-input-len 128 --random-output-len 128 --random-range-ratio 0.0 \
  --num-prompts "$BS" --max-num-seqs "$BS" --tensor-parallel-size 1 \
  --output-json "bench_results/vllm_throughput_bs${BS}.json"
```

**指标口径**: nano 的 `aggregate_tps` 为**仅生成 token** 总和 / 墙钟；vLLM JSON 里的 `tokens_per_second` 含 prompt+生成，对比时应使用 **`(num_requests * 128) / elapsed_time`** 作为生成吞吐。

| batch_size | nano `aggregate_tps` | vLLM 生成 tok/s | vLLM / nano |
|------------|----------------------|-----------------|-------------|
| 1 | ~54.1 | ~40.3 | ~0.75×（单流 vLLM 冷启动/前端路径更重，见 JSON 单次） |
| 2 | ~60.8 | ~86.3 | ~1.42× |
| 4 | ~118.9 | ~195.3 | ~1.64× |
| 8 | ~230.6 | ~383.3 | ~1.66× |
| 16 | ~435.4 | ~731.0 | ~1.68× |

**原始结果文件**: `bench_results/batch_20260506_083245.json`；`bench_results/vllm_throughput_bs{1,2,4,8,16}.json`；汇总说明 `bench_results/batch_nano_vs_vllm.json`。

**Batch 结论要点**:

1. **nano 仅在 `decode` 批大小为 1 且 greedy 时走 CUDA Graph**（`engine._can_use_cuda_graph`: `len(seqs) != 1` 则 false）。因此 **batch 基准里 B≥2 时全程 eager**，无法吃到与 basic 单流相同的 graph 红利；vLLM 仍可对多档 batch 做 graph/capture。
2. **decode 注意力仍是教学实现**：`decode_paged_attention` 按 `max_ctx = max_blocks * block_size` 展开再全长 matmul+softmax，算力随上限上下文缩放；vLLM 使用 FlashAttention-2 等专用后端与成熟 paged kernel。
3. vLLM bench 默认采样与 nano `temperature=0` 不完全一致；对「倍量级」差距解释力有限，主因仍是 **内核 + 多序列 graph + 调度**。

### 14.2 Basic（`bench.py basic` vs 流式对齐脚本）

**nano 命令**:

```bash
python profiler/bench.py basic \
  --model /model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ \
  --prompt-len 128 --max-tokens 128 --runs 3 --warmup 1
# 开启 decode CUDA Graph（仅 batch=1 greedy 路径）:
python profiler/bench.py basic ... --cuda-graph
```

**vLLM 对齐脚本**（与 `bench.make_token_ids` 同源 prompt、greedy、`ignore_eos`，流式拆 TTFT / decode）：`profiler/vllm_basic_match_bench.py`。

```bash
conda activate vllm
python profiler/vllm_basic_match_bench.py \
  --model /model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/ \
  --prompt-len 128 --max-tokens 128 --runs 3 --warmup 1 \
  --max-model-len 8192 --gpu-memory-utilization 0.92
```

**实测均值（同机同模型）**:

| 配置 | TTFT（约） | decode_tps（约） |
|------|------------|------------------|
| nano，无 `--cuda-graph` | ~27.7 ms | ~31.3 tok/s |
| nano，`--cuda-graph` | ~28.4 ms | ~54.2 tok/s（约 **1.73×**，graph 主要省 launch/Python） |
| vLLM 流式对齐 | ~19.5 ms | ~63.3 tok/s |

**原始结果文件**: `bench_results/basic_20260506_090209.json`（无 graph）；`bench_results/basic_20260506_091252.json`（有 graph）；`bench_results/vllm_basic_match_20260506_090525.json`；对照说明 `bench_results/basic_nano_vs_vllm.json`。

**Basic 结论要点**:

1. vLLM 优势来自 **专用注意力内核 + torch.compile/inductor + 多档 CUDAGraph + 算子融合与调度**，nano 为教学体量不会在默认路径复制整套栈。
2. nano 开 graph 后 decode 明显提升，但 **graph 无法缩小「decode 按 max_ctx 稠密注意力」的算法开销**；与 vLLM 剩余差距主要来自该层与内核等级差异。
3. **仅靠小改动无法「追齐」vLLM**；若要接近，需 **替换 decode 注意力实现或引入等价 fused/paged kernel**（及可选的多序列 graph），属于中等以上工程量，与「~2000 行教学框架」目标需权衡。

### 14.3 经验表补充（可并入面试叙事）

| 主题 | 要点 |
|------|------|
| vLLM bench 默认数据集 | 未指定 dataset 时走 random；`--random-input-len` 默认 1024 会覆盖 `--input-len`，与 nano 固定 ISL 对齐时必须显式设 128 |
| vLLM KV 与 `max_model_len` | 长 `max_model_len` 会按「至少服务满上下文」预留 KV；4090 上 7B 需限制 `--max-model-len` 或提高 `gpu_memory_utilization` |
| nano batch vs graph | `len(seqs)==1` 才启用 decode CUDA Graph；batch sweep 的 B≥2 无 graph 红利，与 basic 单流对比时要单独说明 |
| 指标对齐 | basic 拆 TTFT / decode；batch 比生成吞吐时 vLLM 勿直接用 JSON 的 `tokens_per_second`（含 prompt） |

---

## 进展 #15: 基准工具精简与脚本化对齐（2026-05-08）

**目标**: 将 `profiler/bench.py` 从大而全脚本收敛为面向当前需求的最小工具（仅 basic/spec），并补齐与 `vllm_benchmark.sh` 对齐的一键运行脚本。

### 15.1 `bench.py` 重构（仅保留 basic/spec）

**本次改动**:

1. 删除 `batch` / `kv-utilization` 模式与相关 CLI、执行分支。
2. 参数统一为 vLLM 语义：
   - `--input-len`
   - `--output-len`
   - `--num-prompts`
   - `--max-num-seqs`
   - `--max-model-len`
   - `--tensor-parallel-size`
3. `spec` 收敛为单路径 K 扫描：`--k-values`（如 `1,2,3,4,5,6,7`），去掉历史双轨参数与分叉开关。
4. 输出结构保留丰富指标，但主口径统一在 `throughput` 字段（`elapsed_s` / `request_throughput_rps` / `output_token_throughput_tps`）。

**代码结构调整**:

- `profiler/bench.py`：仅保留 CLI 与模式分发（轻量入口）。
- `profiler/bench_core.py`：承载 basic/spec 的核心执行逻辑与指标汇总。

**结果**:

- `bench.py` 从千行级降到双位数行数（当前约 89 行）。
- `python profiler/bench.py --help` 只显示 `{basic,spec}` 两个子命令，符合预期最小参数面。

### 15.2 新增 `profiler/nano_benchmark.sh`

为对齐 `profiler/vllm_benchmark.sh` 的使用体验，新增 nano 侧一键脚本：

- 流程：`baseline` + `k=1..7` 扫描。
- 变量风格与 vLLM 脚本保持一致（`TARGET_MODEL` / `DRAFT_MODEL` / `INPUT_LEN` / `OUTPUT_LEN` / `NUM_PROMPTS` 等）。
- 调用路径：
  - baseline: `python profiler/bench.py basic ...`
  - spec: `python profiler/bench.py spec ... --k-values "${K}"`
- 汇总方式：从每轮生成的 JSON 提取吞吐与 spec 指标（accept_rate / tokens_per_round / speedup_vs_baseline）写入 `summary.log` 并打印最终对比表。

**校验**:

- `bash -n profiler/nano_benchmark.sh` 通过。
- 脚本已设置可执行权限：`chmod +x profiler/nano_benchmark.sh`。

### 15.3 当前状态与下一步

- 当前基准入口已统一为：
  - 交互式命令：`profiler/bench.py`（basic/spec）
  - 批量脚本：`profiler/nano_benchmark.sh`
- 下一步可选项：
  1. 给 `nano_benchmark.sh` 增加 `FAST_MODE`（小 `NUM_PROMPTS` + 小 `K_VALUES`）用于快速 smoke。
  2. 追加 nano/vLLM 双脚本结果的同口径汇总 JSON，方便自动化对比与画图。

---

## 进展 #16: H10 机器 vLLM 基准复测（OSL=256 / 512）

**背景**: 在 H10 机器上使用 `profiler/vllm_benchmark.sh` 连续跑了 2 轮，分别测试 `OUTPUT_LEN=256` 和 `OUTPUT_LEN=512`。以下记录按你提供的汇总值整理。

### 16.1 OSL=256 结果

| 配置 | requests/s | total tokens/s | output tokens/s | 相对 baseline（output） |
|------|-----------:|---------------:|----------------:|------------------------:|
| baseline | 0.19 | 71.21 | 47.47 | 1.00x |
| spec_k1 | 0.25 | 97.24 | 64.83 | 1.37x |
| spec_k2 | 0.29 | 113.27 | 75.51 | 1.59x |
| spec_k3 | 0.31 | 118.83 | 79.22 | 1.67x |
| spec_k4 | 0.32 | 120.97 | 80.65 | **1.70x** |
| spec_k5 | 0.31 | 118.72 | 79.15 | 1.67x |
| spec_k6 | 0.30 | 115.76 | 77.17 | 1.63x |
| spec_k7 | 0.26 | 100.40 | 66.93 | 1.41x |

### 16.2 OSL=512 结果

| 配置 | requests/s | total tokens/s | output tokens/s | 相对 baseline（output） |
|------|-----------:|---------------:|----------------:|------------------------:|
| baseline | 0.09 | 59.65 | 47.72 | 1.00x |
| spec_k1 | 0.13 | 83.70 | 66.96 | 1.40x |
| spec_k2 | 0.15 | 98.23 | 78.59 | 1.65x |
| spec_k3 | 0.16 | 105.29 | 84.23 | 1.77x |
| spec_k4 | 0.17 | 108.50 | 86.80 | **1.82x** |
| spec_k5 | 0.17 | 108.09 | 86.47 | 1.81x |
| spec_k6 | 0.16 | 104.04 | 83.24 | 1.74x |
| spec_k7 | 0.15 | 97.53 | 78.02 | 1.64x |

### 16.3 结论摘要

1. 两组 OSL 都呈现一致趋势：`K` 从 1 增长到 4 左右时吞吐提升明显，`K>=5` 后进入平台并在更大 K 出现回落。
2. 当前最佳点都在 `K=4` 附近：
   - `OSL=256`: `80.65 output tok/s`（约 `1.70x`）
   - `OSL=512`: `86.80 output tok/s`（约 `1.82x`）
3. baseline 的 output 吞吐在两组长度下接近（47.47 vs 47.72 tok/s），说明在当前配置下 target-only decode 稳态速度较稳定；spec 的收益主要体现在每轮接受更多 token 带来的有效前进速度提升。

---

## 进展 #17: 投机解码优化与基准工具增强（2026-05-09）

**目标**: 以"少代码优先、允许到注意力内核级"为约束，制定 7B→32B speculative 路径追近 vLLM 部分性能的 ROI 改造路线，并按优先级落地第一批改动。

### 17.1 Speculative CUDA Graph 覆盖增强（P0-1）

**问题**: `SpeculativeDecoder` 里 draft 侧的 step / resync / final 三个前向路径各自组装 buffer/metadata，存在大量重复代码和临时张量构造；draft graph 仅按 `seq_id` 缓存、且只覆盖单步 decode，resync 和 final 无法命中 graph。

**修改内容** (`nano_dist_spec/speculative.py`, +96 / -65):

1. 抽出统一的 draft 前向入口 `_run_draft_forward(seq_id, start_pos, tokens, params)`：
   - graph 路径：按 `seq_len`（而非 `seq_id`）缓存 draft graph，支持 n=1 和 n>1 复用。
   - eager 路径：复用 `_resync_inp/_resync_pos/_resync_slots` 长期 buffer。
2. `_draft_step_one` / `_resync_draft_batched` / `_draft_final_one` 全部改为调用 `_run_draft_forward`：
   - 消除三套重复的 `compute_slot_mapping_into` + `fill_block_table_padded` + `InputMetadata` 构造。
   - `resync` 和 `final` 新增 `params` 参数，使其可走 graph 路径。
3. draft graph 缓存键从 `seq_id` 改为 `seq_len`：
   - n=1 时保留 warmup 采样路径（`warmup_extra`）。
   - n>1 时仅构图、不采样（resync 不需要采样输出）。

**验证**: `python -m pytest tests/test_speculative.py -v` 全部通过（7/7）。

### 17.2 基准工具 `--num-gpu-blocks` 参数

**问题**: `profiler/bench.py` 的 spec 模式没有暴露 `num_gpu_blocks`，在 24GB 卡上自动估算会把 KV 吃满，导致双模型（7B+1.5B 或 32B+7B）OOM。

**修改内容**:

- `profiler/bench.py`: 新增共享参数 `--num-gpu-blocks`（默认 `None`）。
- `profiler/bench_core.py`: `SharedArgs` 新增 `num_gpu_blocks: Optional[int]`；`_build_target_engine` 和 `run_spec` 中创建 `LLM` 时透传。
- `profiler/nano_benchmark.sh`: 新增可配置变量 `NUM_GPU_BLOCKS=""`，非空时自动拼接到 `basic/spec` 命令。

**验证**: `python profiler/bench.py spec ... --num-gpu-blocks 128` 正常执行，JSON 输出包含 `"num_gpu_blocks": 128`。

### 17.3 `run_basic` 默认启用 CUDA Graph

**问题**: `run_basic` 内部 `_build_target_engine(... use_cuda_graph=False)`，benchmark 的 baseline 未利用 graph 提速。

**修改内容** (`profiler/bench_core.py`):

- `_build_target_engine` 新增 `use_cuda_graph: bool = True` 参数。
- `run_basic()` 默认走 `use_cuda_graph=True`。
- `run_spec()` 的内部 baseline 已移除（见 17.4），不影响。

### 17.4 `run_spec` 移除内部 baseline 跑分

**问题**: `run_spec` 内部先加载一次 target-only baseline，再加载 target+draft，两次模型加载在 24GB 卡上容易 OOM；且外层 `nano_benchmark.sh` 已单独执行了 `basic` baseline。

**修改内容** (`profiler/bench_core.py`):

- 移除 `run_spec` 内的 `_build_target_engine` + `_bench_basic_prompt_set` baseline 测量。
- 移除 sweep 中每项的 `speedup_vs_baseline` 计算。
- 移除返回 JSON 里的顶层 `"baseline"` 字段。
- 加速比由用户根据 `basic` JSON 与 `spec` JSON 自行计算。

### 17.5 Random Prompt 对齐 vLLM

**问题**: 原有 `make_token_ids` 使用固定英文故事重复填充，与 vLLM `bench throughput --random-input-len L --random-range-ratio 0` 的 i.i.d. 随机 token 模式不对齐，导致接受率虚高（固定 prompt 下 draft/target 几乎完全一致）。

**修改内容** (`profiler/bench_core.py`, +80 / -16):

1. 新增 `make_random_token_ids(tokenizer, length, rng)`：
   - 在 `[0, vocab_size)` 均匀抽样，排除 BOS/EOS/PAD。
   - 对齐 vLLM 的 synthetic prompt 语义。
2. 新增 `_make_prompt_list(tokenizer, shared, rng)`：
   - `prompt_mode="fixed"`: 原有行为（同一段文本重复）。
   - `prompt_mode="random"`: 每个 prompt 独立采样。
3. `SharedArgs` 新增 `prompt_mode: Literal["fixed", "random"] = "random"` 和 `bench_seed: int = 42`。
4. `_bench_basic_prompt_set` / `_bench_spec_prompt_set` 改为接收 `prompts: Sequence[Sequence[int]]`，每条请求使用独立 prompt。
5. `run_spec` 的 K 扫描对同一组 `prompts` 复用，保证不同 K 之间可比。
6. `profiler/bench.py` 新增 `--prompt-mode {random,fixed}` 和 `--bench-seed` CLI 参数。

### 17.6 SpecDecoding 指标对齐 vLLM

**问题**: 原有 `draft_accept_rate_by_pos` 是前缀累积率（`count[pos] / num_rounds`），与 vLLM 的 "Per-position acceptance rate"（条件概率）不一致；且缺少 vLLM 的 `Mean acceptance length`、`Accepted/Drafted throughput` 等关键指标。

**修改内容** (`profiler/bench_core.py`, `profiler/nano_benchmark.sh`):

1. `_bench_spec_single` / `_bench_spec_prompt_set` 新增统计：
   - `drafted_counts_by_pos[i]`: draft 循环实际走到位置 i 的轮次数（条件概率的分母）。
   - `per_pos_accept_rate[i]`: `accept_by_pos[i] / drafted_by_pos[i]`，即 `P(accept at pos i | loop reached pos i)`。
   - `mean_acceptance_length`: 每轮平均接受的 draft token 数。
   - `accepted_throughput`: 被 target 接受的 draft token 吞吐 (tok/s)。
   - `drafted_throughput`: draft 模型总产出 token 吞吐 (tok/s)。
2. `nano_benchmark.sh` 的 `extract_spec_metrics` 输出格式对齐 vLLM：
   ```
   SpecDecoding metrics: Mean acceptance length: 3.96, Accepted throughput: 40.93 tokens/s, Drafted throughput: 69.22 tokens/s, Accepted: 411 tokens, Drafted: 695 tokens
   Per-position acceptance rate: 0.727, 0.647, 0.547, 0.518, 0.518
   ```

### 17.7 Debug 脚本

**新增** `profiler/debug_spec_accept_by_pos.py`：

- 支持两种模式：
  - `--from-json <spec.json>`: 只解析已有 bench JSON（不占 GPU）。
  - `--target-model ... --draft-model ...`: 现场跑一轮并输出逐位分析。
- 输出三种率：cumulative_rate（前缀累积）、conditional_rate（逐步条件）、per_pos_accept_rate（精确条件，来自 `drafted_counts_by_pos`）。
- 打印 vLLM 风格的汇总指标。

### 17.8 实测数据（7B target + 1.5B draft, random prompt）

使用 `nano_benchmark.sh`（`NUM_GPU_BLOCKS=128`, `NUM_PROMPTS=5`, `INPUT_LEN=128`, `OUTPUT_LEN=256`, `PROMPT_MODE=random`）：

| K | output tok/s | accept_rate | tokens_per_round | num_rounds |
|---|---:|---:|---:|---:|
| baseline | 54.71 | — | — | — |
| 1 | 33.23 | 99.2% | 1.99 | 640 |
| 2 | 37.78 | 98.8% | 2.98 | 320 |
| 3 | 44.76 | 98.5% | 3.95 | 213 |
| 4 | 51.23 | 98.1% | 4.92 | 160 |
| 5 | 55.65 | 97.7% | 5.89 | 128 |
| 6 | 59.91 | 97.4% | 6.84 | 110 |
| 7 | 64.08 | 97.0% | 7.79 | 165 |

**关键发现**:

1. **接受率极高且几乎不随 pos 下降**: K=7 时 `Per-position acceptance rate` 七个位置都在 ~0.97，`conditional_rate` 全为 1.0。原因是 random prompt 在 7B/1.5B 上都是"无意义 token 序列"，draft 和 target 对这类输入的"续写"几乎没有分歧——两者都不太可能对纯随机 token 形成强偏好差异。
2. **吞吐随 K 单调上升**: 因为接受率接近 100%，每轮多投入 K 步 draft 的成本几乎全被「更少轮数 / 更少固定编排开销」抵消。这与论文里"大 K 后期大量被拒"的典型曲线不一致，属于 workload 特性而非实现 bug。
3. **K=7 spec (64 tok/s) 已超过 baseline (55 tok/s)**: 在高接受率 + 单流模式下，spec 的收益已经翻正。
4. **与 vLLM K=4 最优（1.7x）的差异**: vLLM 使用真实对话数据（chat template），接受率会随 pos 显著下降；当前 random prompt 无法复现该曲线，需后续换真实数据验证。

> ⚠️ **后续修正（见 #18/#19）**：上面四条结论其实是「last_token 漏写 target KV」off-by-one bug 的产物——target KV 整体被左移一位，导致 verify 时拿错位的 logits 与 draft 比较，于是首位接受率虚高、随 pos 单调上升、加速比不饱和。该 bug 修复后，曲线恢复成「先升后稳/略降」的典型形态，第一位接受率也降到与 vLLM 类似的范围。

### 17.9 待做 / 下一步

- [ ] 使用 chat template + 真实对话 prompt 重新跑 K-sweep，观察 `per_pos_accept_rate` 是否随 pos 下降。
- [ ] P1: 替换 `decode_paged_attention` 教学实现为更高效内核（当前按 `max_ctx` 稠密展开，是与 vLLM 的主要结构差距）。
- [ ] P2: 控制流微优化打包（`append_slots` 批量化、final 逻辑融合、减少 `.item()` 同步、自适应 K）。
- [ ] 在更大显存或 TP 环境下验证 32B+7B 组合。

---

## 进展 #18: 投机解码 `last_token` 漏写 target KV 的 off-by-one Bug 诊断（2026-05-11）

**背景**: #17.8 实测在 7B+1.5B 上观察到三条与论文 / vLLM 都不符的反常现象：

1. **加速比随 K 单调上升**，没有出现典型的"先升后降"。
2. **per-position 接受率随 pos 递增**（K=7 时 0.97 → 1.00），不是论文里随 pos 衰减。
3. **第一位接受率异常高**（~0.97），远高于 vLLM 在同模型族 random prompt 下的水平。

最初怀疑是 `nano_dist_spec/attention.py` 的路径分流问题——直觉是「verify 的 K-token forward 因为 `block_tables is not None` 走了 `decode_paged_attention`，而 decode 路径只检查 `pos < context_lens`，缺 query 间 causal mask，导致 K 个 query 互相能看见对方」。

### 18.1 第一个假设被证伪：attention 路径其实是对的

通读 `nano_dist_spec/model.py` 的 `Attention.attend()` 分流逻辑 + `nano_dist_spec/attention.py` 三条路径的实现，确认：

- `seq_len == 1` → `decode_paged_attention`（无 causal mask 需求）。
- `seq_len > 1 and block_tables is not None` → `extend_attention`（K-token verify 走这条）。
- `extend_attention` 内部 `mask = q_pos[:, None] >= k_pos[None, :]`，**有正确的 causal mask**。

所以"K 个 query 互相能看见对方"的假设不成立。验证后排除 attention 实现问题。

### 18.2 真根因：verify 时 `last_token` 没写进 target KV

回到 `nano_dist_spec/speculative.py:speculative_step` 的验证 forward 入口。原实现是：

```python
# 上一轮采样出来的 last_token（即 "上一轮 bonus 或首 token"）已经在 prompt 末尾，
# 但只塞了 logits/probs，没塞回 target 的 KV cache。
out = _run_target_forward(seq_id, start_pos=P, tokens=draft_tokens)  # K 个 draft token
```

`last_token` 是上一轮 step 的输出（或 `prefill` 的 `first_token`），它对应的 query/KV 从未在 target 上 forward 过：

- prefix 已写到 KV 的位置: `[0, P-1]` 共 P 个。
- 当前 step 写入: `[P, P+K-1]` 共 K 个 draft token。
- **`last_token` 应该在 KV 的位置 `P`，但被跳过了**。

结果是：每轮 verify 的 target 看到的 KV 比"真实历史"少一项，整条 KV 被左移一位。这造成：

1. **首位接受率虚高**: target 在错位 KV 下产生的 logits，恰好对应 draft 已经看过的同一历史；两者预测高度一致，几乎必接受。
2. **per-pos 随 pos 上升**: 错位累积，后期 KV「错的越久越自洽」（target 自己产生的错位 logits 会喂给下一步），与同样错位的 draft KV 越来越像。
3. **加速比单调升**: 接受率因 bug 被拉到不合理高位，每轮接受越多 token，K 越大越赚——但这是假象，并非真实算法行为。

### 18.3 为什么 `verify_draft_kv_against_groundtruth` 没抓到

`speculative.py:657` 已有的调试函数会比对 draft 侧 KV 与"ground truth"。但 ground truth 自己也是从 `first_token` / `last_token` 拼出来的——同样漏了 `last_token`，于是与漏了 `last_token` 的 draft KV 自洽，校验通过。这也是为什么 bug 能潜伏这么久。

### 18.4 诊断阶段结论

- attention 实现没问题；问题在 `speculative_step` 的 verify 输入构造。
- 修复方向：把 `last_token` 拼进 verify forward 的 token 序列，即用 `[last_token] + draft_tokens` 喂给 target，对应写 `P` 到 `P+K`（共 K+1 个位置）的 KV。
- 同步需要修：buffer 容量 +1、cache rollback 边界 +1、ground truth 校验输入对齐、metric 取 logits 的索引偏移。

修复实现与正确性回归见 #19，metric 含义引发的二次困惑与口径对齐见 #20。

---

## 进展 #19: K+1 token verify 重构与端到端正确性回归（2026-05-11）

**目标**: 把 #18 定位的 off-by-one 修干净，并补一条"draft == target 时严格等价于 baseline greedy"的端到端测试，作为该路径以后任何重构的不变量保护。

### 19.1 `speculative.py` 主重构：verify 输入扩到 K+1

**修改内容** (`nano_dist_spec/speculative.py`):

1. `__init__`: 三个 target 侧 buffer 容量从 `cap` 提到 `cap + 1`：
   ```python
   self._target_inp    = torch.empty(cap + 1, dtype=torch.long,  device=device)
   self._target_pos    = torch.empty(cap + 1, dtype=torch.long,  device=device)
   self._target_slots  = torch.empty(cap + 1, dtype=torch.long,  device=device)
   ```
   多出来的那一格用来承载本轮的 `last_token`。

2. `prefill`: 返回从 `(first_token, saved_probs)` 简化为单值 `first_token: int`。`first_token` 不再立即写 KV，而是作为"pending"留到下一轮 spec step 由 verify forward 写入——彻底消灭"采到了但没塞 KV"这个错位状态。

3. `speculative_step`:
   - 签名移除 `saved_target_probs` 参数（不再需要跨步保存 probs）。
   - verify 输入构造改成 `tokens = [last_token] + draft_tokens`，循环边界 `for _ in range(K + 1)`。
   - 取 logits 时索引从 `[:K-1]` 改为 `[:K]`：现在 logits 序列长度是 K+1，前 K 个对应 verify K 个 draft token，第 K+1 个是"全接受时用来采 bonus"的 logits。
   - rollback / resync 逻辑重写：
     - 全接受路径: `target_mgr` / `draft_mgr` 各 `append_slots(seq_id, 1)`，再用 `_run_draft_forward` 把 `d_{K-1}` 写到 draft 的 P+K 位置，保证下一轮 draft 起点对齐。
     - 拒绝路径（第 j 位拒绝）: 两个 KV manager 同步 rollback 到 `prefix_len + n_draft_accepted + 1`（多出来的"+1"就是被采纳了的 `last_token` 那一格）。
   - 删除 `_resync_draft_batched` / `_draft_final_one` 两个辅助方法和原来的"final target/draft forward"分支——它们存在的唯一原因就是补 `last_token` 没写 KV 留下的窟窿，现在窟窿在源头堵死了，这些迂回路径不需要了。

4. 同步联动:
   - `nano_dist_spec/engine.py::_generate_speculative` 改成不传/不收 `saved_probs`。
   - `profiler/bench_core.py::run_spec_benchmark` 调用点同步删掉 `saved_probs`。同时把内部调试用的 `ground_truth` 跟踪改正：`ground_truth = prompt_ids`（不再预先 append `first_token`），新增 `pending_token` 局部变量在每轮开头追加；这样 ground truth 与"`last_token` 还没写 KV"的真实状态对齐。
   - `profiler/profile_spec.py::SEGMENT_NAMES` 删掉 `resync_draft` / `final`，新增 `draft_write_last`。

### 19.2 新增端到端正确性测试

**新增**: `tests/test_spec_end_to_end.py`

核心思路：**draft 模型 = target 模型时，speculative decoding 必须 token-for-token 等于 baseline greedy**。这条不变量与具体接受率、具体采样温度都无关，是 spec 算法的最强正确性边界。

```python
@pytest.mark.parametrize("k", [1, 4])
def test_spec_matches_baseline_when_draft_equals_target(k):
    # 同一份权重、dtype=float32、greedy
    llm_spec     = LLM(..., draft_model_path=SAME, num_speculative_tokens=k, dtype="float32")
    llm_baseline = LLM(..., dtype="float32")
    out_spec     = llm_spec.generate(prompt, max_new_tokens=N, temperature=0.0)
    out_baseline = llm_baseline.generate(prompt, max_new_tokens=N, temperature=0.0)
    assert out_spec == out_baseline   # 严格逐 token 相等
```

**dtype 注记**: 测试只在 `float32` 下断言严格等价。`bfloat16` 下会偶发首次分歧（实测在 pos 3 左右出现一个 token 偏差），原因不是算法 bug，而是数值精度：

- baseline 跑的全是 `decode_paged_attention` (seq_len=1)。
- spec 跑的混合 `extend_attention` (seq_len=K+1 verify) + `decode_paged_attention` (写 K 位置)。
- 两条 kernel 路径的 reduce 顺序不同；在 `bf16` 7 位尾数下，logits 接近平局时 argmax 翻转。
- 切到 `fp32` 后 reduce 误差量级远低于 logits 分辨率，结果一致。

这是 nano 教学实现选择的精度/简洁性 trade-off，不是 bug；测试里有 docstring 注明。

### 19.3 回归验证

- 单元测试: `python -m pytest tests/ -v` 全绿，含新增 `test_spec_end_to_end.py`（float32 模式严格 token 一致）。
- 7B + 1.5B 实测（`nano_spec_bench_logs_20260511_131227/bench_results/spec_*.json`, 修复后）:

| K | output tok/s | accept_rate | tok/round | per_pos (vLLM 口径，见 #20) |
|---|---:|---:|---:|---|
| 1 | 18.99 | 96.76% | 1.97 | 0.968 |
| 2 | 21.02 | 95.23% | 2.91 | 0.955, 0.950 |
| 3 | 22.14 | 93.37% | 3.80 | 0.941, 0.932, 0.929 |
| 4 | 22.88 | 91.67% | 4.67 | 0.924, 0.917, 0.913, 0.913 |
| 5 | **23.48** | 90.43% | 5.52 | 0.914, 0.905, 0.901, 0.901, 0.901 |
| 6 | 23.13 | 88.81% | 6.33 | 0.902, 0.892, 0.887, 0.887, 0.882, 0.877 |
| 7 | 23.40 | 87.45% | 7.12 | 0.890, 0.878, 0.873, 0.873, 0.873, 0.867, 0.867 |

修复后曲线特性（对照 #17.8 修复前）:

1. **per-pos 接受率单调非递增** ✓（修复前是上升，K=7 末位 ≈ 1.0；修复后 K=7 末位 ≈ 0.867）。
2. **第一位接受率从 ~0.97 降到 ~0.89–0.96**（同模型族 + random prompt 下仍偏高，是 workload 性质而非 bug）。
3. **加速比"先升后稳"**: 7B+1.5B 上 K=5 达到 23.48 tok/s 后进入平台；H20 7B+32B 上更明显的"先升后降"（用户实测 K=3 峰值 17.87 tok/s 后回落到 K=7 的 15.94 tok/s），与论文 / vLLM 形态一致。
4. **整体 accept_rate 随 K 平稳衰减**: K=1 96.76% → K=7 87.45%，单调下降，符合 K 越大 draft 越激进、被拒概率越高的直觉。

---

## 进展 #20: Per-position 接受率口径与命名对齐 vLLM（2026-05-11）

**背景**: #19 修完之后，仍有一项"看起来不对"的现象残留——nano 输出的 `per_pos_accept_rate` 依然随 pos **递增**（例如 K=7 时 0.69 → 0.76 → 0.87 → 0.92 → 0.92 → 0.93 → 0.96），而 vLLM 运行时 metrics 显示的是单调**递减**。第一反应又是怀疑 bug，但深挖之后发现是 **nano 和 vLLM 在用同一个 label 报两种不同的口径**。

### 20.1 两种口径的区别

| 口径 | 定义 | 分母含义 | 单调性 |
|---|---|---|---|
| **无条件率** (vLLM "Per-position acceptance rate") | `accept_by_pos[i] / num_rounds` | 所有轮次 | 单调非递增（构造性的） |
| **条件率** (nano 原 `per_pos_accept_rate`) | `accept_by_pos[i] / drafted_by_pos[i]` | 仅"走到 pos i"的轮次 | 可能随 pos **上升** |

条件率上升源于 **survivorship bias**：能走到 pos 5 的，都是前 5 位都被接受的"幸运轮"；这些轮里 draft / target 高度一致，于是 pos 5 自身被接受的概率也高。两个口径都对，但描述的是不同的问题——

- **vLLM 选无条件率**: 把它当"draft 在这个 K 配置下平均能跑多远"的指标，直接乘 num_rounds 就是该位置总接受数，便于吞吐推算。
- **nano 原来选条件率**: 把它当"给定历史已接受，下一位还能接受多少"的诊断指标。在 K-sweep 比较时不直观。

二者数据 `bench_core` 都已经在算（`draft_accept_rate_by_pos` 是无条件率，`per_pos_accept_rate` 是条件率），只是默认报告口径和字段命名都和 vLLM 错位。

### 20.2 修改内容（命名 + 默认报告口径双对齐）

1. **`profiler/bench_core.py`**: 在 `_bench_spec_single` 和 `_bench_spec_prompt_set` 两处:
   - 将原 `per_pos_accept_rate`（条件率）**重命名为** `per_pos_accept_rate_conditional`，命名上明确"这是 nano 内部诊断指标，不是 vLLM 那个"。
   - `draft_accept_rate_by_pos`（无条件率）字段名保留，并在输出 dict 边上加注释明确"这是 vLLM 对齐口径，按构造单调非递增，是默认展示项"。
   - 同步删掉原先写在条件率上方的误导注释 `# vLLM: "Per-position acceptance rate"`。

2. **`profiler/nano_benchmark.sh`** 的 `extract_spec_metrics`:
   - 读取字段从 `per_pos_accept_rate` 切到 `draft_accept_rate_by_pos`。
   - 输出行追加口径注解：
     ```
     Per-position acceptance rate: 0.890, 0.878, ...   # accept[i] / num_rounds, aligned with vLLM logs (monotonically non-increasing)
     ```

3. **`profiler/debug_spec_accept_by_pos.py`**:
   - 修正 `_conditional_rates` docstring 里"条件率匹配 vLLM Per-position acceptance rate"的错误说明，改为明确指出 vLLM 报的是 `cumul_rate` 列（即 `count[i] / num_rounds`），条件率仅作为生存偏倚的诊断输出。
   - 读取处加 fallback `row.get("per_pos_accept_rate_conditional", row.get("per_pos_accept_rate"))`，旧 JSON 仍可解析。

### 20.3 验证

- `profiler/nano_benchmark.sh` 中 `extract_spec_metrics` 用今天的实测 JSON 跑一遍：
  ```
  Per-position acceptance rate: 0.890, 0.878, 0.873, 0.873, 0.873, 0.867, 0.867
    # accept[i] / num_rounds, aligned with vLLM logs (monotonically non-increasing)
  ```
  0.890 → 0.867 单调非递增，曲线形态与 vLLM 一致。
- `debug_spec_accept_by_pos.py --from-json <old_json>` 在旧 schema（仅有 `per_pos_accept_rate`）下仍能输出三列 `cumul_rate / cond_rate`，向后兼容。

### 20.4 最终命名对照

| 含义 | nano JSON 字段（统一后） | nano 默认展示 |
|---|---|---|
| vLLM "Per-position acceptance rate"（无条件） | `draft_accept_rate_by_pos` | `nano_benchmark.sh` 的 `Per-position acceptance rate:` 行 |
| nano 内部生存诊断（条件） | `per_pos_accept_rate_conditional` | 仅在 `debug_spec_accept_by_pos.py` 的 `cond_rate` 列 |

> 教训记一笔：**指标名字和文档注释必须明确写清楚分母是什么**。今天这场误判（先以为是 bug 二次发作，再深挖才发现是命名歧义）完全可以靠把字段名从 `per_pos_accept_rate` 改成 `per_pos_accept_rate_conditional`、并在 dict 旁边写一行注释来提前避免。后续任何"按位 / 按 step / 按 token"的统计字段都应遵循「名字带分母提示 + 邻近注释写 vLLM 是否对齐」的规则。

### 20.5 待做 / 下一步（接 #17.9）

- [ ] 用 chat template + 真实对话 prompt 重跑 K-sweep，验证 first-pos 接受率是否会从当前的 ~0.89 进一步下降（同模型族 random prompt 上限仍偏高）。
- [ ] 把"K-token verify forward 输入要包含 last_token"这一点写进 `nano_dist_spec/speculative.py` 顶部 docstring 的算法描述，避免未来重构再次踩同一个坑。

---

## Bug #21: 投机解码开启 CUDA Graph 后接受率掉到 0（eager 正常）

**现象**（`bench.py spec` + `SpeculativeDecoder.use_cuda_graph=True`，greedy / `T=0`）：

- `draft_accept_rate`、`total_draft_accepted` 等为 **0**；`drafted_counts_by_pos` 形如 `[N, 0, ...]`（第二位从未进入拒绝循环），等价于 **每个 round 在验证第一个 draft token 处必拒**。
- 同一配置关闭 graph（eager verify）时接受率曲线正常。

**根因**：`speculative.py::_build_graph` 在 **warmup（独立 CUDA stream）** 与 **`torch.cuda.graph` 捕获** 阶段，图输入缓冲区初始为全零：`context_lens=0`、`slot_mapping=0`、占位 `input_ids/positions` 等。

1. **Warmup 不在 capturing 流上**：`extend_attention` 走 eager 分支，`prefix_lens = context_lens - seq_len` 在 `context_lens=0`、`seq_len=K+1` 时为 **负数**，`prefix_len > 0` 不成立 → **几乎不从 KV 读 prefix**，却在 **真实 slot（全零映射时反复写同一物理槽）** 上做多步前向，与真实 verify 语义完全不一致。
2. **Capture 在 capturing 内**：走 `_extend_attention_cuda_graph_safe`，同样基于 **错误的 `context_lens`/slot 状态** 完成构图。
3. 之后每次 `replay()` 前虽在 Python 侧把 `context_lens`、`slot_mapping` 等改成正确值，但 **已捕获的算子序列** 与「在错误状态下捕获」时的行为绑在一起；与 eager 下「每步用当前 manager 状态直接前向」不等价，导致 **target verify logits 与 draft 预测系统性对不齐**，greedy 比较下首位必拒 → 接受率 **0**。

**本质一句话**：CUDA Graph 的 warmup/capture 必须在 **与生产 replay 相同的元数据** 上执行，不能用全零占位冒充一步 verify。

**修复**（`nano_dist_spec/speculative.py`）：

- 为 `_build_graph` 增加可选回调 **`fill_inputs(inp, pos, sm, bt, cl)`**：在 **两次 warmup 之前** 以及 **进入 `torch.cuda.graph` 捕获之前** 各调用一次，用与 `_run_target_forward` / `_run_draft_forward` 在 `replay` 前相同的逻辑写入真实 token、position、`compute_slot_mapping_into`、`fill_block_table_padded`、`context_lens`。
- `_get_target_graph` / `_get_draft_graph` 首次构图时 **必须** 传入该回调；`replay` 前仍保留原有的显式 fill。

**涉及文件**：`nano_dist_spec/speculative.py`（调试用 NDJSON 在确认修复后已移除，仅保留 `fill_inputs` 逻辑）。

---

## 进展 #22: `profiler/nano_benchmark.sh` 拆分 baseline / spec 并可单独执行

**目标**：脚本维护与 CI/手工跑分场景下，有时只需 target-only baseline 或只需 spec sweep，不必每次都跑满全套。

**改动**：

1. 抽出 **`run_baseline`**：执行 `bench.py basic`，写 `baseline.log`，汇总 `extract_basic_metrics`。
2. 抽出 **`run_spec_suite <start_idx> <total>`**：对 `K_VALUES` 循环执行 `bench.py spec`，进度号与总步数由调用方传入（`both` 时为 `2 .. N+1`，纯 `spec` 时为 `1 .. N`）。
3. **第一个位置参数**选择模式（缺省 `both`）：
   - `baseline` 或 **`base`**：只跑 baseline；
   - `spec`：只跑 spec；
   - `both`：先 baseline 再 spec（与历史默认行为一致）。
4. **`--help` / `-h` / help**：打印用法；未知参数报错退出。
5. `summary.log` 开场增加一行 **`Mode: ...`**，便于区分当次运行组合。

**示例**：

```bash
./profiler/nano_benchmark.sh           # 或 both
./profiler/nano_benchmark.sh baseline  # 或 base
./profiler/nano_benchmark.sh spec
```
