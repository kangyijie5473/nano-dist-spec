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
