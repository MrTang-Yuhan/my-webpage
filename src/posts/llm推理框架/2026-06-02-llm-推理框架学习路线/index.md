---
layout: post.njk
post_id: 2026-06-02-llm-推理框架学习路线
archive: LLM推理框架
title: LLM 推理框架学习路线
date: 2026-06-02
tags:
  - post
---
> **以下内容为 AI 生成**

下面我不再给“30天强行冲刺”的安排，而是按你要的顺序，整理一份**LLM 推理框架研究地图**：  
**发展脉络 → 综述/论文 → 代码资源**。  
重点面向后续做 **LLM inference serving / AI Infra / Systems for ML** 方向科研。

---

# （1）LLM 推理框架发展的脉络

LLM 推理框架的发展，本质上是在解决几个核心矛盾：

1. **模型越来越大，单卡放不下**
2. **上下文越来越长，KV Cache 显存压力巨大**
3. **用户请求长度不一致，GPU 利用率低**
4. **Decode 阶段逐 token 生成，计算并行度差**
5. **线上服务既要低延迟，又要高吞吐**

可以按下面这条主线理解。

---

## 1. 第一阶段：传统深度学习推理框架

早期 Transformer / BERT / GPT 推理主要依赖：

- PyTorch eager mode
- TensorFlow Serving
- ONNX Runtime
- TensorRT
- Triton Inference Server

这些框架擅长优化普通深度学习模型，但面对 LLM 自回归生成时，有几个问题：

- 不能很好处理动态长度输入
- 对 KV Cache 管理不够细
- Batch 内请求长度差异导致 padding 浪费
- Decode 阶段 batch 很小、矩阵形状扁平，GPU 利用率不高

代表项目：

| 框架 | 作用 | 链接 |
|---|---|---|
| NVIDIA TensorRT | 通用高性能推理优化器 | https://developer.nvidia.com/tensorrt |
| NVIDIA Triton Inference Server | 通用推理服务框架 | https://github.com/triton-inference-server/server |
| ONNX Runtime | 跨平台推理框架 | https://github.com/microsoft/onnxruntime |

---

## 2. 第二阶段：Transformer 专用推理优化库

LLM 兴起前后，业界开始针对 Transformer 做专门优化。

代表是 NVIDIA **FasterTransformer**。它用 C++/CUDA 实现了高性能 Transformer kernels，支持 BERT、GPT、T5 等模型。后来 NVIDIA 官方也说明 FasterTransformer 的开发已转向 TensorRT-LLM。[^1]

| 框架 | 特点 | 链接 |
|---|---|---|
| FasterTransformer | C++/CUDA Transformer 推理优化库 | https://github.com/NVIDIA/FasterTransformer |
| DeepSpeed-Inference | 微软分布式推理优化 | https://github.com/microsoft/DeepSpeed |
| TurboTransformers | 腾讯 Transformer 推理优化库 | https://github.com/Tencent/TurboTransformers |

这一阶段的关键词是：

- kernel fusion
- tensor parallelism
- pipeline parallelism
- FP16 / INT8
- GEMM 优化
- Faster attention kernel

但它们还没有很好解决 LLM serving 中最重要的系统问题：  
**动态请求调度 + KV Cache 显存管理 + 高并发服务吞吐。**

---

## 3. 第三阶段：连续批处理与 LLM Serving 系统

LLM 线上服务和普通模型推理不同。

传统推理通常是：

```text
收集一批请求 → 一起 forward → 返回结果
```

但 LLM 是逐 token 生成：

```text
prefill prompt → decode token1 → decode token2 → decode token3 → ...
```

不同用户请求的 prompt 长度、生成长度都不同。如果用普通 batch，很容易出现：

- 短请求等长请求
- batch 中大量 padding
- GPU 空闲
- 服务延迟不稳定

因此出现了 **continuous batching / iteration-level scheduling**。

代表性系统是 **Orca**。

| 技术 | 解决问题 | 论文 / 项目 |
|---|---|---|
| Iteration-level scheduling | 每生成一步动态调整 batch | https://www.usenix.org/conference/osdi22/presentation/yu |
| Continuous batching | 动态插入新请求，提高 GPU 利用率 | https://github.com/huggingface/text-generation-inference |

这一步奠定了现代 LLM 推理框架的服务形态。

---

## 4. 第四阶段：KV Cache 成为核心瓶颈，PagedAttention 出现

LLM 推理中，KV Cache 是最大瓶颈之一。

每个请求都会保存每一层 attention 的 Key / Value：

```text
KV Cache size ≈ batch_size × sequence_length × num_layers × num_heads × head_dim × 2
```

当：

- batch 大
- 上下文长
- 模型层数多
- 并发用户多

KV Cache 会占用大量显存。

传统做法通常为每个请求预分配连续显存，问题是：

- 内存碎片严重
- 预分配浪费
- beam search / parallel sampling 下 KV 共享困难

**vLLM / PagedAttention** 的核心贡献是：  
把 KV Cache 管理成类似操作系统虚拟内存的分页结构。

也就是：

```text
逻辑 token block → 物理 KV block
```

这样可以：

- 非连续分配 KV Cache
- 减少显存碎片
- 支持请求间 KV 共享
- 提升吞吐

PagedAttention 是 vLLM 的核心技术，论文发表在 SOSP 2023。[^2]

| 框架 | 核心机制 | 链接 |
|---|---|---|
| vLLM | PagedAttention + continuous batching | https://github.com/vllm-project/vllm |
| vLLM 文档 | 官方使用与架构说明 | https://docs.vllm.ai/ |
| PagedAttention 论文 | SOSP 2023 | https://dl.acm.org/doi/10.1145/3600006.3613165 |

---

## 5. 第五阶段：Attention Kernel 优化，FlashAttention 系列成为基础设施

LLM 推理中，Attention 既消耗算力，也消耗显存带宽。

普通 attention 会显式构造：

```text
QK^T attention matrix
```

长序列时开销巨大。

**FlashAttention** 的关键思想是：

- 不显式 materialize 完整 attention matrix
- 分块计算 attention
- 利用 GPU SRAM / shared memory
- 减少 HBM 读写
- 保持数值稳定的 online softmax

代表论文：

| 技术 | 作用 | 链接 |
|---|---|---|
| FlashAttention | IO-aware exact attention | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2 | 更好利用 GPU 并行度 | https://arxiv.org/abs/2307.08691 |
| FlashAttention GitHub | 官方代码 | https://github.com/Dao-AILab/flash-attention |

对推理框架来说，FlashAttention 主要影响：

- prefill 阶段长 prompt attention
- 长上下文推理
- 多模态长序列输入
- 高性能 attention kernel 实现

---

## 6. 第六阶段：TensorRT-LLM / TGI / LMDeploy 等生产级框架成熟

LLM 商业化后，推理框架开始分化为几类：

| 类型 | 代表框架 | 特点 |
|---|---|---|
| 高吞吐研究/工程框架 | vLLM | PagedAttention，易用，适合研究 |
| NVIDIA 生态生产框架 | TensorRT-LLM | TensorRT 编译优化，高性能 kernel，多 GPU 通信 |
| HuggingFace 生产服务 | TGI | Rust + Python，稳定服务，生态好 |
| 国内工程框架 | LMDeploy | TurboMind，量化，多卡部署 |
| 编程模型 + 推理 runtime | SGLang | 面向复杂 LLM 程序、agent、structured generation |
| 本地轻量推理 | llama.cpp | CPU / Metal / CUDA，量化友好 |

NVIDIA 官方介绍 TensorRT-LLM 是面向 NVIDIA GPU 的开源 LLM 推理优化库，包含优化 kernels、多 GPU / 多节点通信 primitives，并提供 Python API。[^3]

| 框架 | 链接 |
|---|---|
| TensorRT-LLM | https://github.com/NVIDIA/TensorRT-LLM |
| TensorRT-LLM 官方介绍 | https://developer.nvidia.com/tensorrt-llm |
| HuggingFace TGI | https://github.com/huggingface/text-generation-inference |
| LMDeploy | https://github.com/InternLM/lmdeploy |
| SGLang | https://github.com/sgl-project/sglang |
| llama.cpp | https://github.com/ggerganov/llama.cpp |

---

## 7. 第七阶段：Decode 加速与 Speculative Decoding

LLM decode 阶段有一个天然问题：

```text
每次只能生成下一个 token
```

这导致并行度差。

**Speculative Decoding** 的核心思想是：

1. 用一个小模型 draft model 快速猜多个 token
2. 用大模型 target model 一次性验证这些 token
3. 如果猜对，就一次接受多个 token
4. 如果猜错，从错误位置重新采样

这样能减少大模型 forward 次数。

NVIDIA TensorRT-LLM 已支持 speculative decoding，并在官方博客中报告可显著提升总 token 吞吐。[^4]

| 技术 | 作用 | 链接 |
|---|---|---|
| Speculative Decoding | 用小模型草稿 + 大模型验证 | https://arxiv.org/abs/2211.17192 |
| Fast Inference from Transformers via Speculative Decoding | 经典 speculative decoding 论文 | https://arxiv.org/abs/2211.17192 |
| TensorRT-LLM Speculative Decoding | 工业实现说明 | https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/ |
| Medusa | 多头预测加速 decode | https://arxiv.org/abs/2401.10774 |
| Lookahead Decoding | 无 draft model 的并行 decode | https://arxiv.org/abs/2402.02057 |

---

## 8. 第八阶段：长上下文、MoE、多机多卡、异构推理

目前比较前沿的方向包括：

| 方向 | 典型问题 |
|---|---|
| 长上下文推理 | 1M tokens context，KV Cache 放不下 |
| KV Cache 压缩 | 降低显存占用 |
| Prefix Cache / Prompt Cache | 多请求共享 prefix |
| MoE 推理 | expert routing、expert parallel、load balance |
| 多机多卡推理 | tensor parallel、pipeline parallel、expert parallel |
| 异构推理 | GPU + CPU + SSD offloading |
| structured generation | JSON / grammar constrained decoding |
| multimodal inference | 图像 token / 视频 token 造成更大上下文压力 |

代表框架：

| 框架 / 项目 | 方向 | 链接 |
|---|---|---|
| SGLang | LLM 程序执行、structured generation、高性能 runtime | https://github.com/sgl-project/sglang |
| DeepSpeed-FastGen | 长 prompt 下的高吞吐文本生成 | https://arxiv.org/abs/2401.08671 |
| TensorRT-LLM | NVIDIA 高性能生产推理 | https://github.com/NVIDIA/TensorRT-LLM |
| DeepSpeed-MII | 微软模型推理服务 | https://github.com/microsoft/DeepSpeed-MII |

---

# （2）LLM 推理框架对应的综述和论文

我建议你把论文分成 **综述类、系统类、kernel 类、调度类、decode 加速类、长上下文类、量化类** 来读。

---

## A. 综述类：先建立全局地图

| 主题 | 名称 | 链接 | 建议 |
|---|---|---|---|
| Efficient Transformers 综述 | Efficient Transformers: A Survey | https://arxiv.org/abs/2009.06732 | 了解 attention 优化大背景 |
| LLM 高效推理综述 | A Survey on Efficient Inference for Large Language Models | https://arxiv.org/abs/2404.14294 | 直接对应 LLM inference |
| LLM 推理系统综述 | LLM Inference Unveiled: Survey and Roofline Model Insights | https://arxiv.org/abs/2402.16363 | 适合系统研究视角 |
| Serving systems 综述线索 | NVIDIA LLM inference benchmarking | https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/ | 工程指标、benchmark 入门 |

---

## B. LLM Serving / 推理系统核心论文

| 论文 | 会议 / 年份 | 核心贡献 | 链接 |
|---|---|---|---|
| Orca: A Distributed Serving System for Transformer-Based Generative Models | OSDI 2022 | iteration-level scheduling，continuous batching | https://www.usenix.org/conference/osdi22/presentation/yu |
| Efficient Memory Management for Large Language Model Serving with PagedAttention | SOSP 2023 | PagedAttention，KV Cache 分页管理 | https://dl.acm.org/doi/10.1145/3600006.3613165 |
| DeepSpeed-FastGen | 2024 | Dynamic SplitFuse，优化长 prompt 服务吞吐 | https://arxiv.org/abs/2401.08671 |
| Sarathi-Serve | 2024 | chunked prefill，混合 prefill/decode 调度 | https://arxiv.org/abs/2403.02310 |
| FlexGen | ICML 2023 | GPU/CPU/SSD offloading，低资源推理大模型 | https://arxiv.org/abs/2303.06865 |

---

## C. Attention / Kernel 优化论文

| 论文 | 核心贡献 | 链接 |
|---|---|---|
| FlashAttention | IO-aware exact attention，减少 HBM 访问 | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2 | 更高 GPU 利用率，更优并行划分 | https://arxiv.org/abs/2307.08691 |
| FlashDecoding++ | 针对 LLM decoding 的 GPU kernel 优化 | https://arxiv.org/abs/2311.01282 |
| xFormers | Facebook attention / transformer kernel 工具库 | https://github.com/facebookresearch/xformers |

---

## D. Speculative Decoding / Decode 加速论文

| 论文 | 核心贡献 | 链接 |
|---|---|---|
| Fast Inference from Transformers via Speculative Decoding | draft model + target model 验证 | https://arxiv.org/abs/2211.17192 |
| SpecInfer | speculative inference serving system | https://arxiv.org/abs/2305.09781 |
| Medusa | 多个 decoding heads 预测后续 token | https://arxiv.org/abs/2401.10774 |
| Lookahead Decoding | 不依赖 draft model 的并行解码 | https://arxiv.org/abs/2402.02057 |
| PLD+ | 利用 language model artifacts 加速推理 | https://arxiv.org/abs/2412.01447 |

---

## E. KV Cache / 长上下文推理论文

| 论文 | 核心贡献 | 链接 |
|---|---|---|
| StreamingLLM | attention sink，支持 streaming long-context | https://arxiv.org/abs/2309.17453 |
| H2O: Heavy-Hitter Oracle | KV Cache eviction / 压缩 | https://arxiv.org/abs/2306.14048 |
| FastGen | adaptive KV cache compression | https://arxiv.org/abs/2310.01801 |
| LongLoRA | 长上下文扩展训练方法，也影响推理场景 | https://arxiv.org/abs/2309.12307 |
| Ring Attention | 长序列分布式 attention | https://arxiv.org/abs/2310.01889 |

---

## F. 量化与低精度推理论文

| 论文 | 核心贡献 | 链接 |
|---|---|---|
| SmoothQuant | 激活和权重量化平滑，支持 W8A8 | https://arxiv.org/abs/2211.10438 |
| GPTQ | post-training quantization for GPT | https://arxiv.org/abs/2210.17323 |
| AWQ | activation-aware weight quantization | https://arxiv.org/abs/2306.00978 |
| SpQR | 稀疏量化大模型 | https://arxiv.org/abs/2306.03078 |
| BitNet | 低比特 LLM 方向 | https://arxiv.org/abs/2310.11453 |

---

# （3）适合学习的入门代码、科研界代码和学术界代码

下面按学习路径分三层。

---

# 3.1 入门代码：适合建立直觉

这部分目标不是性能，而是帮你理解：

- Transformer forward
- KV Cache
- prefill / decode
- sampling
- attention
- simple batching

---

## A. nanoGPT

| 项目 | 链接 |
|---|---|
| nanoGPT | https://github.com/karpathy/nanoGPT |

适合看：

```text
model.py
```

你应该重点理解：

- causal self-attention
- attention mask
- block size
- autoregressive generation
- `generate()` 函数

缺点：  
它不是 LLM serving 框架，也没有复杂 KV Cache 管理。

---

## B. HuggingFace Transformers

| 项目 | 链接 |
|---|---|
| Transformers | https://github.com/huggingface/transformers |
| generation utils | https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py |

适合看：

- `generate()`
- `past_key_values`
- greedy search
- beam search
- sampling
- logits processor
- stopping criteria

这是理解 LLM 推理 API 语义最好的入口。

---

## C. llama.cpp

| 项目 | 链接 |
|---|---|
| llama.cpp | https://github.com/ggerganov/llama.cpp |

适合学习：

- 本地推理
- GGUF 模型格式
- 量化
- KV Cache
- CPU / GPU 后端
- speculative decoding 的工程实现

如果你 CUDA 不熟，llama.cpp 是非常好的入门工程，因为它相对独立，不依赖庞大的 PyTorch runtime。

---

## D. minGPT

| 项目 | 链接 |
|---|---|
| minGPT | https://github.com/karpathy/minGPT |

比 nanoGPT 更教学化，适合理解 Transformer 结构。

---

# 3.2 科研界 / 工程研究代码：适合做 AI Infra 研究

这部分是你真正应该长期精读和改造的。

---

## A. vLLM

| 项目 | 链接 |
|---|---|
| vLLM GitHub | https://github.com/vllm-project/vllm |
| vLLM 文档 | https://docs.vllm.ai/ |
| PagedAttention 论文 | https://dl.acm.org/doi/10.1145/3600006.3613165 |

建议重点看：

```text
vllm/core/
vllm/engine/
vllm/worker/
vllm/model_executor/
vllm/attention/
csrc/
```

核心模块：

| 模块 | 学什么 |
|---|---|
| scheduler | 请求调度、continuous batching |
| block_manager | KV Cache block 分配与回收 |
| attention backend | PagedAttention / FlashAttention 后端 |
| worker | 多 GPU worker 执行 |
| engine | serving 主循环 |

vLLM 是最适合做论文起点的框架之一。

适合研究方向：

- KV Cache 管理
- 长上下文调度
- speculative decoding integration
- prefix caching
- 多租户 serving
- prefill/decode disaggregation
- heterogenous serving

---

## B. SGLang

| 项目 | 链接 |
|---|---|
| SGLang | https://github.com/sgl-project/sglang |
| SGLang 文档 | https://docs.sglang.ai/ |

适合学习：

- LLM 程序执行
- structured generation
- regex / JSON constrained decoding
- RadixAttention
- prefix cache
- 高性能 runtime

如果你对 **agent / RAG / structured output / 多轮复杂调用** 感兴趣，SGLang 比 vLLM 更适合作为研究入口。

---

## C. TensorRT-LLM

| 项目 | 链接 |
|---|---|
| TensorRT-LLM GitHub | https://github.com/NVIDIA/TensorRT-LLM |
| TensorRT-LLM 官方页面 | https://developer.nvidia.com/tensorrt-llm |
| TensorRT-LLM speculative decoding blog | https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/ |

适合学习：

- NVIDIA GPU 上极致推理优化
- TensorRT 编译优化
- fused kernels
- multi-GPU communication
- speculative decoding
- FP8 / INT8 / INT4
- production deployment

缺点：  
代码复杂度较高，不建议作为第一个精读项目。

---

## D. HuggingFace Text Generation Inference, TGI

| 项目 | 链接 |
|---|---|
| TGI | https://github.com/huggingface/text-generation-inference |

适合学习：

- 生产级 LLM serving
- Rust + Python 架构
- streaming response
- tensor parallel
- continuous batching
- metrics / tracing / deployment

如果你关心生产部署和服务可靠性，TGI 值得看。

---

## E. LMDeploy

| 项目 | 链接 |
|---|---|
| LMDeploy | https://github.com/InternLM/lmdeploy |
| LMDeploy 文档 | https://lmdeploy.readthedocs.io/ |

适合学习：

- TurboMind
- C++/CUDA 推理引擎
- INT4 / INT8 量化
- tensor parallel
- OpenAI-compatible server

---

## F. DeepSpeed-MII / DeepSpeed-FastGen

| 项目 / 论文 | 链接 |
|---|---|
| DeepSpeed | https://github.com/microsoft/DeepSpeed |
| DeepSpeed-MII | https://github.com/microsoft/DeepSpeed-MII |
| DeepSpeed-FastGen 论文 | https://arxiv.org/abs/2401.08671 |

适合学习：

- 分布式推理
- 长 prompt serving
- Dynamic SplitFuse
- 和 DeepSpeed 训练生态的衔接

---

# 3.3 学术界代码：适合复现论文和找研究问题

下面这些项目更贴近论文实现或学术 prototype。

---

## A. FlexGen

| 项目 | 链接 |
|---|---|
| FlexGen GitHub | https://github.com/FMInference/FlexGen |
| FlexGen 论文 | https://arxiv.org/abs/2303.06865 |

适合学习：

- GPU / CPU / SSD offloading
- 资源受限环境下推理
- inference cost model
- batch scheduling

这是非常典型的系统论文代码。

---

## B. FlashAttention

| 项目 | 链接 |
|---|---|
| FlashAttention GitHub | https://github.com/Dao-AILab/flash-attention |
| FlashAttention 论文 | https://arxiv.org/abs/2205.14135 |
| FlashAttention-2 论文 | https://arxiv.org/abs/2307.08691 |

适合学习：

- CUDA kernel
- tiling
- shared memory
- IO complexity
- online softmax
- attention backward / forward 优化

如果你要做 kernel 方向，这是必读代码。

---

## C. xFormers

| 项目 | 链接 |
|---|---|
| xFormers | https://github.com/facebookresearch/xformers |

适合学习：

- 多种 attention kernel
- memory efficient attention
- PyTorch extension
- attention backend 设计

---

## D. Speculative Decoding / Medusa

| 项目 / 论文 | 链接 |
|---|---|
| Speculative Decoding 论文 | https://arxiv.org/abs/2211.17192 |
| Medusa GitHub | https://github.com/FasterDecoding/Medusa |
| Medusa 论文 | https://arxiv.org/abs/2401.10774 |

适合学习：

- draft / verify 机制
- tree-based decoding
- 多 token prediction
- 与 serving 框架结合的难点

---

## E. StreamingLLM

| 项目 / 论文 | 链接 |
|---|---|
| StreamingLLM GitHub | https://github.com/mit-han-lab/streaming-llm |
| StreamingLLM 论文 | https://arxiv.org/abs/2309.17453 |

适合学习：

- 长上下文推理
- attention sink
- KV Cache eviction
- streaming generation

---

## F. H2O

| 项目 / 论文 | 链接 |
|---|---|
| H2O GitHub | https://github.com/FMInference/H2O |
| H2O 论文 | https://arxiv.org/abs/2306.14048 |

适合学习：

- KV Cache pruning
- heavy hitter tokens
- 长上下文 memory reduction

---

## G. AWQ / GPTQ / SmoothQuant

| 项目 / 论文 | 链接 |
|---|---|
| AWQ GitHub | https://github.com/mit-han-lab/llm-awq |
| AWQ 论文 | https://arxiv.org/abs/2306.00978 |
| GPTQ 论文 | https://arxiv.org/abs/2210.17323 |
| AutoGPTQ | https://github.com/AutoGPTQ/AutoGPTQ |
| SmoothQuant GitHub | https://github.com/mit-han-lab/smoothquant |
| SmoothQuant 论文 | https://arxiv.org/abs/2211.10438 |

适合学习：

- post-training quantization
- weight-only quantization
- W8A8
- W4A16
- 量化和 serving 框架结合

---

# 推荐你实际阅读 / 学习顺序

如果你现在 PyTorch 分布式和 CUDA 都不熟，我建议不要从 TensorRT-LLM 或 FlashAttention CUDA kernel 开始。

更合理的顺序是：

## 第一层：理解 LLM 推理语义

1. nanoGPT  
   https://github.com/karpathy/nanoGPT

2. HuggingFace generation  
   https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py

3. llama.cpp  
   https://github.com/ggerganov/llama.cpp

---

## 第二层：理解 LLM serving 系统

1. Orca  
   https://www.usenix.org/conference/osdi22/presentation/yu

2. vLLM / PagedAttention  
   https://github.com/vllm-project/vllm  
   https://dl.acm.org/doi/10.1145/3600006.3613165

3. TGI  
   https://github.com/huggingface/text-generation-inference

4. SGLang  
   https://github.com/sgl-project/sglang

---

## 第三层：理解 kernel / 底层优化

1. FlashAttention  
   https://github.com/Dao-AILab/flash-attention  
   https://arxiv.org/abs/2205.14135

2. xFormers  
   https://github.com/facebookresearch/xformers

3. FasterTransformer  
   https://github.com/NVIDIA/FasterTransformer

4. TensorRT-LLM  
   https://github.com/NVIDIA/TensorRT-LLM

---

## 第四层：寻找科研问题

可以优先围绕这些方向找问题：

| 方向 | 为什么值得做 |
|---|---|
| KV Cache 管理 | 显存瓶颈长期存在 |
| Prefix caching | 多用户共享上下文，实际服务价值大 |
| Prefill / Decode 分离 | 工业界和学术界都在关注 |
| Speculative decoding serving | 单请求有效，但系统层调度仍有很多问题 |
| 长上下文推理 | 1M context 下系统瓶颈巨大 |
| MoE serving | expert routing + 通信 + batch 不均衡 |
| 异构推理 | GPU 昂贵，CPU/SSD offload 仍有价值 |
| 多租户 LLM serving | 真实云服务场景重要 |
| Structured generation serving | JSON / function calling 越来越重要 |

---

# 最后总结

如果你想做 **LLM 推理框架方向科研**，建议把主线抓成一句话：

> LLM 推理框架的发展，是从“单模型高性能 kernel 优化”，逐渐走向“面向动态请求、长上下文、KV Cache、分布式、多租户和复杂生成程序的系统优化”。

最重要的三篇起点论文是：

1. **Orca**：理解 continuous batching  
   https://www.usenix.org/conference/osdi22/presentation/yu

2. **PagedAttention / vLLM**：理解 KV Cache 系统化管理  
   https://dl.acm.org/doi/10.1145/3600006.3613165

3. **FlashAttention**：理解 attention kernel 为什么是 IO-bound 问题  
   https://arxiv.org/abs/2205.14135

最适合你开始精读的三个代码库是：

1. **nanoGPT**：理解模型生成逻辑  
   https://github.com/karpathy/nanoGPT

2. **vLLM**：理解现代 LLM serving 框架  
   https://github.com/vllm-project/vllm

3. **SGLang**：理解新一代 LLM runtime 和复杂生成服务  
   https://github.com/sgl-project/sglang

---

## 参考链接

[^1]: [NVIDIA FasterTransformer GitHub](https://github.com/NVIDIA/FasterTransformer/)  
[^2]: [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://dl.acm.org/doi/10.1145/3600006.3613165)  
[^3]: [NVIDIA TensorRT-LLM 官方介绍](https://developer.nvidia.com/tensorrt-llm)  
[^4]: [TensorRT-LLM Speculative Decoding Boosts Inference Throughput by up to 3.6x](https://developer.nvidia.com/blog/tensorrt-llm-speculative-decoding-boosts-inference-throughput-by-up-to-3-6x/)
