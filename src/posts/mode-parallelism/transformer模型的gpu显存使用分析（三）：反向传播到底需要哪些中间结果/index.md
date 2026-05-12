---
layout: post.njk
archive: mode-parallelism
title: transformer模型的GPU显存使用分析（三）：反向传播到底需要哪些中间结果
date: 2026-05-12
description: 本文旨在梳理训练阶段前向传播中必须保留的中间结果（如激活值、掩码、统计量等），以正确支持反向传播
tags:
  - post
  - GPU memory usage
  - transformer
  - model parallelism
---
# 基本原则

**核心原则：反向传播时，某个梯度公式如果要用到前向里的某个“中间值”，这个“中间值”就要暂存。**

以线性层举例。对于

$$
y = xW
$$

反向传播：

$$
\frac{\partial L}{\partial W} = x^\top \frac{\partial L}{\partial y}
$$

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} W^\top
$$

因此为了算 $\partial L / \partial W $，需要保存输入 $x$[^1]。

[^1]: 为了算 $\partial L / \partial x $，也需要保存权重 $W$。但是权重是本来就需要保存的，不算“中间值”。

## transformer 架构中常存“中间值”汇总

设：

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `T` | sequence length |
| `D` | hidden size / model dimension |
| `h` | attention heads 数 |
| `d` | 每个 head 的维度，`d = D / h` |
| `M` | FFN 中间层维度，常见为 `4D` |
| `Vocab` | 词表大小 |

假设使用普通 eager attention，而不是 FlashAttention，也没有 activation checkpointing。


| 模块 | 前向公式 | 反向需要保存的大中间值 | 常见维度 | 显存压力 |
|---|---|---|---|---|
| Embedding | $x = \mathrm{Embed}(\mathrm{tokens})$ | embedding 输出 $x$ | $B \times T \times D$ | 中等 |
| LayerNorm 1 | $u = \mathrm{LN}_1(x)$ | 输入 $x$ 或归一化结果 $\hat{x}$ | $x: B \times T \times D$ | 中等 |
| QKV Linear | $Q = uW_Q,\ K = uW_K,\ V = uW_V$ | 输入 $u$，以及输出 $Q,K,V$ | $u: B \times T \times D$，$Q,K,V: B \times h \times T \times d$ | 较大 |
| Attention scores | $S = \frac{QK^\top}{\sqrt{d}}$ | 通常不一定保存 $S$，但可能保存 mask 后的 scores | $B \times h \times T \times T$ | 很大 |
| Softmax | $P = \mathrm{softmax}(S)$ | softmax 概率 $P$ | $B \times h \times T \times T$ | 极大 |
| Attention Dropout | $\tilde{P} = \mathrm{Dropout}(P)$ | dropout mask，或 dropout 后的 $\tilde{P}$ | $B \times h \times T \times T$ | 极大 |
| Attention Value 聚合 | $O = \tilde{P}V$ | $\tilde{P}$，$V$，有时保存 $O$ | $O: B \times h \times T \times d$ | 较大 |
| Attention 输出投影 | $a = cW_O$ | 输入 $c = \mathrm{concat}(O)$ | $B \times T \times D$ | 中等 |
| Residual Add 1 | $x' = x + a$ | 通常不需要额外保存，但 $x'$ 会被后续 LN/FFN 用到 | $B \times T \times D$ | 中等 |
| LayerNorm 2 | $v = \mathrm{LN}_2(x')$ | 输入 $x'$ 或 $\hat{x}'$，以及 $\mathrm{mean}/\mathrm{rstd}$ | $B \times T \times D$ | 中等 |
| FFN Linear 1 | $z = vW_1 + b_1$ | 输入 $v$，输出 $z$ | $v: B \times T \times D$，$z: B \times T \times M$ | 很大 |
| GELU/SwiGLU 激活 | $g = \mathrm{GELU}(z)$ | $z$，有时也保存 $g$ | $B \times T \times M$ | 很大 |
| FFN Linear 2 | $y = gW_2 + b_2$ | 输入 $g$ | $B \times T \times M$ | 很大 |
| FFN Dropout | $\tilde{y} = \mathrm{Dropout}(y)$ | dropout mask | $B \times T \times D$ | 中等 |
| Residual Add 2 | $\mathrm{out} = x' + \tilde{y}$ | 通常不额外保存，但输出要给下一层 | $B \times T \times D$ | 中等 |
| LM Head / Classifier | $\ell = xW_{\mathrm{vocab}}$ | 输入 $x$，有时保存 logits | $\mathrm{logits}: B \times T \times \mathrm{Vocab}$ | 可能极大 |
---
