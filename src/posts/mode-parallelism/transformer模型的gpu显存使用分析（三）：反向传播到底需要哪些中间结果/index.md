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

这些值是用 GPT 5.5 总结的。暂时感觉没问题：

设：

| 符号 | 含义 |
|---|---|
| `B` | batch size |
| `T` | sequence length |
| `D` | hidden size |
| `h` | attention heads 数 |
| `d` | 每个 head 的维度，`d = D / h` |
| `M` | FFN 中间层维度，常见为 `4D` |
| `Vocab` | 词表大小 |

假设使用普通 eager attention，而不是 FlashAttention，也没有 activation checkpointing。


| 模块 | 前向公式 | 反向需要保存的大中间值 | 常见维度 |
|---|---|---|---|
| Embedding | $x = \mathrm{Embed}(\mathrm{tokens})$ | embedding backward 需要 $\mathrm{tokens}$；$x$ 作为后续 Transformer block 输入通常会被保存或 checkpoint 重算 | $\mathrm{tokens}: B \times T$，$x: B \times T \times D$ |
| LayerNorm 1 | $u = \mathrm{LN}_1(x)$ | LayerNorm backward 通常需要输入 $x$ 或归一化结果 $\hat{x}$，以及 $\mathrm{mean}/\mathrm{rstd}$ | $x: B \times T \times D$，$\hat{x}: B \times T \times D$，$\mathrm{mean}: B \times T$，$\mathrm{rstd}: B \times T$ |
| QKV Linear | $Q = uW_Q,\ K = uW_K,\ V = uW_V$ | Linear backward 需要输入 $u$；attention backward 需要 $Q,K,V$ 或通过 recompute 得到它们 | $u: B \times T \times D$，$Q: B \times h \times T \times d$，$K: B \times h \times T \times d$，$V: B \times h \times T \times d$ |
| Attention scores | $S = \frac{QK^\top}{\sqrt d}$ | 普通 softmax backward 通常不必须保存 $S$；某些实现可能保存 mask 后 scores；FlashAttention 类实现通常不保存完整 $S$ | $S: B \times h \times T \times T$ |
| Attention mask，可选 | $S_{\mathrm{mask}} = S + M_{\mathrm{attn}}$ | causal/padding mask 通常可由输入或规则重构，不一定作为大激活保存 | $M_{\mathrm{attn}}: 1 \times 1 \times T \times T$ 或 $B \times 1 \times 1 \times T$ 或 $B \times 1 \times T \times T$ |
| Softmax | $P = \mathrm{softmax}(S_{\mathrm{mask}})$ | 标准 eager attention 通常保存 softmax 概率 $P$，用于 softmax backward；FlashAttention 不保存完整 $P$，而保存较小统计量并在 backward 重算 | $P: B \times h \times T \times T$ |
| Attention Dropout，可选 | $\tilde P = \mathrm{Dropout}(P)=\frac{m\odot P}{1-p}$ | 严谨地说，softmax backward 需要 $P$，dropout backward 需要 mask $m$；$\tilde P$ 可由 $P,m$ 重构，也可能被实现直接保存 | $P: B \times h \times T \times T$，$m: B \times h \times T \times T$，$\tilde P: B \times h \times T \times T$ |
| Attention Value 聚合 | $O = \tilde P V$ | matmul backward 需要 $\tilde P$ 和 $V$；$\tilde P$ 可由 $P,m$ 重构；后续输出投影需要 $O$ reshape 后的 $c$ 作为输入 | $\tilde P: B \times h \times T \times T$，$V: B \times h \times T \times d$，$O: B \times h \times T \times d$，$c: B \times T \times D$ |
| Attention 输出投影 | $a = cW_O$ | Linear backward 需要输入 $c$；通常不因该 Linear 本身必须保存输出 $a$ | $c: B \times T \times D$，$a: B \times T \times D$ |
| Residual Add 1 | $x' = x + a$ | residual add 本身通常不需要保存大中间值；但 $x'$ 作为后续 LayerNorm/FFN 输入通常会被保存或重算 | $x': B \times T \times D$ |
| LayerNorm 2 | $v = \mathrm{LN}_2(x')$ | LayerNorm backward 通常需要输入 $x'$ 或归一化结果 $\hat{x}'$，以及 $\mathrm{mean}/\mathrm{rstd}$ | $x': B \times T \times D$，$\hat{x}': B \times T \times D$，$\mathrm{mean}: B \times T$，$\mathrm{rstd}: B \times T$ |
| FFN Linear 1，GELU-FFN | $z = vW_1 + b_1$ | Linear backward 需要输入 $v$；GELU backward 需要 $z$ 或等价中间值 | $v: B \times T \times D$，$z: B \times T \times M$ |
| GELU 激活 | $g = \mathrm{GELU}(z)$ | GELU backward 需要 $z$ 或等价中间值；FFN Linear 2 backward 需要输入 $g$ | $z: B \times T \times M$，$g: B \times T \times M$ |
| FFN Linear 1，SwiGLU-FFN，可选 | $z_1 = vW_{\mathrm{gate}},\ z_2 = vW_{\mathrm{up}}$ | 两个 Linear backward 需要输入 $v$；SwiGLU backward 需要 $z_1,z_2$ 或等价中间值 | $v: B \times T \times D$，$z_1: B \times T \times M$，$z_2: B \times T \times M$ |
| SwiGLU 激活，可选 | $g = \mathrm{SiLU}(z_1)\odot z_2$ | SwiGLU backward 需要 $z_1,z_2$ 或等价中间值；后续 Linear backward 需要输入 $g$ | $z_1: B \times T \times M$，$z_2: B \times T \times M$，$g: B \times T \times M$ |
| FFN Linear 2 | $y = gW_2 + b_2$ | Linear backward 需要输入 $g$；该 Linear 本身不必须保存输出 $y$ | $g: B \times T \times M$，$y: B \times T \times D$ |
| FFN Dropout，可选 | $\tilde y = \mathrm{Dropout}(y)=\frac{m_{\mathrm{ffn}}\odot y}{1-p}$ | dropout backward 需要 mask $m_{\mathrm{ffn}}$；若后续需要，可保存或重算 $y,\tilde y$ | $y: B \times T \times D$，$m_{\mathrm{ffn}}: B \times T \times D$，$\tilde y: B \times T \times D$ |
| Residual Add 2 | $\mathrm{out} = x' + \tilde y$ | residual add 本身通常不需要额外保存大中间值；$\mathrm{out}$ 作为下一层输入通常会被保存或 checkpoint 重算 | $\mathrm{out}: B \times T \times D$ |
| LM Head / Classifier | $\ell = x_{\mathrm{final}}W_{\mathrm{vocab}}$ | Linear backward 需要输入 $x_{\mathrm{final}}$；普通 cross entropy 可能保存 logits；fused cross entropy 可能不完整保存 logits | $x_{\mathrm{final}}: B \times T \times D$，$\ell: B \times T \times \mathrm{Vocab}$ |
