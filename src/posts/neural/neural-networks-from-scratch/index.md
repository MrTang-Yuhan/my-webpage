---
layout: post.njk
title: "深度学习：注意力机制入门"
date: 2026-05-06
description: "理解注意力机制如何让深度学习模型能够'关注'输入的相关部分。"
tags:
  - post
  - deep learning
  - attention
  - neural networks
---

近年来，注意力机制彻底改变了深度学习领域。从自然语言处理到计算机视觉，这种简单而强大的思想成为了现代 AI 突破的核心。<sup class="footnote-ref"><a href="#fn1">[1]</a></sup>

## 序列到序列的问题

在注意力机制出现之前，翻译这样的序列到序列任务面临一个根本问题：如何处理输入和输出长度不同的情况？

传统的编码器-解码器架构会将整个输入序列压缩成一个固定大小的向量，然后从中"解码"出目标序列。<sup class="footnote-ref"><a href="#fn2">[2]</a></sup>

```python
class EncoderDecoder:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.encoder = RNN(input_dim, hidden_dim)
        self.decoder = RNN(hidden_dim, output_dim)

    def forward(self, input_seq, target_seq):
        # 编码整个序列
        context = self.encoder(input_seq)

        # 从单个上下文向量解码
        outputs = []
        hidden = context
        for i in range(len(target_seq)):
            hidden = self.decoder(target_seq[i], hidden)
            outputs.append(hidden)
        return outputs
```

<aside id="fn1" class="footnote">
  <p>2017年 Vaswani 等人发表的《Attention Is All You Need》论文介绍了 Transformer 架构，完全基于注意力机制，摒弃了循环神经网络。</p>
</aside>

![注意力机制图示](https://picsum.photos/seed/attention/800/400)

### aaaasdasd

asdasdasd

## 注意力机制的核心思想

注意力机制允许解码器在生成每个输出时，"关注"输入的不同部分。类似于人类翻译时会回头查看原文的相关部分。<sup class="footnote-ref"><a href="#fn3">[3]</a></sup>

```python
def attention(query, keys, values):
    """
    注意力机制的核心计算

    query: 当前解码器状态
    keys: 所有编码器状态的"键"
    values: 对应的"值"
    """
    # 计算 query 和每个 key 的相似度
    scores = np.dot(query, keys.T) / np.sqrt(keys.shape[-1])

    # softmax 得到注意力权重
    weights = softmax(scores, axis=-1)

    # 加权求和 values
    context = np.dot(weights, values)

    return context, weights
```

<aside id="fn2" class="footnote">
  <p>这种"瓶颈"问题意味着模型必须将所有信息压缩到固定大小的向量中，对于长序列尤其困难。</p>
</aside>

<aside id="fn3" class="footnote">
  <p>注意力权重可以可视化为热力图，显示模型在生成每个输出时关注输入的哪些部分。</p>
</aside>

## 多头注意力

实际应用中，通常使用多个注意力头，每个头学习不同类型的依赖关系：<sup class="footnote-ref"><a href="#fn4">[4]</a></sup>

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)

    def split_heads(self, x):
        # 分割成多个头
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, keys, values):
        batch_size = query.shape[0]

        # 线性变换并分割成多头
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(keys))
        V = self.split_heads(self.W_v(values))

        # 计算注意力
        context, _ = attention(Q, K, V)

        # 合并多头并输出
        context = context.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(context)
```

<aside id="fn4" class="footnote">
  <p>不同的注意力头可以学习关注不同类型的关系，比如语法依赖、语义相似性或位置关系。</p>
</aside>

## 为何注意力机制如此有效？

1. **并行化** — 不需要逐时间步处理，可以并行计算整个序列
2. **长距离依赖** — 直接连接解决了梯度消失问题
3. **可解释性** — 注意力权重提供了模型决策的洞察
4. **通用性** — 适用于文本、图像、音频等多种模态

## 结论

注意力机制代表了深度学习的一个范式转变。它简洁优雅，却解决了困扰循环模型多年的根本问题。理解注意力是理解现代 AI 系统的关键。

下次你使用翻译工具或与 AI 对话时，想想那些优雅的矩阵运算——它们正是你输入的每个词被"关注"的方式。
