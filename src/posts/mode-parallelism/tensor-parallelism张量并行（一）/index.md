---
layout: post.njk
archive: mode-parallelism
title: Tensor Parallelism张量并行（一）
date: 2026-05-13
tags:
  - post
---
论文来自 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)。

# 什么是 Tensor Parallelism (张量并行)

大模型训练有两个问题：

1. 在多卡 GPU 训练时，每个 GPU 都拷贝一份整个模型参数是不现实的。
2. 集合通信开销大。

**张量并行的核心思想就是将权重矩阵进行切分并且划分到不同的 GPU 上执行，并且通过巧妙的张量切分减少集合通信开销**。

论文中的 transformer 架构如下图 2 所示。

![](img/transformer-block.png)

对于其中的 MLP 块。
