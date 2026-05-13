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

对于其中的 MLP 块，输出计算公式：

$$
Y = \text{GeLU}(XA)
$$

假设现在有 2 个 GPU，为了将权重 A 切分到 2 个 GPU 中，有**行切分**和**列切分**两种方式。

- **行切分**：将权重矩阵 A 进行行切分，为了计算 Y，需要同时将 输入 X 进行列切分，即：
  $$
  X = [X_1, X_2], \quad A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}. \tag{2}
  $$
  此时 GPU0 保存 $X_1$ 和 $A_1$, GPU1 保存 $X_2$ 和 $A_2$。

  这个划分导致 $Y = \text{GeLU}(X_1A_1 + X_2A_2)$，由于 $\text{GeLU}$ 是非线性激活函数，故 $\text{GeLU}(X_1A_1 + X_2A_2)\neq \text{GeLU}(X_1A_1) +\text{GeLU}(X_2A_2)$。所以为了计算 $\text{GeLU}$，此时不得不采用 All-Reduce。

- **列切分**：为了解决行切分的弊端，故引入列切分。将权重矩阵 A 进行列切分，此时$A = [A_1, A_2]$，此时可直接在每个 GPU 上单独计算一部分激活：

  $$
  Y = [Y_1, Y_2] = [\text{GeLU}(XA_1), \text{GeLU}(XA_2)] \tag{3}  
  $$

  从而避免了此时采用集合通信。

# 张量并行应用于 MLP 和 Attention



