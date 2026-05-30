---
layout: post.njk
post_id: 2026-05-25-transformer-架构-scaled-dot-product-attention-的缩放因子推导
archive: 数学原理
title: Transformer 架构：Scaled Dot-Product Attention 的缩放因子推导
date: 2026-05-25
description: Transformer 中的数值稳定性：为什么 Attention 需要除以 $\sqrt{d_k}$
tags:
  - post
---
在最初的 transformer 架构中，完整的 attention 公式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

这篇文章推导公式中的 $\sqrt{d_k}$ 是如何得到的。

值得说明一点，**这里的 $\sqrt{d_k}$ 在代码中常被称为 `norm_factor`，它的取值可能根据模型架构、attention 变体以及长度外推方法不同而有所改变。**

# 推导过程


$d_k$ 表示每个 attention head 中 query/key 向量的维度，也就是通常所说的 `head_dim`。

假设某个 query 向量和 key 向量分别为：

$$
q = [q_1, q_2, \ldots, q_{d_k}], \quad k = [k_1, k_2, \ldots, k_{d_k}]
$$

它们的点积为：

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

也就是：

$$
q \cdot k = q_1 k_1 + q_2 k_2 + \cdots + q_{d_k} k_{d_k}
$$

为了分析这个点积的数值尺度，假设 $q_i$ 和 $k_i$ 是相互独立的随机变量，并且满足：

$$
E[q_i] = 0, \quad E[k_i] = 0, \quad \text{Var}(q_i) = 1, \quad \text{Var}(k_i) = 1
$$

令：

$$
X_i = q_i k_i
$$

则：

$$
q \cdot k = \sum_{i=1}^{d_k} X_i
$$

首先计算单项 $X_i$ 的期望：

$$
E[X_i] = E[q_i k_i]
$$

由于 $q_i$ 和 $k_i$ 独立：

$$
E[q_i k_i] = E[q_i] E[k_i]
$$

又因为 $E[q_i] = 0$，$E[k_i] = 0$，所以：

$$
E[X_i] = 0
$$

接着计算 $X_i$ 的方差：

$$
\text{Var}(X_i) = E[X_i^2] - E[X_i]^2
$$

因为 $E[X_i] = 0$，所以：

$$
\text{Var}(X_i) = E[X_i^2]
$$

代入 $X_i = q_i k_i$：

$$
\text{Var}(X_i) = E[(q_i k_i)^2] = E[q_i^2 k_i^2]
$$

由于 $q_i$ 和 $k_i$ 独立：

$$
E[q_i^2 k_i^2] = E[q_i^2] E[k_i^2]
$$

又因为 $\text{Var}(q_i) = E[q_i^2] - E[q_i]^2 = 1$，且 $E[q_i] = 0$，所以 $E[q_i^2] = 1$。同理 $E[k_i^2] = 1$。

因此：

$$
\text{Var}(X_i) = E[q_i^2] E[k_i^2] = 1
$$

也就是说，每一项乘积 $q_i k_i$ 的方差约为 1。

由于点积 $q \cdot k$ 是 $d_k$ 个这样的项相加：

$$
q \cdot k = X_1 + X_2 + \cdots + X_{d_k}
$$

在各项近似独立的情况下，协方差为0，故和的方差等于方差之和：

$$
\text{Var}(q \cdot k) = \text{Var}\left(\sum_{i=1}^{d_k} X_i\right) = \sum_{i=1}^{d_k} \text{Var}(X_i) = d_k
$$

因此：

$$
\text{Var}(q \cdot k) = d_k
$$

对应的标准差为：

$$
\text{Std}(q \cdot k) = \sqrt{\text{Var}(q \cdot k)} = \sqrt{d_k}
$$

这说明，当 $d_k$ 变大时，未缩放的点积 $q \cdot k$ 的典型数值幅度会随着 $\sqrt{d_k}$ 增大。


而 attention score 后面会进入 softmax：

$$
\text{softmax}(QK^T)
$$

如果 $QK^T$ 的数值过大，softmax 会变得非常尖锐。例如：

$$
\text{softmax}([20, 1, -3, 0]) \approx [1, 0, 0, 0]
$$

这会导致 attention 分布过早饱和，梯度变小，训练变得不稳定。

因此，为了让进入 softmax 的 logits 保持在相对稳定的尺度，需要对点积结果做归一化缩放：

$$
\frac{q \cdot k}{\sqrt{d_k}}
$$

缩放之后：

$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{\text{Var}(q \cdot k)}{d_k} = \frac{d_k}{d_k} = 1
$$

对应标准差为：

$$
\text{Std}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1
$$

也就是说，除以 $\sqrt{d_k}$ 后，attention logits 的尺度大致回到 $O(1)$ 的量级，不会随着 $ \text{head\\_dim} $ 增大而持续变大。

因此，在原始 Transformer 中：

$$
\text{norm\_factor} = \sqrt{d_k}
$$

其作用是控制 $QK^T$ 的数值尺度，使 softmax 不至于过度饱和，从而提升训练稳定性。


