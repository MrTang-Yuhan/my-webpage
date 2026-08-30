---
layout: post.njk
post_id: 2026-08-05-大模型中的数学-001-gumbel-max-trick-代替随机采样
archive: 数学原理
title: 大模型中的数学（001）：Gumbel-Max Trick 代替随机采样
date: 2026-08-05
tags:
  - post
---
# Gumbel-Max Trick

> 这里不展开 Gumbel-Max 的严格数学证明，只说明它解决什么问题，以及为什么需要它。

## 1. 我们想解决什么问题？

假设有一个离散随机变量 $X$，它有三个可能取值：

$$
P(X=1)=p_1,\quad P(X=2)=p_2,\quad P(X=3)=p_3
$$

其中：

$$
p_1+p_2+p_3=1
$$

我们希望按照这个概率分布采样一个 $X$。

例如，如果：

$$
[p_1,p_2,p_3]=[0.5,0.3,0.2]
$$

那么多次采样后，三个类别大约会分别出现 50%、30% 和 20%。

问题在于：如果 $p_1,p_2,p_3$ 是神经网络输出的结果，直接进行**离散采样后，得到的类别 $X$ 通常不能直接用于反向传播。**

也就是说，采样结果虽然依赖于 $p_1,p_2,p_3$，但这个离散采样过程本身不可导，梯度无法顺利传回网络。

## 2. Gumbel-Max Trick 的核心想法

Gumbel-Max Trick 把“按照概率随机采样”改写成了下面的形式：

$$
k=\arg\max_i(\log p_i+g_i)
$$

其中：

$$
g_i=-\log(-\log u_i)),\qquad u_i\sim \text{Uniform}(0,1)
$$

这里的 $g_i$ 称为 Gumbel 噪声。

直觉上：

- $\log p_i$ 表示类别 $i$ 原本的偏好程度；
- $g_i$ 为每个类别加入一个随机扰动；
- 最终选择 **“偏好程度 + 随机扰动”** 最大的类别。

概率较大的类别通常更容易获胜，但概率较小的类别也可能因为随机噪声较大而被选中。

**一个重要结论是：**

$$
k=\arg\max_i(\log p_i+g_i)
$$

**选出的类别 $k$，恰好服从原来的类别分布 $p$。**

## 3. 为什么实际中常写成 logits？

神经网络通常输出的是 logits $a_i$，而不是概率 $p_i$。由于：

$$
p_i=\operatorname{softmax}(a_i)
$$

并且 softmax 中的归一化项对所有类别都相同，所以可以直接使用：

$$
k=\arg\max_i(a_i+g_i)
$$

因此，Gumbel-Max Trick 的常见流程是：

1. 神经网络输出 logits；
2. 生成 Gumbel 噪声；
3. 将噪声加到 logits 上；
4. 对结果取 $\argmax$；
5. 得到一个离散类别或 one-hot 向量。

对应的 one-hot 表示为：

$$
z_i=
\begin{cases}
1,& i=\arg\max_j(a_j+g_j)\\
0,& \text{otherwise}
\end{cases}
\tag{1}
$$

## 4. Gumbel-Softmax：可导的近似

Gumbel-Max Trick 尽管替代了离散采样，然而，公式 (1) 中仍然存在不可导的 $\argmax$。

**所以我们可以用可导的带温度参数 $\tau$ 的 softmax 代替 $\argmax$，即可得到最终的可导的 $z$:**

$$
\tilde z_i=
\operatorname{softmax}_j({\frac{a_i+g_i}{\tau})}=

\frac{\exp(\frac{(a_i+g_i)}{\tau})}
{\sum_j\exp(\frac{(a_j+g_j)}{\tau})}
$$

得到的 $\tilde z$ 是一个连续向量，因此可以进行反向传播。

温度参数 $\tau$ 控制输出的“尖锐程度”：

- $\tau$ 较大：输出更平滑；
- $\tau$ 较小：输出更接近 one-hot；
- 当 $\tau\to 0$ 时，结果越来越接近 $\argmax$。

因此：

- Gumbel-Max：精确的离散采样，但不可导；
- Gumbel-Softmax：可导的连续近似，但不再是严格的离散 one-hot 输出。

## 5. PyTorch 中的简单示意代码

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# 三个类别的 logits
logits = torch.tensor([2.0, 1.0, 0.0])

# 生成 Uniform(0, 1) 随机数
u = torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)

# 生成 Gumbel 噪声
gumbel_noise = -torch.log(-torch.log(u))

# Gumbel-Max：离散采样
scores = logits + gumbel_noise
sampled_class = torch.argmax(scores)

print("logits:", logits)
print("Gumbel noise:", gumbel_noise)
print("sampled class:", sampled_class.item())

# Gumbel-Softmax：可导的近似
temperature = 0.5
soft_sample = F.softmax(
    (logits + gumbel_noise) / temperature,
    dim=-1
)

print("soft sample:", soft_sample)
```

其中：

- `sampled_class` 是真正采样得到的类别；
- `soft_sample` 是接近 one-hot 的连续向量；
- `temperature` 越小，`soft_sample` 越接近 one-hot。
