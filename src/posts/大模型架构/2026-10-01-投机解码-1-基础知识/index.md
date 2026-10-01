---
layout: post.njk
post_id: 2026-10-01-投机解码-1-基础知识
archive: 大模型架构
title: 投机解码 (1)：拒绝采样机制
date: 2026-10-01
updated: 2026-10-02
tags:
  - post
---
## 1. 问题设定与动机

自回归语言模型的逐 token 生成是串行的：生成第 $t$ 个 token 需要先完成前 $t-1$ 个 token 的前向计算。目标模型（target model）$p$ 虽然质量高，但单次前向延迟 $\tau_p$ 较大；草稿模型（draft model）$q$ 前向延迟 $\tau_q \ll \tau_p$，但分布与 $p$ 存在偏差。

Speculative Sampling（又称 Speculative Decoding）的核心目标是：**利用快速的草稿模型生成候选 token 序列，再由目标模型一次性并行验证，最终输出的分布与单独运行目标模型完全一致（lossless）**。这里的“一致”是严格的分布意义：对于任意输出序列 $x_{1:T}$，有 $\mathbb{P}^{\text{SpecDecoding}}(x_{1:T}) = p(x_{1:T})$。

---

## 2. 单步 Speculative Sampling 的完整数学推导

先处理最简单的情形：给定一个上下文，草稿模型 $q$ 生成一个候选 token $\tilde{x}$，目标模型 $p$ 需要决定是否接受它，或在拒绝时重新采样。整个过程必须使得最终输出的 token $X$ 满足 $X \sim p$。

### 2.1 算法描述

**单步 Speculative Sampling** 的生成过程如下：

1. 从草稿分布采样 $\tilde{x} \sim q(\cdot)$。

2. 以概率
$$
\alpha(\tilde{x}) = \min\!\left(1,\; \frac{p(\tilde{x})}{q(\tilde{x})}\right)
$$
接受该 token。

3. 若被拒绝，则从残差分布（residual distribution）
$$
\mu(x) = \frac{\max(0,\; p(x) - q(x))}{\sum_{x'} \max(0,\; p(x') - q(x'))}
$$
中重新采样一个 token 并输出。这里的分母是指当前单步推理时，词表中所有可能出现的 token 的集合。

### 2.2 接受路径的概率质量

对于任意 token $x$，它通过“接受路径”被输出的概率为：

$$
P(\text{accept}, X = x) = q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right)
$$

分两种情形化简：

- 若 $p(x) \geq q(x)$，则 $\min(1, p(x)/q(x)) = 1$，于是
  $$
  q(x) \cdot 1 = q(x) = \min(q(x), p(x))
  $$
- 若 $p(x) < q(x)$，则 $\min(1, p(x)/q(x)) = p(x)/q(x)$，于是
  $$
  q(x) \cdot \frac{p(x)}{q(x)} = p(x) = \min(q(x), p(x))
  $$

因此恒有：

$$
\boxed{P(\text{accept}, X = x) = \min(q(x), p(x))}
$$

### 2.3 拒绝概率与残差分布

首先计算总拒绝概率。对所有可能的草稿采样求和：

$$
\begin{aligned}
P(\text{reject})
&= \sum_{\tilde{x}} q(\tilde{x}) \left(1 - \min\!\left(1, \frac{p(\tilde{x})}{q(\tilde{x})}\right)\right) \\
&= \sum_{\tilde{x}} \left(q(\tilde{x}) - \min(q(\tilde{x}), p(\tilde{x}))\right) \\
&= \sum_{\tilde{x}} \max(0, q(\tilde{x}) - p(\tilde{x}))
\end{aligned}
$$

记 $\beta = \sum_x \max(0, q(x) - p(x))$，这就是总拒绝概率。而且能够明显的看出，**拒绝概率来源于草稿模型 $q$ 比目标模型 $p$ 多出来的那部分概率质量。**

残差分布的归一化常数恰好也是 $\beta$：因为 $\sum_x \max(0, p(x) - q(x)) = \sum_x \max(0, q(x) - p(x)) = \beta$，这来自于 $\sum_x p(x) = \sum_x q(x) = 1$ 的事实。

> **残差分布的归一化常数恰好也是 $\beta$ 推导过程**：
>
> 对于任意两个实数 $a$ 和 $b$，都有：
> $$
> a - b = \max(0, a - b) - \max(0, b - a)
> $$
> 现在我们将 $a$ 替换为 $p(x)$，将 $b$ 替换为 $q(x)$，对等式两边同时遍历词表中的所有 token $x$ 进行求和（$\sum_x$） 即可证毕。





### 2.4 拒绝路径的概率质量

若拒绝发生，重新采样的 token 服从 $\mu$。因此通过“拒绝路径”输出 token $x$ 的概率为：

$$
\begin{aligned}
P(\text{reject}, X = x)
&= \beta \cdot \mu(x) \\
&= \beta \cdot \frac{\max(0, p(x) - q(x))}{\beta} \\
&= \max(0, p(x) - q(x))
\end{aligned}
$$

### 2.5 总概率质量：精确等于目标分布

最终输出 token $X = x$ 的总概率是两条路径之和：

$$
\begin{aligned}
P(X = x) 
&=  P(\text{accept}, X = x) + P(\text{reject}, X = x) \\
&= \min(q(x), p(x)) + \max(0, p(x) - q(x))
\end{aligned}
$$

逐项验证两种情形：

- **情形 A：$p(x) \geq q(x)$。** 此时 $\min = q(x)$，$\max(0, p-q) = p(x) - q(x)$，相加得 $q(x) + p(x) - q(x) = p(x)$。
- **情形 B：$p(x) < q(x)$。** 此时 $\min = p(x)$，$\max(0, p-q) = 0$，相加得 $p(x) + 0 = p(x)$。

两种情形均给出：

$$
\boxed{P(X = x) = p(x), \quad \forall x}
$$

这证明了单步 Speculative Sampling 的输出分布**严格等于**目标分布 $p$，与草稿模型 $q$ 的质量无关。草稿模型再差，也只会增加拒绝率、浪费计算，而不会产生偏离目标分布的 token。

---

## 3. 序列层面的 lossless 保证

单步正确性只是基础。实际 Speculative Decoding 中，草稿模型一次生成 $\gamma$ 个候选 token，目标模型并行计算所有位置的条件概率，然后从前到后依次验证。我们需要证明：**输出序列的联合分布与目标模型自回归生成的联合分布完全一致**。

证明采用归纳法。

**基例（$t=1$）。** 由单步 Speculative Sampling 的正确性，直接得到 $\mathbb{P}^{\mathcal{A}}(x_1 \mid x_0) = p(x_1 \mid x_0)$，其中 $x_0$ 是 prompt。由于 $x_0$ 的分布与 $p, q$ 无关，进一步有 $\mathbb{P}^{\mathcal{A}}_1(x_1) = p_1(x_1)$。

> 这里的 $\mathcal{A}$ 指代的就是 Speculative Decoding（投机解码）的核心算法流程。

**归纳步骤。** 假设对于所有长度为 $t$ 的前缀，$\mathbb{P}^{\mathcal{A}}_t(x_1, \ldots, x_t) = p_t(x_1, \ldots, x_t)$。需要证明第 $t+1$ 个 token 的条件分布匹配。

设 $\tilde{x}_{t+1} \sim q(\cdot \mid x_{1:t})$ 是草稿采样。由全概率公式：

$$
\begin{aligned}
\mathbb{P}^{\mathcal{A}}(x_{t+1} \mid x_{1:t})
&= \mathbb{P}^{\mathcal{A}}(\tilde{x}_{t+1} = x_{t+1} \mid x_{1:t}) \cdot \mathbb{P}^{\mathcal{A}}(\text{acc} \mid \tilde{x}_{t+1} = x_{t+1}, x_{1:t}) \\
&\quad + \mathbb{P}^{\mathcal{A}}(\tilde{x}_{t+1} \text{ rej} \mid x_{1:t}) \cdot \mathbb{P}^{\mathcal{A}}(x_{t+1} \mid \tilde{x}_{t+1} \text{ rej}, x_{1:t})
\end{aligned}
$$

第一项（接受路径）的计算与单步情形完全相同：

$$
q(x_{t+1} \mid x_{1:t}) \cdot \min\!\left(1, \frac{p(x_{t+1} \mid x_{1:t})}{q(x_{t+1} \mid x_{1:t})}\right) = \min(p, q)(x_{t+1} \mid x_{1:t})
$$

第二项（拒绝路径）中，拒绝概率为 $1 - \sum_{x'} \min(p, q)(x' \mid x_{1:t})$，而重采样分布正是以 $\max(0, p - q)$ 为比例的归一化分布。两项相加再次得到 $p(x_{t+1} \mid x_{1:t})$。

因此 $\mathbb{P}^{\mathcal{A}}(x_{t+1} \mid x_{1:t}) = p(x_{t+1} \mid x_{1:t})$，结合归纳假设即得长度为 $t+1$ 的联合分布匹配。归纳完成，**Speculative Decoding 在序列层面也严格 lossless**。

---

## 4. 接受率与总变差距离

单 token 的平均接受率（acceptance rate）定义为：

$$
\alpha = \sum_x q(x) \cdot \min\!\left(1, \frac{p(x)}{q(x)}\right) = \sum_x \min(q(x), p(x))
$$

利用恒等式 $\sum_x \min(p, q) = 1 - \frac{1}{2}\sum_x |p(x) - q(x)|$，可得：

$$
\boxed{\alpha = 1 - d_{\text{TV}}(p, q)}
$$

其中 $d_{\text{TV}}(p, q) = \frac{1}{2}\sum_x |p(x) - q(x)|$ 是草稿分布与目标分布的总变差距离。这给出了一个简洁的定量关系：**草稿模型与目标模型的 TV 距离越小，接受率越高**。

对于一次生成 $\gamma$ 个草稿 token 的情形，在近似独立假设下，期望接受 token 数为：

$$
\mathbb{E}[A] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} - 1
$$

实际加速比还需考虑草稿模型的前向延迟和验证开销，经典估计为：

$$
\text{Speedup} \approx \frac{\mathbb{E}[A] \cdot \tau_p}{\gamma \cdot \tau_q + \tau_p}
$$

当 $\tau_q / \tau_p$ 很小时，加速比主要由接受率 $\alpha$ 和推测长度 $\gamma$ 决定。

---

