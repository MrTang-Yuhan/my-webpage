---
layout: post.njk
post_id: 2026-08-07-大模型中的数学-002-dp-中的梯度缩放
archive: 数学原理
title: 大模型中的数学（002）：DP 中的梯度缩放
date: 2026-08-07
tags:
  - post
---
# 数据并行中的梯度缩放

考虑一个大小为 $N$ 的全局 batch $\mathcal{B} = \{(\mathbf{x}_i, \mathbf{y}_i)\}_{i=1}^N$。全局目标损失函数采用 mean reduction：

$$
L(\mathbf{W}) = \frac{1}{N} \sum_{i=1}^{N} \ell_i(\mathbf{W})
$$

其梯度为基准真值：

$$
\mathbf{g}^\star = \nabla L(\mathbf{W}) = \frac{1}{N} \sum_{i=1}^{N} \nabla \ell_i(\mathbf{W}) \tag{1}
$$

---

## 1. 单卡梯度累加（Gradient Accumulation）

将 $\mathcal{B}$ 拆分为 $M$ 个 micro-batch $\{\mathcal{B}_m\}_{m=1}^M$，每个大小为 $B = N/M$。对第 $m$ 个 micro-batch 计算局部损失（mean reduction）：

$$
L_m(\mathbf{W}) = \frac{1}{|\mathcal{B}_m|} \sum_{i \in \mathcal{B}_m} \ell_i(\mathbf{W}) = \frac{M}{N} \sum_{i \in \mathcal{B}_m} \ell_i(\mathbf{W})
$$

反向传播得到局部梯度：

$$
\mathbf{g}_m = \nabla L_m(\mathbf{W}) = \frac{M}{N} \sum_{i \in \mathcal{B}_m} \nabla \ell_i(\mathbf{W})
$$

在同一张卡上累加 $M$ 个 micro-batch 的梯度：

$$
\sum_{m=1}^{M} \mathbf{g}_m = \sum_{m=1}^{M} \frac{M}{N} \sum_{i \in \mathcal{B}_m} \nabla \ell_i(\mathbf{W}) = \frac{M}{N} \sum_{i=1}^{N} \nabla \ell_i(\mathbf{W}) = M \cdot \mathbf{g}^\star \tag{2}
$$

**结论**：累加后的梯度是基准真值的 $M$ 倍。为保证等价性，必须执行 **缩放**。有两种等价实现：

- **方式一（缩放梯度）**：在优化器更新前将累加梯度除以 $M$
  $$
  \mathbf{g}_{\text{acc}} = \frac{1}{M} \sum_{m=1}^{M} \mathbf{g}_m = \mathbf{g}^\star
  $$

- **方式二（缩放损失）**：将每个 micro-batch 的损失预先缩放
  $$
  \tilde{L}_m(\mathbf{W}) = \frac{1}{M} \cdot L_m(\mathbf{W}) = \frac{1}{N} \sum_{i \in \mathcal{B}_m} \ell_i(\mathbf{W})
  $$
  此时 $\tilde{\mathbf{g}}_m = \frac{1}{M} \mathbf{g}_m$，累加后自然得到 $\sum_{m=1}^M \tilde{\mathbf{g}}_m = \mathbf{g}^\star$，无需额外缩放梯度。

---

## 2. 多卡数据并行（Data Parallel）

将 $\mathcal{B}$ 拆分为 $D$ 份（$D$ 为数据并行维度，即 DP size），卡 $k$ 持有子集 $\mathcal{B}_k$，满足 $|\mathcal{B}_k| = N/D$ 且 $\bigcup_{k=1}^D \mathcal{B}_k = \mathcal{B}$。卡 $k$ 的局部损失：

$$
L_k(\mathbf{W}) = \frac{1}{|\mathcal{B}_k|} \sum_{i \in \mathcal{B}_k} \ell_i(\mathbf{W}) = \frac{D}{N} \sum_{i \in \mathcal{B}_k} \ell_i(\mathbf{W})
$$

局部梯度：

$$
\mathbf{g}_k = \nabla L_k(\mathbf{W}) = \frac{D}{N} \sum_{i \in \mathcal{B}_k} \nabla \ell_i(\mathbf{W})
$$

各卡执行 `all_reduce` 求和（SUM）：

$$
\mathbf{g}_{\text{sum}} = \sum_{k=1}^{D} \mathbf{g}_k = \sum_{k=1}^{D} \frac{D}{N} \sum_{i \in \mathcal{B}_k} \nabla \ell_i(\mathbf{W}) = \frac{D}{N} \sum_{i=1}^{N} \nabla \ell_i(\mathbf{W}) = D \cdot \mathbf{g}^\star \tag{3}
$$

**结论**：all-reduce SUM 后的梯度是基准真值的 $D$ 倍。为保证等价性，必须执行 **平均**：

$$
\mathbf{g}_{\text{dp}} = \frac{1}{D} \sum_{k=1}^{D} \mathbf{g}_k = \mathbf{g}^\star
$$


---

## 3. 梯度累加与数据并行联合使用

当同时使用梯度累加（$M$ 个 micro-batch）和数据并行（$D$ 张卡）时，每张卡上的 $N/D$ 个样本再细分为 $M$ 个 micro-batch，每个 micro-batch 大小为 $N/(DM)$。卡 $k$ 的第 $m$ 个局部损失：

$$
L_{k,m}(\mathbf{W}) = \frac{1}{|\mathcal{B}_{k,m}|} \sum_{i \in \mathcal{B}_{k,m}} \ell_i(\mathbf{W}) = \frac{DM}{N} \sum_{i \in \mathcal{B}_{k,m}} \ell_i(\mathbf{W})
$$

对应局部梯度：

$$
\mathbf{g}_{k,m} = \nabla L_{k,m}(\mathbf{W}) = \frac{DM}{N} \sum_{i \in \mathcal{B}_{k,m}} \nabla \ell_i(\mathbf{W})
$$

先进行单卡上的梯度累加，再进行卡间的 all-reduce SUM：

$$
\sum_{k=1}^{D} \sum_{m=1}^{M} \mathbf{g}_{k,m} = \frac{DM}{N} \sum_{k=1}^{D} \sum_{m=1}^{M} \sum_{i \in \mathcal{B}_{k,m}} \nabla \ell_i(\mathbf{W}) = \frac{DM}{N} \sum_{i=1}^{N} \nabla \ell_i(\mathbf{W}) = DM \cdot \mathbf{g}^\star
$$

**结论**：联合累加后的梯度是基准真值的 $M \cdot D$ 倍。最终缩放应为：

$$
\mathbf{g}_{\text{total}} = \frac{1}{M \cdot D} \sum_{k=1}^{D} \sum_{m=1}^{M} \mathbf{g}_{k,m} = \mathbf{g}^\star
$$

