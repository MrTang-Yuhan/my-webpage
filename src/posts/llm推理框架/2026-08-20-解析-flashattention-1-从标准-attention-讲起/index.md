---
layout: post.njk
post_id: 2026-08-20-解析-flashattention-1-从标准-attention-讲起
archive: llm推理框架
title: 解析 FlashAttention（1）：从标准 Attention 讲起
date: 2026-08-20
tags:
  - post
---
## 1. 标准 Attention 的定义

在正式分析 FlashAttention 之前，要先彻底地了解**标准 Attention** 的做法和含义。

给定单个 Query 向量 $\mathbf{q} \in \mathbb{R}^{N \times d}$，以及 Key 矩阵 $\mathbf{K} \in \mathbb{R}^{N_{kv} \times d}$ 和 Value 矩阵 $\mathbf{V} \in \mathbb{R}^{N_{kv} \times d}$，其中 $N_{kv}$ 是 KV Cache 的序列长度（可能很长），$d$ 是注意力头维度。标准的缩放点积注意力（Scaled Dot-Product Attention）定义为：

$$
\mathrm{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\left(\frac{\mathbf{q}\mathbf{K}^{\top}}{\sqrt{d}}\right) \mathbf{V}
$$

---

## 2. 以简化的矩阵为例进行推导

以 $n=2$（两个词）、$d=3$（向量维度）为例，$Q, K, V \in \mathbb{R}^{2 \times 3}$，完整推导 Attention 计算过程。

$$
Q = \begin{bmatrix} q_1 \\ q_2 \end{bmatrix} = \begin{bmatrix} q_{11} & q_{12} & q_{13} \\ q_{21} & q_{22} & q_{23} \end{bmatrix}
$$

$$
\quad K = \begin{bmatrix} k_1 \\ k_2 \end{bmatrix} = \begin{bmatrix} k_{11} & k_{12} & k_{13} \\ k_{21} & k_{22} & k_{23} \end{bmatrix}
$$

$$
\quad V = \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} v_{11} & v_{12} & v_{13} \\ v_{21} & v_{22} & v_{23} \end{bmatrix}
$$

其中 $q_i, k_i, v_i \in \mathbb{R}^{1 \times 3}$ 为第 $i$ 个词对应的 query、key、value 行向量。

---

## 2. $S = QK^\top$：注意力得分矩阵


$$
S = QK^\top = \begin{bmatrix} q_1 k_1^\top & q_1 k_2^\top \\ q_2 k_1^\top & q_2 k_2^\top \end{bmatrix} = \begin{bmatrix} \sum_{l=1}^{3} q_{1l}k_{1l} & \sum_{l=1}^{3} q_{1l}k_{2l} \\[6pt] \sum_{l=1}^{3} q_{2l}k_{1l} & \sum_{l=1}^{3} q_{2l}k_{2l} \end{bmatrix} \in \mathbb{R}^{2 \times 2}
$$

**含义：**

- 元素 $S_{ij} = q_i k_j^\top = \sum_{l=1}^{3} q_{il} k_{jl}$：第 $i$ 个词的 query 向量 $q_i$ 与第 $j$ 个词的 key 向量 $k_j$ 的**内积**。
- 行向量 $S_i = [S_{i1},\; S_{i2}]$：第 $i$ 个词的 query 与序列中**所有词的 key** 计算得到的注意力得分向量。

---

## 3. Softmax：注意力权重矩阵

对 $S$ **逐行做 Softmax**。先减去行最大值 $m_i = \max_j S_{ij}$ 保证数值稳定，令 $\tilde{S}_{ij} = S_{ij} - m_i$，再归一化：

$$
P_{ij} = \frac{e^{\tilde{S}_{ij}}}{\sum_{j'=1}^{2} e^{\tilde{S}_{ij'}}}
$$

$$
P = \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix} \in \mathbb{R}^{2 \times 2}, \quad \text{满足 } P_{11}+P_{12}=1,\; P_{21}+P_{22}=1
$$

**含义：**

- 元素 $P_{ij} \in [0,1]$：第 $i$ 个词的 query 对第 $j$ 个词的 key 的**注意力权重**，即第 $i$ 个词**关注第 $j$ 个词的概率**。
- 行向量 $P_i = [P_{i1},\; P_{i2}]$：第 $i$ 个词的 query 与所有 key 经 Softmax 后的**注意力分布**（概率向量，和为 1）。

---

## 4. $O = PV$：输出矩阵

$$
O = PV = \begin{bmatrix} P_{11} & P_{12} \\ P_{21} & P_{22} \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} P_{11}v_1 + P_{12}v_2 \\ P_{21}v_1 + P_{22}v_2 \end{bmatrix} = \begin{bmatrix} O_1 \\ O_2 \end{bmatrix} \in \mathbb{R}^{2 \times 3}
$$

逐行展开：

$$
O_1 = P_{11}[v_{11},\; v_{12},\; v_{13}] + P_{12}[v_{21},\; v_{22},\; v_{23}] = [O_{11},\; O_{12},\; O_{13}]
$$

$$
O_2 = P_{21}[v_{11},\; v_{12},\; v_{13}] + P_{22}[v_{21},\; v_{22},\; v_{23}] = [O_{21},\; O_{22},\; O_{23}]
$$

**含义：**

- **$O_1$**：第 1 个词的 query 向量 $q_1$ 与所有 key 向量（$k_1, k_2$）计算注意力得分并归一化为注意力分布 $P_1 = [P_{11},\; P_{12}]$ 后，用该分布作为权重对所有**词的 value 向量**（$v_1, v_2$）**加权求和**的结果。即 $O_1 = P_{11}v_1 + P_{12}v_2$。
- **$O_2$**：同理，第 2 个词的 query 向量 $q_2$ 与所有 key 计算得分并归一化为 $P_2 = [P_{21},\; P_{22}]$ 后，对所有**词的 value 向量**加权求和的结果。即 $O_2 = P_{21}v_1 + P_{22}v_2$。

其中分量形式为：

$$
O_{ij} = P_{i1}v_{1j} + P_{i2}v_{2j} = \sum_{l=1}^{2} P_{il} v_{lj}
$$



**元素 $O_{ij}$ 的详细解释：**

$O_{ij}$ 是输出矩阵 $O$ 的第 $i$ 行第 $j$ 列元素。具体计算过程为：

1. 第 $i$ 个词的 query 向量 $q_i$ 与所有词的 key 向量 $k_1, k_2$ 计算注意力得分，经 Softmax 归一化得到注意力分布 $P_i = [P_{i1},\; P_{i2}]$；
2. 用该分布对所有词的 value 向量 $v_1, v_2$ 做加权求和，得到第 $i$ 个词的输出向量 $O_i = P_{i1}v_1 + P_{i2}v_2$；
3. $O_{ij}$ 就是这个输出向量 $O_i$ 的第 $j$ 个分量（即嵌入维度的第 $j$ 维）。

从矩阵乘法展开看，$O_{ij}$ 等于所有词的 value 向量第 $j$ 维分量按第 $i$ 个词对各词的关注概率加权求和：

$$
O_{ij} = P_{i1} \cdot v_{1j} + P_{i2} \cdot v_{2j}
$$

即：第 1 个词 value 向量的第 $j$ 维分量 $v_{1j}$ 乘以第 $i$ 个词对第 1 个词的关注概率 $P_{i1}$，加上第 2 个词 value 向量的第 $j$ 维分量 $v_{2j}$ 乘以第 $i$ 个词对第 2 个词的关注概率 $P_{i2}$。

- 行向量 $O_i = [O_{i1},\; O_{i2},\; O_{i3}]$：第 $i$ 个词的 query 经注意力机制后，按注意力分布 $P_i$ 对所有**词的 value 向量**加权求和得到的**输出向量**。

---

## 5. 加入 Causal Mask（因果掩码）

Decoder 中需防止第 $i$ 个词看到第 $j > i$ 个词。对 $n=2$：

$$
M = \begin{bmatrix} 0 & -\infty \\ 0 & 0 \end{bmatrix}
$$

### Masked 分数

$$
S^{\text{mask}} = S + M = \begin{bmatrix} S_{11} & -\infty \\ S_{21} & S_{22} \end{bmatrix}
$$

### Masked Softmax

第 1 行中 $e^{-\infty}=0$，故：

$$
P_{11}^{\text{mask}} = \frac{e^{\tilde{S}_{11}}}{e^{\tilde{S}_{11}}+0} = 1, \quad P_{12}^{\text{mask}} = 0
$$

第 2 行正常计算：

$$
P_{21}^{\text{mask}} = \frac{e^{\tilde{S}_{21}}}{e^{\tilde{S}_{21}}+e^{\tilde{S}_{22}}}, \quad P_{22}^{\text{mask}} = \frac{e^{\tilde{S}_{22}}}{e^{\tilde{S}_{21}}+e^{\tilde{S}_{22}}}
$$

### Masked 输出

$$
O^{\text{mask}} = P^{\text{mask}} V = \begin{bmatrix} 1 \cdot v_1 + 0 \cdot v_2 \\ P_{21}^{\text{mask}} v_1 + P_{22}^{\text{mask}} v_2 \end{bmatrix} = \begin{bmatrix} v_1 \\ P_{21}^{\text{mask}} v_1 + P_{22}^{\text{mask}} v_2 \end{bmatrix}
$$

结果：第 1 个词只能看到自己（$O_1 = v_1$），第 2 个词可以看到自己和第 1 个词。
