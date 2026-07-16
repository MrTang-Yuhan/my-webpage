---
layout: post.njk
post_id: 2026-07-16-flashdecoding-数学推导-传统方法与-lse-方法
archive: llm推理框架
title: FlashDecoding 数学推导：传统方法与 LSE 方法
date: 2026-07-16
tags:
  - post
---
# FlashDecoding 数学推导：传统 m/l/o 方法与 LSE 方法

> 参考网页：
> - LSE 方法推导：[Log-Sum-Exp (LSE) 方法](https://my-webpage-adu.pages.dev/posts/llm%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6/2026-07-13-log-sum-exp-%E6%96%B9%E6%B3%95/)
> - FlashDecoding 传统方法推导：
- [推理长序列利器：ChunkedPrefill&FlashDecoding原理详解](https://zhuanlan.zhihu.com/p/1988996116017086993)
- 



---


## 一、公式作用概述


FlashDecoding 是一种用于**大语言模型（LLM）推理解码阶段（Decoding Stage）** 的高效注意力计算算法。在 Decoding 阶段，模型每次只生成一个新 token，因此 Query 的长度为 1，这导致 GPU 的并行度严重不足（大量流式多处理器 SM 处于空闲状态）。FlashDecoding 的核心思想是：**将 KV Cache（键值缓存）沿序列维度切分为多个 Tile（子块），分配到不同的 SM 上并行计算局部注意力结果，最后通过数学合并技术将各局部结果合并为全局正确的输出**。这种方法在不改变计算结果精度的前提下，显著提升了长序列解码的 GPU 利用率。

本文档将介绍两种数学上等价的 FlashDecoding 实现方法，并分别证明它们与标准 Attention 的等价性：

1. **传统方法（m/l/o 三元组法）**：每个 Tile 存储局部最大值 $m^{(b)}$、局部指数和 $\ell^{(b)}$ 和局部输出 $\mathbf{o}^{(b)}$。

2. **LSE 方法（Log-Sum-Exp 二元组法）**：每个 Tile 存储局部 LSE 值 $S^{(b)} = \text{LSE}(\text{Tile } b)$ 和局部输出 $\mathbf{o}^{(b)}$。

两种方法均与标准 Attention **数学等价**（忽略浮点舍入误差），但 LSE 方法在通信量和实现简洁性上更具优势。


---


## 二、两种 FlashDecoding 方法的完整推导过程


### 2.1 问题背景与动机


#### 2.1.1 标准 Attention 的定义


给定单个 Query 向量 $\mathbf{q} \in \mathbb{R}^{1 \times d}$（解码阶段 Query 序列长度为 1），以及 Key 矩阵 $\mathbf{K} \in \mathbb{R}^{N_{kv} \times d}$ 和 Value 矩阵 $\mathbf{V} \in \mathbb{R}^{N_{kv} \times d}$，其中 $N_{kv}$ 是 KV Cache 的序列长度（可能很长），$d$ 是注意力头维度。标准的缩放点积注意力（Scaled Dot-Product Attention）定义为：


$$
\mathrm{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \mathrm{softmax}\left(\frac{\mathbf{q}\mathbf{K}^{\top}}{\sqrt{d}}\right) \mathbf{V}
$$


展开为逐元素形式。设 $\mathbf{s} = \frac{\mathbf{q}\mathbf{K}^{\top}}{\sqrt{d}} \in \mathbb{R}^{1 \times N_{kv}}$ 为注意力分数向量，$s_j = \frac{\mathbf{q} \cdot \mathbf{K}_j}{\sqrt{d}}$ 为第 $j$ 个位置的分数，则：


$$
\mathbf{o} = \mathrm{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \sum_{j=1}^{N_{kv}} \frac{\exp(s_j)}{\sum_{k=1}^{N_{kv}} \exp(s_k)} \cdot \mathbf{V}_j
$$


> **【知识卡片：Softmax 函数】**
> - **定义**：Softmax 函数将一个实数向量转换为概率分布，使得所有输出值在 $(0, 1)$ 之间且和为 1。
> - **公式**：对于向量 $\mathbf{x} = [x_1, x_2, \ldots, x_n]$，$\mathrm{softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j=1}^{n} \exp(x_j)}$。


#### 2.1.2 解码阶段的并行度瓶颈


在**预填充阶段（Prefill Stage）**，输入序列长度 $N_q$ 较大，Query 矩阵 $\mathbf{Q} \in \mathbb{R}^{N_q \times d}$ 有很多行。FlashAttention 的做法是：将 $\mathbf{Q}$ 沿行维度切分为多个 Tile，不同 Tile 分配给不同的 SM 并行计算。**每个 SM 处理一部分 Query，但需要使用完整的 $\mathbf{K}$ 和 $\mathbf{V}$**。

然而，在**解码阶段（Decoding Stage）**，模型正在自回归地生成新 token，每次只需要计算**一个 Query**（即新 token 的 Query），所以 $\mathbf{q} \in \mathbb{R}^{1 \times d}$。这意味着：

- **无法通过切分 Query 来获得并行度**（Query 只有一行）。
- 如果使用 FlashAttention 的原版策略，只有一个 SM（或少量 SM）在工作，**大量 SM 空闲**。
- 同时，KV Cache 的序列长度 $N_{kv}$ 可能非常长（例如 32K、64K 甚至更长），读取 KV Cache 成为瓶颈。


#### 2.1.3 FlashDecoding 的核心思想


FlashDecoding 的解决方案是**反转并行策略**：

1. **不再切分 Query**（因为 Query 只有一行）。
2. **改为切分 KV Cache**：将 $\mathbf{K}$ 和 $\mathbf{V}$ 沿序列维度切分为 $B$ 个 Tile：$$\mathbf{K} = [\mathbf{K}^{(1)}; \mathbf{K}^{(2)}; \ldots; \mathbf{K}^{(B)}], \quad \mathbf{V} = [\mathbf{V}^{(1)}; \mathbf{V}^{(2)}; \ldots; \mathbf{V}^{(B)}]$$
   其中每个 $\mathbf{K}^{(b)}, \mathbf{V}^{(b)} \in \mathbb{R}^{N_{\text{tile}} \times d}$。

3. **每个 SM 处理一个 KV Tile**：SM $b$ 计算 $\mathbf{q}$ 与 $\mathbf{K}^{(b)}, \mathbf{V}^{(b)}$ 的局部注意力结果。

4. **使用数学合并技术合并局部结果**：由于 softmax 涉及全局归一化，各 SM 的局部结果不能直接相加，需要通过数学技巧进行归一化合并。


---


### 2.2 传统方法（m/l/o 三元组法）


#### 2.2.1 每个 Tile 的局部计算


对于第 $b$ 个 Tile（$b = 1, 2, \ldots, B$），SM $b$ 计算以下三个局部统计量：

**（1）局部最大值**（running max）：


$$
m^{(b)} = \max_{j=1}^{N_{\text{tile}}} s_j^{(b)}
$$


其中 $s_j^{(b)} = \frac{\mathbf{q} \cdot \mathbf{K}_j^{(b)}}{\sqrt{d}}$ 是 Tile $b$ 中第 $j$ 个位置的注意力分数。$m^{(b)}$ 用于数值稳定，避免指数溢出。

**（2）局部指数和**（running sum）：


$$
\ell^{(b)} = \sum_{j=1}^{N_{\text{tile}}} \exp\bigl(s_j^{(b)} - m^{(b)}\bigr)
$$


这是 Tile $b$ 内所有（经数值稳定后的）指数值之和。

**（3）局部加权输出**：


$$
\mathbf{o}^{(b)} = \frac{\sum_{j=1}^{N_{\text{tile}}} \exp\bigl(s_j^{(b)} - m^{(b)}\bigr) \cdot \mathbf{V}_j^{(b)}}{\ell^{(b)}} \in \mathbb{R}^{1 \times d}
$$


这是 Tile $b$ 的局部 softmax 结果，即 Value 的加权平均，权重来自 Tile 内部的 softmax 归一化。




#### 2.2.2 全局合并公式


**目标**：给定所有 Tile 的局部统计量 $\{(m^{(b)}, \ell^{(b)}, \mathbf{o}^{(b)})\}_{b=1}^{B}$，求全局正确的注意力输出 $\mathbf{o}_{\text{final}}$。

**步骤 1：求全局最大值**


$$
m_{\text{global}} = \max_{b=1}^{B} m^{(b)}
$$


$\mathbf{q}$ 对所有 Key 的注意力分数中的真正最大值。

**步骤 2：计算全局归一化因子**


$$
\ell_{\text{global}} = \sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)}
$$


> **推导依据**：
> $$\ell_{\text{global}} = \sum_{b=1}^{B} \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m_{\text{global}}) = \\
\sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m^{(b)}) = \sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)}$$

**步骤 3：合并局部输出**


$$
\mathbf{o}_{\text{final}} = \frac{\sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)}}{\ell_{\text{global}}}
$$


> **推导依据**：
> $$\mathbf{o}_{\text{final}} = \frac{\sum_{b=1}^{B} \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m_{\text{global}}) \cdot \mathbf{V}_j^{(b)}}{\sum_{b=1}^{B} \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m_{\text{global}})}$$
> 分子展开：
> $$\sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m_{\text{global}}) \cdot \mathbf{V}_j^{(b)} = \\
\exp(m^{(b)} - m_{\text{global}}) \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m^{(b)}) \cdot \mathbf{V}_j^{(b)} = \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)}$$
> 代入即得全局合并公式。


#### 2.2.3 传统方法与标准 Attention 的等价性证明


**定理**：传统 FlashDecoding 方法的分块计算结果与全序列直接计算的标准 Attention 结果完全等价。

**证明**：

全序列直接计算的标准 Attention 输出为：


$$
\mathbf{o}_{\text{direct}} = \frac{\sum_{j=1}^{N_{kv}} \exp(s_j - m_{\text{global}}) \cdot \mathbf{V}_j}{\sum_{j=1}^{N_{kv}} \exp(s_j - m_{\text{global}})}
$$


将序列按 Tile 划分后，分子和分母都可以按 Tile 分解。对于 Tile $b$ 内部的元素，利用 $m^{(b)}$ 进行分解：


$$
\exp(s_j - m_{\text{global}}) = \exp(s_j - m^{(b)}) \cdot \exp(m^{(b)} - m_{\text{global}})
$$


因此，Tile $b$ 对分子的贡献为：


$$
\sum_{j \in \text{Tile } b} \exp(s_j - m_{\text{global}}) \cdot \mathbf{V}_j = \\
\exp(m^{(b)} - m_{\text{global}}) \sum_{j \in \text{Tile } b} \exp(s_j - m^{(b)}) \cdot \mathbf{V}_j = \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)}
$$


分母为：


$$
\sum_{j=1}^{N_{kv}} \exp(s_j - m_{\text{global}}) = \sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} = \ell_{\text{global}}
$$


因此：


$$
\mathbf{o}_{\text{direct}} = \frac{\sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)}}{\ell_{\text{global}}} = \mathbf{o}_{\text{final}}
$$


**证毕。**


---


### 2.3 LSE 方法（Log-Sum-Exp 二元组法）


#### 2.3.1 从三元组到二元组的思路


传统方法需要在 Tile 之间传递三个量 $(m^{(b)}, \ell^{(b)}, \mathbf{o}^{(b)})$。LSE 方法的核心洞察是：**可以将 $m^{(b)}$ 和 $\ell^{(b)}$ 融合为一个标量 $S^{(b)}$**，从而将通信量从三个量减少到两个量。

> **【知识卡片：Log-Sum-Exp (LSE) 函数】**
> - **定义**：对于向量 $\mathbf{x} = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n$，$\text{LSE}(\mathbf{x}) = \log\left(\sum_{i=1}^{n} \exp(x_i)\right)$。
> - **关键性质**：$\exp(\text{LSE}(\mathbf{x})) = \sum_{i=1}^{n} \exp(x_i)$，恰好是 softmax 的分母。
> - **数值稳定版本**：$\text{LSE}(\mathbf{x}) = m + \log\left(\sum_{i=1}^{n} \exp(x_i - m)\right)$，其中 $m = \max_i x_i$。
> - 证明过程见: [softmax 修正公式：Log-Sum-Exp (LSE) 方法](https://my-webpage-adu.pages.dev/posts/llm%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6/2026-07-13-log-sum-exp-%E6%96%B9%E6%B3%95/)


#### 2.3.2 定义局部 LSE 值 $S^{(b)}$


对于第 $b$ 个 Tile，定义其局部 LSE 值为该 Tile 中所有注意力分数的 log-sum-exp：


$$
S^{(b)} = \text{LSE}(\mathbf{s}^{(b)}) = \log\left(\sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)})\right)
$$


利用 safe LSE 技巧（以 $m^{(b)}$ 为参考最大值），数值稳定地计算：


$$
S^{(b)} = m^{(b)} + \log\left(\sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m^{(b)})\right) = m^{(b)} + \log(\ell^{(b)})
$$


> **推导验证**：
> $$\sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)}) = \sum_{j=1}^{N_{\text{tile}}} \exp(m^{(b)} + s_j^{(b)} - m^{(b)}) = \\
\exp(m^{(b)}) \cdot \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m^{(b)}) = \exp(m^{(b)}) \cdot \ell^{(b)}$$
> 取对数：
> $$S^{(b)} = \log(\exp(m^{(b)}) \cdot \ell^{(b)}) = m^{(b)} + \log(\ell^{(b)})$$

> **关键观察**：$S^{(b)} = m^{(b)} + \log(\ell^{(b)})$，这说明 $S^{(b)}$ 完全由 $m^{(b)}$ 和 $\ell^{(b)}$ 决定，二者编码了相同的信息。


#### 2.3.3 LSE 方法的局部计算


LSE 方法下，每个 Tile 只需计算和存储两个量：

| 符号 | 名称 | 计算公式 |
|------|------|----------|
| $S^{(b)}$ | 局部 LSE 值 | $S^{(b)} = \log\left(\sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)})\right) = m^{(b)} + \log(\ell^{(b)})$ |
| $\mathbf{o}^{(b)}$ | 局部输出 | $\mathbf{o}^{(b)} = \frac{\sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m^{(b)}) \cdot \mathbf{V}_j^{(b)}}{\ell^{(b)}}$（与传统方法相同） |


#### 2.3.4 LSE 方法的全局合并公式


**目标**：给定所有 Tile 的局部量 $\{(S^{(b)}, \mathbf{o}^{(b)})\}_{b=1}^{B}$，求全局正确的注意力输出 $\mathbf{o}_{\text{final}}$。

**步骤 1：计算全局 LSE**


$$
S_{\text{global}} = \text{LSE}(S^{(1)}, S^{(2)}, \ldots, S^{(B)}) = \log\left(\sum_{b=1}^{B} \exp(S^{(b)})\right)
$$


**步骤 2：计算全局输出**


$$
\mathbf{o}_{\text{final}} = \sum_{b=1}^{B} \exp(S^{(b)} - S_{\text{global}}) \cdot \mathbf{o}^{(b)}
$$


> **推导依据**：
> 全局正确的 Attention 输出为：
> $$\mathbf{o}_{\text{final}} = \frac{\sum_{b=1}^{B} \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)}) \cdot \mathbf{V}_j^{(b)}}{\sum_{b=1}^{B} \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)})}$$
> 分子（Tile $b$ 的贡献）：
> $$\sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)}) \cdot \mathbf{V}_j^{(b)} = \exp(m^{(b)}) \cdot \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m^{(b)}) \cdot \mathbf{V}_j^{(b)} = \\
\exp(m^{(b)}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)} = \exp(S^{(b)}) \cdot \mathbf{o}^{(b)}$$
> 分母：
> $$\sum_{b=1}^{B} \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)}) = \sum_{b=1}^{B} \exp(S^{(b)}) = \exp(S_{\text{global}})$$
> 因此：
> $$\mathbf{o}_{\text{final}} = \frac{\sum_{b=1}^{B} \exp(S^{(b)}) \cdot \mathbf{o}^{(b)}}{\exp(S_{\text{global}})} = \sum_{b=1}^{B} \exp(S^{(b)} - S_{\text{global}}) \cdot \mathbf{o}^{(b)}$$


#### 2.3.5 LSE 合并算子 $\oplus$ 及其结合律


为书写简洁，定义合并算子 $\oplus$ 作用于二元组 $(\mathbf{o}, S)$：


$$
\begin{bmatrix} \mathbf{o}(I \cup J) \\ S(I \cup J) \end{bmatrix} = \begin{bmatrix} \mathbf{o}(I) \\ S(I) \end{bmatrix} \oplus \begin{bmatrix} \mathbf{o}(J) \\ S(J) \end{bmatrix}
$$


其中 $\oplus$ 的具体运算规则为：


$$
\begin{bmatrix} \mathbf{o}_1 \\ S_1 \end{bmatrix} \oplus \begin{bmatrix} \mathbf{o}_2 \\ S_2 \end{bmatrix} = \begin{bmatrix} \displaystyle\frac{\exp(S_1) \cdot \mathbf{o}_1 + \exp(S_2) \cdot \mathbf{o}_2}{\exp(S_1) + \exp(S_2)} \\ \log(\exp(S_1) + \exp(S_2)) \end{bmatrix}
$$


**关键性质**：该算子 $\oplus$ 满足**结合律**（associative），即：


$$
(A \oplus B) \oplus C = A \oplus (B \oplus C)
$$


> **数学依据**：因为 LSE 和加权和的运算本质上是对指数和的累加，而求和运算天然满足结合律。这意味着无论按什么顺序合并多个 Tile，最终结果都相同。


#### 2.3.6 LSE 方法与标准 Attention 的等价性证明


**定理**：LSE FlashDecoding 方法的分块计算结果与全序列直接计算的标准 Attention 结果完全等价。

**证明**：

全序列直接计算的标准 Attention 输出为：


$$
\mathbf{o}_{\text{direct}} = \frac{\sum_{j=1}^{N_{kv}} \exp(s_j) \cdot \mathbf{V}_j}{\sum_{j=1}^{N_{kv}} \exp(s_j)}
$$


将序列按 Tile 划分后，利用 $\exp(S^{(b)}) = \sum_{j \in \text{Tile } b} \exp(s_j)$（由 LSE 定义直接可得），全局分母为：


$$
\sum_{j=1}^{N_{kv}} \exp(s_j) = \sum_{b=1}^{B} \sum_{j \in \text{Tile } b} \exp(s_j) = \sum_{b=1}^{B} \exp(S^{(b)}) = \exp(S_{\text{global}})
$$


全局分子中，Tile $b$ 的贡献为：


$$
\sum_{j \in \text{Tile } b} \exp(s_j) \cdot \mathbf{V}_j = \exp(S^{(b)}) \cdot \mathbf{o}^{(b)}
$$


因此：


$$
\mathbf{o}_{\text{direct}} = \frac{\sum_{b=1}^{B} \exp(S^{(b)}) \cdot \mathbf{o}^{(b)}}{\exp(S_{\text{global}})} = \sum_{b=1}^{B} \exp(S^{(b)} - S_{\text{global}}) \cdot \mathbf{o}^{(b)} = \mathbf{o}_{\text{final}}
$$


**证毕。**


---


### 2.4 两种 FlashDecoding 方法之间的等价性证明


**定理**：传统 m/l/o 方法与 LSE S/o 方法在数学上完全等价，即对于相同的输入和相同的 Tile 划分，两种方法产生的全局输出完全相同。

**证明**：

**第一部分：局部量之间的关系**

对于任意 Tile $b$，由 2.3.2 节的推导：


$$
S^{(b)} = m^{(b)} + \log(\ell^{(b)}) \iff \exp(S^{(b)}) = \exp(m^{(b)}) \cdot \ell^{(b)}
$$


**第二部分：全局归一化因子的等价性**

传统方法的全局归一化因子为 $\ell_{\text{global}}^{\text{(trad)}} = \sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)}$。

LSE 方法的全局归一化因子为 $\exp(S_{\text{global}})$。


$$
\begin{aligned}
\exp(S_{\text{global}}) &= \sum_{b=1}^{B} \exp(S^{(b)}) = \sum_{b=1}^{B} \exp(m^{(b)}) \cdot \ell^{(b)} \\
&= \sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}} + m_{\text{global}}) \cdot \ell^{(b)} \\
&= \exp(m_{\text{global}}) \cdot \sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \\
&= \exp(m_{\text{global}}) \cdot \ell_{\text{global}}^{\text{(trad)}}
\end{aligned}
$$


因此 $\exp(S_{\text{global}}) = \exp(m_{\text{global}}) \cdot \ell_{\text{global}}^{\text{(trad)}}$。

**第三部分：全局输出的等价性**

LSE 方法的权重：


$$
\exp(S^{(b)} - S_{\text{global}}) = \frac{\exp(S^{(b)})}{\exp(S_{\text{global}})} = \frac{\exp(m^{(b)}) \cdot \ell^{(b)}}{\exp(m_{\text{global}}) \cdot \ell_{\text{global}}^{\text{(trad)}}} = \frac{\exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)}}{\ell_{\text{global}}^{\text{(trad)}}}
$$


代入 LSE 方法的全局输出公式：


$$
\mathbf{o}_{\text{final}}^{\text{(lse)}} = \sum_{b=1}^{B} \frac{\exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)}}{\ell_{\text{global}}^{\text{(trad)}}} \cdot \mathbf{o}^{(b)} = \\
\frac{\sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)}}{\ell_{\text{global}}^{\text{(trad)}}} = \mathbf{o}_{\text{final}}^{\text{(trad)}}
$$


**证毕。**


> **【直观理解】**
> 两种方法的本质区别在于**信息编码方式**：
> - **传统方法**将指数和 $\sum \exp(s_j)$ 编码为两个数 $(m^{(b)}, \ell^{(b)})$ 的乘积形式 $\exp(m^{(b)}) \cdot \ell^{(b)}$。合并时需要先统一参考系（减去全局最大值 $m_{\text{global}}$）。
> - **LSE 方法**将指数和直接编码为对数空间的一个标量 $S^{(b)} = \log(\sum \exp(s_j))$。合并时直接使用对数空间的加法规则。
> 两种编码方式完全等价（因为 $S^{(b)} = m^{(b)} + \log(\ell^{(b)})$），只是"坐标系"不同。


---


## 三、两种 FlashDecoding 方法的完整算法流程与核心代码


### 3.1 传统方法（m/l/o 三元组法）


#### 3.1.1 完整算法流程


**输入**：Query 向量 $\mathbf{q} \in \mathbb{R}^{1 \times d}$，Key 矩阵 $\mathbf{K} \in \mathbb{R}^{N_{kv} \times d}$，Value 矩阵 $\mathbf{V} \in \mathbb{R}^{N_{kv} \times d}$，Tile 大小 $N_{\text{tile}}$。

**输出**：全局注意力结果 $\mathbf{o}_{\text{final}} \in \mathbb{R}^{1 \times d}$。


---

**阶段 1：各 SM 并行计算局部结果（对每个 Tile 并行执行）**

对每个 Tile $b = 1, 2, \ldots, B$：

(1a) 提取 KV Tile：


$$
\mathbf{K}^{(b)} = \mathbf{K}[(b-1) \cdot N_{\text{tile}} \;:\; \min(b \cdot N_{\text{tile}}, N_{kv}), \; :]
$$


$$
\mathbf{V}^{(b)} = \mathbf{V}[(b-1) \cdot N_{\text{tile}} \;:\; \min(b \cdot N_{\text{tile}}, N_{kv}), \; :]
$$


(1b) 计算注意力分数：


$$
\mathbf{s}^{(b)} = \frac{\mathbf{q} \mathbf{K}^{(b)\top}}{\sqrt{d}}
$$


(1c) 计算局部统计量（数值稳定的 Online Softmax）：

初始化：


$$
m_1 = s_1^{(b)}, \quad \ell_1 = 1, \quad \mathbf{o}_1 = \mathbf{V}_1^{(b)}
$$


对 $j = 2, 3, \ldots, N_{\text{tile}}^{(b)}$ 递推：


$$
m_j = \max(m_{j-1}, \, s_j^{(b)})
$$


$$
\ell_j = \ell_{j-1} \cdot \exp(m_{j-1} - m_j) + \exp(s_j^{(b)} - m_j)
$$


$$
\mathbf{o}_j = \frac{\ell_{j-1} \cdot \exp(m_{j-1} - m_j) \cdot \mathbf{o}_{j-1} + \exp(s_j^{(b)} - m_j) \cdot \mathbf{V}_j^{(b)}}{\ell_j}
$$


Tile 最终结果：


$$
m^{(b)} = m_{N_{\text{tile}}^{(b)}}, \quad \ell^{(b)} = \ell_{N_{\text{tile}}^{(b)}}, \quad \mathbf{o}^{(b)} = \mathbf{o}_{N_{\text{tile}}^{(b)}}
$$


---

**阶段 2：全局归约合并（一个轻量级的 Reduce Kernel）**

(2a) 求全局最大值：


$$
m_{\text{global}} = \max_{b=1}^{B} m^{(b)}
$$


(2b) 计算全局归一化因子：


$$
\ell_{\text{global}} = \sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)}
$$


(2c) 合并局部输出：


$$
\mathbf{o}_{\text{final}} = \frac{\sum_{b=1}^{B} \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)}}{\ell_{\text{global}}}
$$


#### 3.1.2 核心代码示例（单流顺序合并版本）


```python
import torch
import torch.nn.functional as F
import math


class FlashDecodingTraditional:
    """传统方法（m/l/o 三元组法）的 Flash-Decoding 实现"""

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def traditional_attention(self, q, k, v):
        """标准 Attention（用于验证正确性）"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def flash_decoding_traditional(self, q, k, v, tile_size_kv: int = 256):
        """
        传统方法 Flash-Decoding：存储 m^(b), l^(b), o^(b) 三个量
        单流顺序处理所有 Tile
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        num_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv

        # 初始化全局归约变量（对应全局 m_global, l_global, o_global）
        global_max = torch.full(
            (batch_size, num_heads, seq_len_q, 1),
            -float('inf'), device=q.device, dtype=q.dtype
        )
        numerator = torch.zeros(
            batch_size, num_heads, seq_len_q, head_dim,
            device=q.device, dtype=q.dtype
        )
        denominator = torch.zeros(
            batch_size, num_heads, seq_len_q, 1,
            device=q.device, dtype=q.dtype
        )

        # 阶段 1：逐个 Tile 计算局部统计量并实时合并到全局
        for tile_idx in range(num_tiles):
            start_idx = tile_idx * tile_size_kv
            end_idx = min(start_idx + tile_size_kv, seq_len_kv)

            k_tile = k[:, :, start_idx:end_idx, :]
            v_tile = v[:, :, start_idx:end_idx, :]

            # (1b) 计算注意力分数
            S_tile = torch.matmul(q, k_tile.transpose(-2, -1)) / math.sqrt(head_dim)

            # (1c) 计算局部统计量
            m_tile = S_tile.max(dim=-1, keepdim=True).values  # m^(b)
            exp_tile = torch.exp(S_tile - m_tile)
            l_tile = exp_tile.sum(dim=-1, keepdim=True)       # l^(b)
            o_tile = torch.matmul(exp_tile, v_tile) / l_tile  # o^(b)（已归一化）

            # 阶段 2：将当前 Tile 合并到全局（实时归约）
            # (2a) 更新全局最大值
            new_global_max = torch.maximum(global_max, m_tile)

            # (2b) 调整历史累积值到新的全局最大值尺度
            if tile_idx > 0:
                scale = torch.exp(global_max - new_global_max)
                numerator = numerator * scale
                denominator = denominator * scale

            global_max = new_global_max

            # (2c) 累加当前 Tile 的贡献（调整到全局尺度）
            tile_scale = torch.exp(m_tile - global_max)
            numerator = numerator + o_tile * l_tile * tile_scale
            denominator = denominator + l_tile * tile_scale

        # 最终归一化
        final_output = numerator / denominator

        return final_output
```


#### 3.1.3 核心代码示例（多流分布式版本）


```python
    def flash_decoding_distributed_tiling(self, q, k, v,
                                          tile_size_kv: int = 256,
                                          num_streams: int = 5):
        """
        传统方法 Flash-Decoding：多流并行处理，最后树形归约
        每个流独立累积 (O, M, L)，最后合并
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        num_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv

        # 创建流数组：每个流有独立的 (O, M, L)
        stream_O = []  # 加权和数组
        stream_M = []  # 最大值数组
        stream_L = []  # exp 和数组

        for _ in range(num_streams):
            stream_O.append(torch.zeros_like(q))
            stream_M.append(torch.full(
                (batch_size, num_heads, seq_len_q, 1),
                -float('inf'), device=q.device, dtype=q.dtype
            ))
            stream_L.append(torch.zeros_like(stream_M[-1]))

        # 阶段 1：模拟流并行处理 tile
        for tile_idx in range(num_tiles):
            stream_id = tile_idx % num_streams

            start_idx = tile_idx * tile_size_kv
            end_idx = min(start_idx + tile_size_kv, seq_len_kv)
            k_tile = k[:, :, start_idx:end_idx, :]
            v_tile = v[:, :, start_idx:end_idx, :]

            # 当前流的累加器
            O_curr = stream_O[stream_id]
            M_curr = stream_M[stream_id]
            L_curr = stream_L[stream_id]

            # 计算当前 tile
            S_tile = torch.matmul(q, k_tile.transpose(-2, -1)) / math.sqrt(head_dim)
            m_tile = S_tile.max(dim=-1, keepdim=True).values

            # Online Softmax 递推：合并当前 tile 到流累加器
            new_M = torch.maximum(M_curr, m_tile)
            scale = torch.exp(M_curr - new_M)
            O_curr = O_curr * scale
            L_curr = L_curr * scale

            exp_tile = torch.exp(S_tile - new_M)
            l_tile = exp_tile.sum(dim=-1, keepdim=True)

            stream_O[stream_id] = O_curr + torch.matmul(exp_tile, v_tile)
            stream_L[stream_id] = L_curr + l_tile
            stream_M[stream_id] = new_M

        # 阶段 2：树形归约所有流
        return self.reduce_stream_arrays(stream_O, stream_M, stream_L)

    def reduce_stream_arrays(self, stream_O, stream_M, stream_L):
        """树形归约：两两合并流，直到只剩一个"""
        current_O = stream_O.copy()
        current_M = stream_M.copy()
        current_L = stream_L.copy()
        remaining = len(current_O)

        while remaining > 1:
            next_O, next_M, next_L = [], [], []
            for i in range(0, remaining, 2):
                if i + 1 < remaining:
                    O1, M1, L1 = current_O[i], current_M[i], current_L[i]
                    O2, M2, L2 = current_O[i+1], current_M[i+1], current_L[i+1]

                    # 全局合并公式
                    new_M = torch.maximum(M1, M2)
                    scale1 = torch.exp(M1 - new_M)
                    scale2 = torch.exp(M2 - new_M)
                    merged_O = O1 * scale1 + O2 * scale2
                    merged_L = L1 * scale1 + L2 * scale2

                    next_O.append(merged_O)
                    next_M.append(new_M)
                    next_L.append(merged_L)
                else:
                    next_O.append(current_O[i])
                    next_M.append(current_M[i])
                    next_L.append(current_L[i])

            current_O, current_M, current_L = next_O, next_M, next_L
            remaining = len(current_O)

        return current_O[0] / current_L[0]
```


| 代码变量 | 对应数学符号 | 含义 |
|----------|-------------|------|
| `m_tile` / `M_curr` | $m^{(b)}$ | 局部/全局最大值 |
| `l_tile` / `L_curr` | $\ell^{(b)}$ | 局部/全局指数和 |
| `o_tile` / `O_curr` | $\mathbf{o}^{(b)}$ | 局部/全局加权输出 |
| `global_max` | $m_{\text{global}}$ | 全局最大值 |
| `numerator` | $\sum_b \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)}$ | 全局分子 |
| `denominator` | $\ell_{\text{global}}$ | 全局归一化因子 |
| `scale = torch.exp(M1 - new_M)` | $\exp(m^{(1)} - m_{\text{global}})$ | 尺度调整因子 |


---


### 3.2 LSE 方法（S/o 二元组法）


#### 3.2.1 完整算法流程


**输入**：Query 向量 $\mathbf{q} \in \mathbb{R}^{1 \times d}$，Key 矩阵 $\mathbf{K} \in \mathbb{R}^{N_{kv} \times d}$，Value 矩阵 $\mathbf{V} \in \mathbb{R}^{N_{kv} \times d}$，Tile 大小 $N_{\text{tile}}$，流数量 $N_{\text{streams}}$。

**输出**：全局注意力结果 $\mathbf{o}_{\text{final}} \in \mathbb{R}^{1 \times d}$。


---

**阶段 1：各 SM 并行计算局部结果（对每个 Tile 并行执行）**

对每个 Tile $b = 1, 2, \ldots, B$：

(1a) 提取 KV Tile（同传统方法）。

(1b) 计算注意力分数（同传统方法）：


$$
\mathbf{s}^{(b)} = \frac{\mathbf{q} \mathbf{K}^{(b)\top}}{\sqrt{d}}
$$


(1c) 计算 LSE 二元组：

先计算局部最大值和局部指数和（用于数值稳定）：


$$
m^{(b)} = \max_{j=1}^{N_{\text{tile}}} s_j^{(b)}, \quad \ell^{(b)} = \sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m^{(b)})
$$


再计算局部 LSE 值：


$$
S^{(b)} = m^{(b)} + \log(\ell^{(b)})
$$


局部输出（与传统方法相同）：


$$
\mathbf{o}^{(b)} = \frac{\sum_{j=1}^{N_{\text{tile}}} \exp(s_j^{(b)} - m^{(b)}) \cdot \mathbf{V}_j^{(b)}}{\ell^{(b)}}
$$


---

**阶段 2：流内合并（每个流独立累积其分配的 Tile）**

对于每个流 $s = 0, 1, \ldots, N_{\text{streams}} - 1$：

初始化：$(\mathbf{o}_{\text{acc}}^{(s)}, S_{\text{acc}}^{(s)}) = (\text{zero-tensor}, -\infty\text{-tensor})$

对流中每个新 Tile $(\mathbf{o}_i, S_i)$，使用 LSE 合并算子 $\oplus$：


$$
\begin{bmatrix} \mathbf{o}_{\text{acc}}^{(s)} \\ S_{\text{acc}}^{(s)} \end{bmatrix} \leftarrow \begin{bmatrix} \mathbf{o}_{\text{acc}}^{(s)} \\ S_{\text{acc}}^{(s)} \end{bmatrix} \oplus \begin{bmatrix} \mathbf{o}_i \\ S_i \end{bmatrix}
$$


其中数值稳定的合并实现为：


$$
S_{\max} = \max(S_{\text{acc}}, S_i), \quad S_{\min} = \min(S_{\text{acc}}, S_i)
$$


$$
S_{\text{merged}} = S_{\max} + \log(1 + \exp(S_{\min} - S_{\max}))
$$


$$
\mathbf{o}_{\text{merged}} = \frac{\mathbf{o}_{\text{acc}} \cdot \exp(S_{\text{acc}} - S_{\text{merged}}) + \mathbf{o}_i \cdot \exp(S_i - S_{\text{merged}})}{1}
$$


---

**阶段 3：全局归约合并**

(3a) 迭代计算全局 LSE：


$$
S_{\text{global}} = S_{\text{acc}}^{(0)} \oplus S_{\text{acc}}^{(1)} \oplus \cdots \oplus S_{\text{acc}}^{(N_{\text{streams}}-1)}
$$


（使用数值稳定的两数 LSE 迭代。）

(3b) 合并所有流的输出：


$$
\mathbf{o}_{\text{final}} = \sum_{s=0}^{N_{\text{streams}}-1} \exp(S_{\text{acc}}^{(s)} - S_{\text{global}}) \cdot \mathbf{o}_{\text{acc}}^{(s)}
$$


#### 3.2.2 核心代码示例


```python
import torch
import torch.nn.functional as F
import math


class FlashDecodingLSE:
    """LSE 方法（S/o 二元组法）的 Flash-Decoding 实现"""

    def __init__(self, d_model: int = 512, num_heads: int = 8):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

    def traditional_attention(self, q, k, v):
        """标准 Attention（用于验证正确性）"""
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights

    def compute_tile_output(self, q, k_tile, v_tile):
        """
        计算单个 Tile 的 LSE 二元组 (O_i, S_i)
        返回: (O_i, S_i) 其中 S_i = m_i + log(l_i) = LSE(tile)
        """
        # 计算注意力分数
        S_tile = torch.matmul(q, k_tile.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 计算局部最大值 m_i 和局部指数和 l_i
        m_i = S_tile.max(dim=-1, keepdim=True).values
        exp_tile = torch.exp(S_tile - m_i)
        l_i = exp_tile.sum(dim=-1, keepdim=True)

        # 计算局部输出 O_i（已归一化）
        O_i = torch.matmul(exp_tile, v_tile) / l_i

        # 计算局部 LSE 值 S_i = m_i + log(l_i)
        S_i = m_i + torch.log(l_i + 1e-12)

        return O_i, S_i

    def merge_two_lse(self, O1, S1, O2, S2):
        """
        数值稳定的两数 LSE 合并
        对应公式: LSE(a, b) = max(a, b) + log(1 + exp(min(a, b) - max(a, b)))
        """
        S_max = torch.maximum(S1, S2)
        S_min = torch.minimum(S1, S2)
        log_term = torch.log1p(torch.exp(S_min - S_max))
        S_merged = S_max + log_term

        # 修正两个部分的输出贡献
        weight1 = torch.exp(S1 - S_merged)
        weight2 = torch.exp(S2 - S_merged)
        O_merged = O1 * weight1 + O2 * weight2

        return O_merged, S_merged

    def flash_decoding_lse(self, q, k, v,
                           tile_size_kv: int = 256,
                           num_streams: int = 4):
        """
        LSE 方法 Flash-Decoding：存储 S^(b), o^(b) 两个量
        """
        batch_size, num_heads, seq_len_q, head_dim = q.shape
        seq_len_kv = k.shape[2]
        num_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv

        # 初始化流数组：每个流存储 (O_acc, S_acc)
        streams_data = []
        for _ in range(num_streams):
            O_stream = torch.zeros_like(q)
            S_stream = torch.full(
                (batch_size, num_heads, seq_len_q, 1),
                -float('inf'), device=q.device, dtype=q.dtype
            )
            streams_data.append((O_stream, S_stream))

        # 阶段 1：处理每个 tile，分配到不同流
        for tile_idx in range(num_tiles):
            stream_id = tile_idx % num_streams

            start_idx = tile_idx * tile_size_kv
            end_idx = min(start_idx + tile_size_kv, seq_len_kv)
            k_tile = k[:, :, start_idx:end_idx, :]
            v_tile = v[:, :, start_idx:end_idx, :]

            # 计算当前 tile 的 LSE 二元组
            O_i, S_i = self.compute_tile_output(q, k_tile, v_tile)

            # 获取当前流的累加器
            O_acc, S_acc = streams_data[stream_id]

            # 阶段 2：使用 LSE 合并算子将 tile 合并到流累加器
            if torch.all(S_acc == -float('inf')):
                # 第一个 tile，直接赋值
                streams_data[stream_id] = (O_i, S_i)
            else:
                O_merged, S_merged = self.merge_two_lse(O_acc, S_acc, O_i, S_i)
                streams_data[stream_id] = (O_merged, S_merged)

        # 阶段 3：归约所有流的结果
        return self.merge_all_streams(streams_data)

    def merge_all_streams(self, streams_data):
        """
        两步合并算法：
        1. 迭代计算全局 S_global
        2. 用 S_global 修正每个流的输出贡献
        """
        if not streams_data:
            return None

        # 步骤 1: 迭代计算全局 S_global
        S_list = [S_i for _, S_i in streams_data]
        S_global = S_list[0].clone()
        for i in range(1, len(S_list)):
            S_i = S_list[i]
            S_max = torch.maximum(S_global, S_i)
            S_min = torch.minimum(S_global, S_i)
            log_term = torch.log1p(torch.exp(S_min - S_max))
            S_global = S_max + log_term

        # 步骤 2: 修正每个流的输出贡献
        O_global = torch.zeros_like(streams_data[0][0])
        for O_i, S_i in streams_data:
            weight = torch.exp(S_i - S_global)
            O_global += O_i * weight

        return O_global
```


| 代码变量 | 对应数学符号 | 含义 |
|----------|-------------|------|
| `S_tile` | $\mathbf{s}^{(b)}$ | Tile $b$ 上的注意力分数矩阵 |
| `m_i` | $m^{(b)}$ | 局部最大值 |
| `l_i` | $\ell^{(b)}$ | 局部指数和 |
| `S_i` | $S^{(b)}$ | 局部 LSE 值 $m^{(b)} + \log(\ell^{(b)})$ |
| `O_i` | $\mathbf{o}^{(b)}$ | 局部 softmax 加权输出 |
| `S_global` | $S_{\text{global}}$ | 全局 LSE 值 |
| `torch.log1p(torch.exp(S_min - S_max))` | $\log(1 + \exp(S_{\min} - S_{\max}))$ | 数值稳定的两数 LSE 增量项 |
| `torch.exp(S_i - S_global)` | $\exp(S^{(b)} - S_{\text{global}})$ | 全局合并权重 |


---


### 3.3 两种方法的特性对比


| 特性 | 传统方法（m/l/o） | LSE 方法（S/o） |
|------|------------------|----------------|
| 每个 Tile 存储量 | 3 个量：$m^{(b)}, \ell^{(b)}, \mathbf{o}^{(b)}$ | 2 个量：$S^{(b)}, \mathbf{o}^{(b)}$ |
| 通信量 | 更多（3 个数/Tile） | 更少（2 个数/Tile） |
| 全局合并公式 | 需先求 $m_{\text{global}}$，再做指数缩放 | 直接在对数空间迭代 LSE |
| 数值稳定性 | 需要显式处理 $m^{(b)} - m_{\text{global}}$ | 自然包含在 LSE 迭代中 |
| 代码复杂度 | 需要分别处理 m、l、o 三个量 | 统一处理 S 和 o，更简洁 |
| 与标准 Attention 等价性 | ✓（2.2.3 节证明） | ✓（2.3.6 节证明） |
| 两种方法间等价性 | ✓（2.4 节证明） | ✓（2.4 节证明） |
| 适用场景 | 通用 | 通信受限场景更优 |


---


## 四、涉及的基本数学知识清单


| 概念名称 | 在本推导中的具体作用 | 一句话定义或公式表达 |
|---------|---------------------|---------------------|
| Softmax 函数 | 将注意力分数转换为概率分布 | $\mathrm{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}$ |
| 缩放点积注意力 | 定义 Query 与 KV 的交互方式 | $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^T/\sqrt{d})V$ |
| Max-Shift 数值稳定 | 避免 softmax 计算中的指数溢出 | $\mathrm{softmax}(x_i) = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}$，$m = \max_j x_j$ |
| Online Softmax | 支持增量式 softmax 计算 | 递推维护 $m_j = \max(m_{j-1}, x_j)$ 和 $\ell_j = \ell_{j-1} \cdot \exp(m_{j-1} - m_j) + \exp(x_j - m_j)$ |
| Log-Sum-Exp (LSE) | 将对数空间的多个值合并为一个标量 | $\text{LSE}(\mathbf{x}) = \log(\sum_i \exp(x_i))$ |
| Safe LSE | 数值稳定的 LSE 计算 | $\text{LSE}(\mathbf{x}) = m + \log(\sum_i \exp(x_i - m))$，$m = \max_i x_i$ |
| LSE 合并算子 $\oplus$ | 将两个局部量合并为全局量 | $[\mathbf{o}_1, S_1] \oplus [\mathbf{o}_2, S_2] = [\frac{e^{S_1}\mathbf{o}_1 + e^{S_2}\mathbf{o}_2}{e^{S_1} + e^{S_2}}, \log(e^{S_1} + e^{S_2})]$ |
| 结合律 | 保证多 Tile 合并顺序不影响结果 | $(A \oplus B) \oplus C = A \oplus (B \oplus C)$ |
| 指数乘法恒等式 | 将局部坐标系的指数值转换到全局坐标系 | $\exp(a - c) = \exp(a - b) \cdot \exp(b - c)$ |
| 对数运算性质 | LSE 推导的核心代数依据 | $\log(a \cdot b) = \log(a) + \log(b)$ |


---


## 五、总结


FlashDecoding 是一种针对 LLM 推理**解码阶段**的高效注意力算法。本文档完整推导了两种数学上等价的实现方法，并分别证明了它们与标准 Attention 的等价性：


### 5.1 传统方法（m/l/o 三元组法）

- **局部计算**：每个 Tile 计算 $(m^{(b)}, \ell^{(b)}, \mathbf{o}^{(b)})$
- **全局合并**：
  $$m_{\text{global}} = \max_b m^{(b)}, \quad \ell_{\text{global}} = \sum_b \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)}, \\ \quad \mathbf{o}_{\text{final}} = \frac{\sum_b \exp(m^{(b)} - m_{\text{global}}) \cdot \ell^{(b)} \cdot \mathbf{o}^{(b)}}{\ell_{\text{global}}}$$


### 5.2 LSE 方法（S/o 二元组法）

- **局部计算**：每个 Tile 计算 $(S^{(b)}, \mathbf{o}^{(b)})$，其中 $S^{(b)} = m^{(b)} + \log(\ell^{(b)})$
- **全局合并**：
  $$S_{\text{global}} = \log\left(\sum_b \exp(S^{(b)})\right), \quad \mathbf{o}_{\text{final}} = \sum_b \exp(S^{(b)} - S_{\text{global}}) \cdot \mathbf{o}^{(b)}$$
- **合并算子**：$[\mathbf{o}_1, S_1] \oplus [\mathbf{o}_2, S_2] = [\frac{e^{S_1}\mathbf{o}_1 + e^{S_2}\mathbf{o}_2}{e^{S_1} + e^{S_2}}, \log(e^{S_1} + e^{S_2})]$


### 5.3 等价性结论

- **传统方法 $\Leftrightarrow$ 标准 Attention**：已证（2.2.3 节）
- **LSE 方法 $\Leftrightarrow$ 标准 Attention**：已证（2.3.6 节）
- **传统方法 $\Leftrightarrow$ LSE 方法**：已证（2.4 节），核心恒等式为 $S^{(b)} = m^{(b)} + \log(\ell^{(b)})$


### 5.4 工程实践建议

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| 一般推理加速 | 传统方法 | 实现成熟，生态完善 |
| 通信受限（多机/跨节点） | LSE 方法 | 通信量减少 33%，更简洁 |
| Ring Attention 序列并行 | LSE 方法 | 结合律保证任意合并顺序正确 |
| CUDA Kernel 手写优化 | 传统方法 | 与 FlashAttention 原版一致 |


该算法已被集成到 FlashAttention 2.2+、xFormers、FlashInfer 等主流推理加速库中，成为长序列 LLM 推理的标准优化技术。


---
