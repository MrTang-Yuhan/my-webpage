---
layout: post.njk
post_id: 2026-06-23-retrieval-augmented-generation-for-knowledge-intensive-nlp-tasks
archive: agent-memory
title: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
date: 2026-06-23
tags:
  - post
---
# 论文链接

[lewis2020retrieval, Retrieval-augmented generation for knowledge-intensive nlp tasks](https://arxiv.org/abs/2005.11401)


# RAG (Retrieval-Augmented Generation) 笔记整理

## 一、符号定义

| 符号 | 含义 |
|------|------|
| $x$ | 输入问题 |
| $y$ | 输出答案序列 |
| $y_i$ | 答案序列中第 $i$ 个 token |
| $z$ | 检索到的文档（passage） |
| $N$ | 输出答案 $y$ 的长度（token 数） |
| $\eta$ | Retriever（检索器）的参数 |
| $\theta$ | Generator（生成器）的参数 |

## 二、概率定义

- **$P_\eta(z\|x)$**：在问题 $x$ 下，检索器 Retriever 给出文档 $z$ 的概率
- **$P_\theta(y\|x,z)$**：生成器 Generator 在给定 $x$ 和 $z$ 条件下，完整生成 $y$ 的概率
- **$P_\theta(y_i\|x, z, y_{1:i-1})$**：生成器生成第 $i$ 个 token 的条件概率（基于输入 $x$、文档 $z$、以及已生成的前 $i-1$ 个 token）

## 三、完整形式（边缘化）

在 RAG 中，我们不知道应该用哪个文档 $z$，想知道答案 $y$ 的总概率。根据**全概率公式**（边缘化），对所有可能文档求和：

$$
\begin{aligned}
P(y|x) = \sum_{z \in \text{all docs}}P(y,z|x) =
\sum_{z \in \text{all docs}} P_\eta(z|x) \cdot P_\theta(y|x,z)
\end{aligned}
$$

### Top-K 近似

理论上边缘化需对所有文档求和，但计算开销太大，因此只考虑最相关的前 $K$ 个文档：

$$P(y|x) \approx \sum_{z \in \text{top-}k(P_\eta(\cdot|x))} P_\eta(z|x) \cdot P_\theta(y|x,z)$$

## 四、链式法则（序列分解）

生成器对完整序列的生成概率可通过链式法则分解为每个 token 条件概率的乘积：

$$P_\theta(y|x,z) = \prod_{i=1}^{N} P_\theta(y_i | x, z, y_{1:i-1})$$

其中 $y_{1:i-1} = (y_1, y_2, \ldots, y_{i-1})$ 表示已生成的前缀序列。

## 五、RAG 的两种模型变体

论文提出了两种边缘化隐文档的不同方式：

### 5.1 RAG-Sequence

**核心思想**：使用**同一个**检索到的文档来生成整个目标序列。

将检索文档视为**单个隐变量**，通过 Top-K 近似边缘化：

$$ 
P_{\text{RAG-Sequence}}(y|x) \approx \sum_{z \in \text{top-}k} P_\eta(z|x) \cdot \underbrace{\prod_{i=1}^{N} P_\theta(y_i|x,z,y_{1:i-1})}_{P_\theta(y|x,z)} 
$$

> 即：先对每个文档用链式法则计算完整的 $P_\theta(y\|x,z)$，再对所有 Top-K 文档按 $P_\eta(z\|x)$ 加权求和。

### 5.2 RAG-Token

**核心思想**：每个目标 token 可以基于**不同的**文档来生成。

对每个生成位置 $i$，分别检索并边缘化：

$$P_{\text{RAG-Token}}(y|x) \approx \prod_{i=1}^{N} \sum_{z \in \text{top-}k} P_\eta(z|x) \cdot P_\theta(y_i|x,z,y_{1:i-1})$$

> 即：**求和在积内**——每个 token 的生成都独立对所有 Top-K 文档做边缘化，允许模型在不同位置从多个文档中聚合信息。

### 两者对比

| 特性 | RAG-Sequence | RAG-Token |
|------|-------------|-----------|
| 文档使用方式 | 整个序列使用同一个文档 | 每个 token 可使用不同文档 |
| 边缘化位置 | 求和在积外（对整个序列） | 求和在积内（对每个 token） |
| 适用场景 | 答案来自单一文档的任务 | 需要多文档信息融合的任务 |
| 解码复杂度 | 较高（需 Thorough/Fast Decoding） | 较低（标准 beam search） |

---

## 六、Retriever: DPR（稠密段落检索）

### 6.1 概率形式

检索组件基于 DPR（Dense Passage Retrieval），采用**双编码器**（bi-encoder）架构：

$$P_\eta(z|x) \propto \exp\big(d(z)^\top q(x)\big)$$

其中：
- $d(z) = \text{BERT}_d(z)$：文档编码器，将文档 $z$ 编码为稠密向量
- $q(x) = \text{BERT}_q(x)$：查询编码器，将问题 $x$ 编码为稠密向量

### 6.2 关于 softmax

由于概率需满足 $\geq 0$ 且总和为 $1$，实际使用 **softmax** 归一化：

$$P_\eta(z|x) = \frac{\exp(d(z)^\top q(x))}{\sum_{z'} \exp(d(z')^\top q(x))}$$

### 6.3 MIPS（最大内积搜索）

找概率最大的 Top-K 文档，等价于找**内积最大**的 Top-K 文档：

$$\underset{z}{\text{top-}k}\; P_\eta(z|x) \;\Longleftrightarrow\; \underset{z}{\text{top-}k}\; d(z)^\top q(x)$$

> **原因**：指数函数 $\exp(\cdot)$ 是单调递增的，最大化概率等价于最大化内积。

### 6.4 内积与相似度的几何关系

两个向量的内积与夹角余弦的关系：

$$a^\top b = \|a\| \|b\| \cos\theta$$

内积越大 → $\cos\theta$ 越大 → 两向量越相似（方向越接近）。

因此，MIPS 本质上是在向量空间中找与查询 $q(x)$ 方向最接近的文档向量 $d(z)$。

### 6.5 非参数记忆（Non-parametric Memory）

- 文档索引被称为**非参数记忆**
- 检索时使用 FAISS 等近似最近邻搜索库，在次线性时间内完成 MIPS
- 文档编码器（及索引）在训练时通常**固定不动**，只微调查询编码器和 BART 生成器

---

## 七、Generator: BART

- 使用 **BART-large**（400M 参数）作为 seq2seq 生成器
- 将输入 $x$ 与检索内容 $z$ **拼接**后输入 BART
- BART 的参数 $\theta$ 被称为**参数记忆**（parametric memory）
- 生成时采用自回归方式，逐 token 生成答案

---

## 八、训练目标（Training Objective）

### 8.1 端到端联合训练

RAG 采用 **端到端联合训练**（end-to-end joint training），同时优化检索器（Retriever）和生成器（Generator），但**不需要**关于"应该检索哪个文档"的直接监督信号。

给定微调训练语料库中的输入/输出对 $(x_j, y_j)$，训练目标为**最小化负边缘对数似然**（negative marginal log-likelihood）：

$$\mathcal{L} = -\log P(y_j|x_j)$$

展开后：

$$\mathcal{L} = -\log \left( \sum_{z \in \text{top-}k} P_\eta(z|x_j) \cdot P_\theta(y_j|x_j, z) \right)$$

> **关键理解**：模型通过答案 $y_j$ 的监督信号，间接学习检索哪些文档 $z$ 对生成目标答案最有帮助。这是一种**弱监督**（weak supervision）或**隐变量学习**的设置。

### 8.2 参数更新策略

| 组件 | 是否训练 | 说明 |
|------|----------|------|
| 文档编码器 $\text{BERT}_d$ | **固定** | 更新成本极高，需定期重建文档索引。实验表明不更新也能达到强性能 |
| 查询编码器 $\text{BERT}_q$ | **微调** | 学习将查询 $x$ 映射到更适合任务的表示 |
| BART 生成器 | **微调** | 学习在给定检索文档的条件下生成答案 |

训练使用 **Adam** 优化器进行随机梯度下降。

---

## 九、解码策略

省略了解码策略，因为感觉有些过时了。
