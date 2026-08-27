---
layout: post.njk
post_id: 2026-08-27-解析-flashattention-4-flashattention-v2
archive: llm推理框架
title: 解析 FlashAttention（4）：FlashAttention-v2 （未完成）
date: 2026-08-27
tags:
  - post
---


以下是基于 FlashAttention v1/v2 论文及博客解析的完整分析笔记。文中引用的图与表均明确标注其来源论文，并附上对应 arXiv 链接。

**论文链接：**
- FlashAttention v1：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022) —— https://arxiv.org/abs/2205.14135
- FlashAttention v2：*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (arXiv 2023) —— https://arxiv.org/abs/2307.08691

---

## 一、FlashAttention v1 核心回顾

在对比 v2 之前，有必要先明确 v1 的核心机制，因为 v2 的所有改进都建立在 v1 的 Tiling 与 Online Softmax 框架之上。

### 1.1 前向传播

标准 Attention 的核心矛盾在于必须将中间矩阵 $S, P \in \mathbb{R}^{N \times N}$ 写入 HBM，导致内存占用和 HBM 访问量均为 $O(N^2)$。v1 的解法是将 $Q, K, V$ 切分为足够小的块加载到 SRAM，通过 Online Softmax 维护全局统计量 $(m, \ell)$（行最大值与指数和），增量地更新输出 $O$。

v1 论文中的 **Algorithm 1** 给出了完整的 Tiling 前向流程。[博客](https://my-webpage-adu.pages.dev/posts/llm%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6/2026-08-20-%E8%A7%A3%E6%9E%90-flashattention-2-flashattention-v1-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD/) 中将其核心增量更新式推导为：

$$O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1}\left(\text{diag}(\ell_i)e^{m_i - m_i^{\text{new}}}O_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}}\tilde{P}_{ij}V_j\right)$$

其特点是每处理一个 $(K_j, V_j)$ 块，就必须对旧输出 $O_i$ 和新贡献分别进行 rescaling（乘以指数修正因子并除以新的 $\ell$）。这意味着内层循环中包含了大量逐元素的乘除法。

### 1.2 反向传播

标准反向传播需要保存 $P \in \mathbb{R}^{N \times N}$ 来计算 $dV, dP, dS$。v1 的解法是不保存 $P$，而是保存前向的 $(m, \ell)$ 和输出 $O$，在反向时重新加载 $Q_i, K_j$ 到 SRAM 中重算 $P_{ij}$。

v1 论文中的 **Algorithm 4** 描述了完整的反向分块算法。[博客](https://my-webpage-adu.pages.dev/posts/llm%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6/2026-08-20-%E8%A7%A3%E6%9E%90-flashattention-3-flashattention-v1-%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD/) 中详细推导了 softmax 梯度的关键简化：$D_i = \text{rowsum}(dO_i \circ O_i)$，这使得计算 $D_i$ 无需遍历 $N$ 维的 $P_i$，仅需两个长度为 $d$ 的向量做点积。反向的外层循环遍历 $K_j, V_j$，内层循环遍历 $Q_i$，梯度 $dK_j, dV_j$ 在 SRAM 中局部累加后再写回 HBM。

---

## 二、FlashAttention-2 相比 v1 的三大核心变化

v2 论文（Section 3）指出，v1 虽然比标准实现快 2–4 倍，但前向仅达到理论峰值 FLOPs/s 的 30–50%，反向更是只有 25–35%。v2 通过算法微调、并行化扩展和 Warp 工作划分三个层面的改进，将利用率提升至 50–73%（A100），实现约 2 倍加速。

### 变化 1：算法微调 —— 减少非 Matmul FLOPs

现代 GPU 上，非矩阵乘法操作（逐元素乘、指数、除法等）与 Matmul 的吞吐量差距极大。以 A100 为例，FP16/BF16 Matmul 的理论峰值约为 312 TFLOPs/s，而 FP32 非 Matmul 仅约 19.5 TFLOPs/s，单个非 matmul FLOP 的成本可达 matmul FLOP 的 16 倍。因此，减少非 matmul 操作、让 GPU 尽可能多地时间在 Tensor Core 上做矩阵乘，是提升利用率的关键。

v2 对此做了两处算法层面的调整：

**第一，延迟输出归一化。** v1 在每轮内层循环结束时都执行完整的 rescaling：先将旧输出 $O_i$ 乘以指数修正因子，再加上新块的贡献，最后除以新的全局指数和 $\ell_i^{\text{new}}$。v2 改为维护一个 "un-scaled" 的输出 $\tilde{O}_i$，在内层循环中仅做累加和指数修正，直到所有 key/value 块处理完毕，才在最后统一除以 $\text{diag}(\ell_i^{(T_c)})^{-1}$ 得到正确输出。这大幅减少了循环内部的逐元素除法。

**第二，合并保存的统计量。** v1 在反向传播时需要分别保存行最大值 $m$ 和行指数和 $\ell$。v2 改为只保存 **logsumexp** $L = m + \log(\ell)$。这不仅减少了额外的内存占用，也简化了反向传播中的非 matmul 计算。

v2 论文中的 **Algorithm 1** 体现了这些调整。其 un-scaled 输出的增量更新式为：

$$\tilde{O}_i^{(j)} = \text{diag}(e^{m_i^{(j-1)} - m_i^{(j)}})^{-1}\tilde{O}_i^{(j-1)} + \tilde{P}_{ij}V_j$$

注意此处不再除以 $\ell$，而是累积未归一化的结果，仅在循环末尾统一归一化。

[](img/flash-atten-v2-algo1.png)

### 变化 2：并行化策略 —— 沿序列长度维度切分

v1 仅在 batch size 和 number of heads 两个维度上并行化。当序列很长时，batch size 和 head 数往往较小，导致 GPU 的 Streaming Multiprocessors (SMs) 占用率（occupancy）不足，大量计算资源空闲。

v2 增加了对序列长度维度的并行化：

**在前向传播中**，v1 的外层循环遍历 key/value 块 $K_j$，内层循环遍历 query 块 $Q_i$。v2 将循环顺序调换：外层循环遍历 query 的行块 $Q_i$，内层循环遍历 key/value 的列块 $K_j$。由于不同 query 行块之间的计算是完全独立的，可以将每个行块分配给一个独立的 thread block，无需任何跨 block 通信。这显著提升了长序列场景下的 SM 占用率。

**在反向传播中**，类似地将外层循环（遍历 key/value 的列块 $K_j$）并行化。每个 thread block 负责一个列块，计算该列块对 $dK, dV$ 的贡献。由于 $dQ$ 需要累加所有列块的贡献，v2 使用 **atomic adds** 来协调不同 thread block 对 $dQ$ 的更新。

v2 论文中的 **Figure 2** 直观展示了这一并行化方案：左图（Forward）显示每个 Worker（thread block）负责注意力矩阵的一个行块；右图（Backward）显示每个 Worker 负责一个列块。

![](img/flash-atten-v2-fig2.png)

### 变化 3：Warp 级别工作划分 —— 避免 Split-K

即使在同一个 thread block 内部，v1 的 warp 划分方式也存在效率瓶颈。

v1 采用的是 **Split-K** 方案。它将 $K$ 和 $V$ 切分到 4 个 warp，而 $Q$ 被所有 warp 共享。每个 warp 计算自己负责的 $K/V$ 切片与 $Q$ 的乘积，得到部分中间结果。由于最终输出需要累加所有 warp 的贡献，这些中间结果必须被写入 shared memory，经过同步后，再由其他 warp 读取并加和。这带来了大量的 shared memory 读写和同步开销。

v2 改为将 **$Q$ 切分到 4 个 warp**，而 $K$ 和 $V$ 被所有 warp 共享。每个 warp 负责计算自己的 $Q_i$ 切片与完整的 $K^\top$ 的乘积，然后直接与对应的 $V$ 切片相乘，得到该 warp 对应的输出切片。由于每个 warp 的输入 $Q_i$ 和输出 $O_i$ 切片是独立的，**warp 之间无需通信**，完全避免了 shared memory 上的中间结果读写。

v2 论文中的 **Figure 3** 清晰对比了两种划分策略：子图 (a) 展示了 FlashAttention v1 的 Split-K 方案（$K^T$ 和 $V$ 被 split 到 Warp 1-4，$Q$ 共享，需要通信累加）；子图 (b) 展示了 FlashAttention-2 的方案（$Q$ 被 split 到 Warp 1-4，$K^T$ 和 $V$ 共享，无通信）。

![](img/flash-atten-v2-fig3.png)

---

## 三、FlashAttention-2 关键公式推导

以下推导沿用博客符号体系。先以极小维度 $N=2, d=3$ 为例，从元素级别展示 v2 的核心机制，再泛化至一般场景。

### 3.1 v2 前向传播：un-scaled 输出的增量更新

**具体例子设定：** 令序列长度 $N=2$，特征维度 $d=3$，分块尺寸 $B_r = B_c = 1$（每个块恰好包含一行）。输入矩阵为：

$$Q = \begin{bmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \end{bmatrix} \in \mathbb{R}^{2 \times 3}, \quad K = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \end{bmatrix} \in \mathbb{R}^{2 \times 3}, \quad V = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \end{bmatrix} \in \mathbb{R}^{2 \times 3}$$

其中 $\mathbf{q}_i, \mathbf{k}_j, \mathbf{v}_j \in \mathbb{R}^{1 \times 3}$ 均为行向量。以第 $i=1$ 个 query 块（即 $\mathbf{q}_1$）为例，展示其输出如何通过 v2 的 un-scaled 累加方式逐步构造。

**变量声明（针对此例子，仅考虑第 $i=1$ 个 query 块）：**
- $S_{1j} = \mathbf{q}_1 \mathbf{k}_j^\top \in \mathbb{R}$：标量，第 1 个 query 与第 $j$ 个 key 的内积。
- $m^{(j)} \in \mathbb{R}$：处理完前 $j$ 个 key 后，当前全局最大值。初始值 $m^{(0)} = -\infty$。
- $\ell^{(j)} \in \mathbb{R}$：处理完前 $j$ 个 key 后，以 $m^{(j)}$ 为基准的全局指数和。初始值 $\ell^{(0)} = 0$。
- $O^{(j)} \in \mathbb{R}^{1 \times 3}$：处理完前 $j$ 个 key 后的 **un-scaled** 累积输出。初始值 $O^{(0)} = \mathbf{0} \in \mathbb{R}^{1 \times 3}$。

**第一步（$j=1$，处理 $\mathbf{k}_1, \mathbf{v}_1$）：**

计算局部分数：
$$S_{11} = \mathbf{q}_1 \mathbf{k}_1^\top \in \mathbb{R}$$

更新全局最大值：
$$m^{(1)} = \max(m^{(0)}, S_{11}) = \max(-\infty, S_{11}) = S_{11}$$

计算局部平移指数（标量）：
$$\tilde{P}_{11} = \exp(S_{11} - m^{(1)}) = \exp(0) = 1$$

更新全局指数和（见 [解析 FlashAttention（2）：FlashAttention-v1 前向传播](https://my-webpage-adu.pages.dev/posts/llm%E6%8E%A8%E7%90%86%E6%A1%86%E6%9E%B6/2026-08-20-%E8%A7%A3%E6%9E%90-flashattention-2-flashattention-v1-%E5%89%8D%E5%90%91%E4%BC%A0%E6%92%AD/) 公式 35）：
$$\ell^{(1)} = \exp(m^{(0)} - m^{(1)}) \cdot \ell^{(0)} + \tilde{P}_{11} = \exp(-\infty - S_{11}) \cdot 0 + 1 = 1$$

更新 un-scaled 输出。由于 $O^{(0)} = \mathbf{0}$，rescaling 项消失：
$$O^{(1)} = \exp(m^{(1)} - m^{(0)}) \cdot O^{(0)} + \tilde{P}_{11} \mathbf{v}_1 = \mathbf{0} + 1 \cdot \mathbf{v}_1 = \mathbf{v}_1 \in \mathbb{R}^{1 \times 3}$$

此时 $O^{(1)}$ 是 un-scaled 的，其值等于 $\exp(S_{11} - m^{(1)}) \mathbf{v}_1$。

**第二步（$j=2$，处理 $\mathbf{k}_2, \mathbf{v}_2$）：**

计算局部分数：
$$S_{12} = \mathbf{q}_1 \mathbf{k}_2^\top \in \mathbb{R}$$

更新全局最大值：
$$m^{(2)} = \max(m^{(1)}, S_{12}) \in \mathbb{R}$$

此处需分两种情况讨论，以展示 rescaling 的核心作用。

**情况 A：$S_{12} \leq S_{11}$（全局最大值未变，$m^{(2)} = m^{(1)} = S_{11}$）**

计算局部平移指数：
$$\tilde{P}_{12} = \exp(S_{12} - m^{(2)}) = \exp(S_{12} - S_{11})$$

更新全局指数和：
$$\ell^{(2)} = \exp(m^{(1)} - m^{(2)}) \cdot \ell^{(1)} + \tilde{P}_{12} = 1 \cdot 1 + \exp(S_{12} - S_{11}) = 1 + \exp(S_{12} - S_{11})$$

更新 un-scaled 输出。由于 $m^{(2)} = m^{(1)}$，rescaling 因子为 $\exp(m^{(1)} - m^{(2)}) = 1$：
$$O^{(2)} = 1 \cdot O^{(1)} + \tilde{P}_{12} \mathbf{v}_2 = \mathbf{v}_1 + \exp(S_{12} - S_{11}) \mathbf{v}_2 \in \mathbb{R}^{1 \times 3}$$

执行最终归一化：
$$O_1 = \frac{1}{\ell^{(2)}} O^{(2)} = \frac{\mathbf{v}_1 + \exp(S_{12} - S_{11}) \mathbf{v}_2}{1 + \exp(S_{12} - S_{11})}$$

验证正确性：标准 softmax 概率为
$$P_1 = \frac{\exp(S_{11} - S_{11})}{\exp(S_{11} - S_{11}) + \exp(S_{12} - S_{11})} = \frac{1}{1 + \exp(S_{12} - S_{11})}$$
$$P_2 = \frac{\exp(S_{12} - S_{11})}{1 + \exp(S_{12} - S_{11})}$$

标准 Attention 输出为 $P_1 \mathbf{v}_1 + P_2 \mathbf{v}_2$，与上述 $O_1$ 完全一致。

**情况 B：$S_{12} > S_{11}$（全局最大值更新，$m^{(2)} = S_{12}$）**

计算局部平移指数：
$$\tilde{P}_{12} = \exp(S_{12} - m^{(2)}) = \exp(0) = 1$$

更新全局指数和：
$$\ell^{(2)} = \exp(m^{(1)} - m^{(2)}) \cdot \ell^{(1)} + \tilde{P}_{12} = \exp(S_{11} - S_{12}) \cdot 1 + 1 = \exp(S_{11} - S_{12}) + 1$$

更新 un-scaled 输出。此处 rescaling 因子为 $\exp(m^{(1)} - m^{(2)}) = \exp(S_{11} - S_{12}) < 1$，用于将旧输出 $O^{(1)}$ 的指数基准从 $m^{(1)}$ 修正到 $m^{(2)}$：
$$O^{(2)} = \exp(S_{11} - S_{12}) \cdot O^{(1)} + \tilde{P}_{12} \mathbf{v}_2 = \exp(S_{11} - S_{12}) \mathbf{v}_1 + \mathbf{v}_2 \in \mathbb{R}^{1 \times 3}$$

执行最终归一化：
$$O_1 = \frac{1}{\ell^{(2)}} O^{(2)} = \frac{\exp(S_{11} - S_{12}) \mathbf{v}_1 + \mathbf{v}_2}{\exp(S_{11} - S_{12}) + 1}$$

验证正确性：标准 softmax 概率为
$$P_1 = \frac{\exp(S_{11} - S_{12})}{\exp(S_{11} - S_{12}) + \exp(S_{12} - S_{12})} = \frac{\exp(S_{11} - S_{12})}{\exp(S_{11} - S_{12}) + 1}$$
$$P_2 = \frac{1}{\exp(S_{11} - S_{12}) + 1}$$

标准 Attention 输出为 $P_1 \mathbf{v}_1 + P_2 \mathbf{v}_2$，与上述 $O_1$ 完全一致。

**泛化至一般场景：**

设一般维度下，$Q, K, V \in \mathbb{R}^{N \times d}$，分块尺寸为 $B_r$（query 块）和 $B_c$（key/value 块）。定义：
- $Q_i \in \mathbb{R}^{B_r \times d}$：第 $i$ 个 query 块，$i = 1, \dots, T_r$，$T_r = \lceil N / B_r \rceil$。
- $K_j, V_j \in \mathbb{R}^{B_c \times d}$：第 $j$ 个 key/value 块，$j = 1, \dots, T_c$，$T_c = \lceil N / B_c \rceil$。
- $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$。
- $m_i^{(j)} \in \mathbb{R}^{B_r}$：第 $i$ 个 query 块处理完前 $j$ 个 key 块后的全局行最大值，初始 $m_i^{(0)} = (-\infty)^{B_r}$。
- $\ell_i^{(j)} \in \mathbb{R}^{B_r}$：对应的全局行指数和，初始 $\ell_i^{(0)} = 0^{B_r}$。
- $O_i^{(j)} \in \mathbb{R}^{B_r \times d}$：un-scaled 累积输出，初始 $O_i^{(0)} = 0^{B_r \times d}$。

对 $j = 1, \dots, T_c$，依次执行：

1. 计算局部分数：$S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$。
2. 更新全局行最大值：$m_i^{(j)} = \max\left(m_i^{(j-1)},\ \text{rowmax}(S_{ij})\right) \in \mathbb{R}^{B_r}$，其中 $\text{rowmax}(\cdot)$ 对每行取最大。
3. 计算局部平移指数：$\tilde{P}_{ij} = \exp\left(S_{ij} - m_i^{(j)}\right) \in \mathbb{R}^{B_r \times B_c}$（逐行广播减法）。
4. 更新全局指数和：$\ell_i^{(j)} = \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) \odot \ell_i^{(j-1)} + \text{rowsum}\left(\tilde{P}_{ij}\right) \in \mathbb{R}^{B_r}$。
5. 更新 un-scaled 输出：
   $$O_i^{(j)} = \text{diag}\left(\exp\left(m_i^{(j-1)} - m_i^{(j)}\right)\right) O_i^{(j-1)} + \tilde{P}_{ij} V_j \in \mathbb{R}^{B_r \times d}$$
   此处 $\text{diag}(v)$ 以向量 $v \in \mathbb{R}^{B_r}$ 构造 $B_r \times B_r$ 对角矩阵，左乘该矩阵即将 $O_i^{(j-1)}$ 的每一行 $r$ 乘以标量 $\exp\left(m_{i,r}^{(j-1)} - m_{i,r}^{(j)}\right)$，实现逐行独立的基准修正。

6. 循环结束后，执行最终归一化：
   $$O_i = \text{diag}\left(\ell_i^{(T_c)}\right)^{-1} O_i^{(T_c)} \in \mathbb{R}^{B_r \times d}$$

**正确性（归纳简述）：** 需证 $O_i^{(j)} = \sum_{t=1}^{j} \exp\left(S_{it} - m_i^{(j)}\right) V_t$。基例 $j=1$ 直接成立。归纳步骤中，旧项 $\text{diag}\left(\exp\left(m_i^{(j-1)} - m_i^{(j)}\right)\right) O_i^{(j-1)}$ 将前 $j-1$ 项的指数基准从 $m_i^{(j-1)}$ 修正至 $m_i^{(j)}$，新项 $\tilde{P}_{ij} V_j = \exp\left(S_{ij} - m_i^{(j)}\right) V_j$ 直接以新基准加入。最终除以 $\ell_i^{(T_c)}$ 即得标准 softmax attention 输出。

### 3.2 v2 保存的统计量：logsumexp $L_i$

v2 不再分别保存 $m_i$ 和 $\ell_i$，而是保存 logsumexp $L_i = m_i^{(T_c)} + \log\left(\ell_i^{(T_c)}\right)$。

**具体例子（$N=2, d=3$）：** 对第 $i=1$ 行，设前向结束后：
- $m = m^{(2)} \in \mathbb{R}$：最终全局最大值。
- $\ell = \ell^{(2)} \in \mathbb{R}$：最终全局指数和。

则保存的标量为：
$$L_1 = m + \log(\ell) \in \mathbb{R}$$

**反向传播中恢复 $P_{1j}$：**

v2 Algorithm 2 第 11 行利用 $L_1$ 重计算概率。对任意 key 块 $j$，先重算分数 $S_{1j} = \mathbf{q}_1 \mathbf{k}_j^\top \in \mathbb{R}$，则：

$$P_{1j} = \exp\left(S_{1j} - L_1\right) = \exp\left(S_{1j} - m - \log(\ell)\right) = \exp\left(S_{1j} - m\right) \cdot \exp\left(-\log(\ell)\right) = \frac{\exp\left(S_{1j} - m\right)}{\ell}$$

这正是全局 softmax 概率：分子为以最终全局最大值 $m$ 为基准的平移指数，分母为全局指数和 $\ell$。

**泛化至一般场景：**

对第 $i$ 个 query 块，保存 $L_i = m_i^{(T_c)} + \log\left(\ell_i^{(T_c)}\right) \in \mathbb{R}^{B_r}$。反向时重算：

$$P_{ij} = \exp\left(S_{ij} - L_i\right) \in \mathbb{R}^{B_r \times B_c}$$

其中 $S_{ij} = Q_i K_j^\top$，减法为逐行广播（$S_{ij}$ 的每一行减去 $L_i$ 的对应元素）。代入 $L_i$ 定义：

$$P_{ij} = \exp\left(S_{ij} - m_i^{(T_c)}\right) \odot \text{diag}\left(\ell_i^{(T_c)}\right)^{-1}$$

即先以最终全局最大值平移，再逐行除以全局指数和，与分别保存 $m, \ell$ 的结果完全一致，但内存占用和反向计算量均减少。

[](img/flash-atten-v2-algo2.png)

### 3.3 v2 反向传播：$D_i$ 的代数简化

在标准反向传播中，softmax 的 Jacobian 引入标量 $D_i$，其原始定义需要遍历整行概率向量。

**具体例子（$N=2, d=3$）：**

设全局矩阵 $P, dP \in \mathbb{R}^{2 \times 2}$，$O, dO \in \mathbb{R}^{2 \times 3}$。对第 $i=1$ 行，标准 softmax 梯度为：

$$dS_{1j} = P_{1j} \left(dP_{1j} - D_1\right), \quad j=1,2$$

其中 $D_1$ 的定义为：
$$D_1 = \sum_{c=1}^{2} P_{1c}\, dP_{1c}$$

此定义需要存储并遍历 $P$ 的第 1 行（长度 $N=2$）。FlashAttention 的简化如下：

由 $O = PV$，第 1 行第 $t$ 个元素为：
$$O_{1t} = \sum_{c=1}^{2} P_{1c} V_{ct}, \quad t=1,2,3$$

由 $dP = dO\, V^\top$，第 1 行第 $c$ 个元素为：
$$dP_{1c} = \sum_{t=1}^{3} dO_{1t} V_{ct}, \quad c=1,2$$

将 $dP_{1c}$ 代入 $D_1$ 的定义：
$$D_1 = \sum_{c=1}^{2} P_{1c} \left(\sum_{t=1}^{3} dO_{1t} V_{ct}\right) = \sum_{t=1}^{3} dO_{1t} \left(\sum_{c=1}^{2} P_{1c} V_{ct}\right)$$

注意到括号内即为 $O_{1t}$，因此：
$$D_1 = \sum_{t=1}^{3} dO_{1t} O_{1t}$$

**泛化至一般场景：**

对第 $i$ 个 query 块，$dO_i, O_i \in \mathbb{R}^{B_r \times d}$，则：
$$D_i = \text{rowsum}\left(dO_i \circ O_i\right) \in \mathbb{R}^{B_r}$$

其中 $\circ$ 为 Hadamard 积（逐元素相乘），$\text{rowsum}(\cdot)$ 对每行求和得到列向量。计算 $D_i$ 无需访问 $P$ 的任何元素，仅需 $dO_i$ 与 $O_i$ 做点积，复杂度 $O(B_r d)$，完全在 SRAM 内完成。

### 3.4 v2 反向传播：$dS_{ij}$ 的分块计算

**具体例子（$N=2, d=3$）：**

对块 $(i=1, j=1)$，已知 $P_{11} \in \mathbb{R}$，$dP_{11} \in \mathbb{R}$，$D_1 \in \mathbb{R}$。分块 softmax 梯度为：

$$dS_{11} = P_{11} \circ \left(dP_{11} - D_1\right) \in \mathbb{R}$$

其中减法为标量减法。同理对 $j=2$：
$$dS_{12} = P_{12} \circ \left(dP_{12} - D_1\right) \in \mathbb{R}$$

**泛化至一般场景：**

对第 $i$ 个 query 块和第 $j$ 个 key/value 块：
$$dS_{ij} = P_{ij} \circ \left(dP_{ij} - D_i \mathbf{1}_{B_c}^\top\right) \in \mathbb{R}^{B_r \times B_c}$$

其中 $dP_{ij} = dO_i V_j^\top \in \mathbb{R}^{B_r \times B_c}$，$D_i \in \mathbb{R}^{B_r}$，$\mathbf{1}_{B_c}^\top \in \mathbb{R}^{1 \times B_c}$ 为全 1 行向量。外积 $D_i \mathbf{1}_{B_c}^\top \in \mathbb{R}^{B_r \times B_c}$ 的每一行均为 $D_i$ 的对应元素，实现对 $dP_{ij}$ 的逐行广播减法。

### 3.5 v2 反向传播：$dQ_i$ 与 $dK_j$ 的分块累加

**具体例子（$N=2, d=3$）：**

设缩放常数 $\tau \in \mathbb{R}$。由 $S = \tau Q K^\top$，链式法则给出：

$$dQ_1 = \tau \left(dS_{11} \mathbf{k}_1 + dS_{12} \mathbf{k}_2\right) \in \mathbb{R}^{1 \times 3}$$

$$dK_1 = \tau \left(dS_{11} \mathbf{q}_1 + dS_{21} \mathbf{q}_2\right) \in \mathbb{R}^{1 \times 3}$$

在 v2 的分块算法中，外层循环遍历 $j$（key/value 块）。当 $j=1$ 时，加载 $\mathbf{k}_1, \mathbf{v}_1$ 到 SRAM，内层循环遍历 $i=1,2$：

- 对 $i=1$：重算 $P_{11}$，计算 $dS_{11}$，更新 $dQ_1 \leftarrow dQ_1 + \tau\, dS_{11} \mathbf{k}_1$。
- 对 $i=2$：重算 $P_{21}$，计算 $dS_{21}$，更新 $dQ_2 \leftarrow dQ_2 + \tau\, dS_{21} \mathbf{k}_1$，同时累加 $dK_1$ 的局部值 $\tilde{dK}_1 \leftarrow \tilde{dK}_1 + \tau\, dS_{21} \mathbf{q}_2$。

内层循环结束后，将 $\tilde{dK}_1$ 与之前 $i=1$ 的贡献 $\tau\, dS_{11} \mathbf{q}_1$ 合并，写回 HBM。

**泛化至一般场景：**

- $dV_j \in \mathbb{R}^{B_c \times d}$：内层循环中累加所有 query 块的贡献。
  $$dV_j = \sum_{i=1}^{T_r} P_{ij}^\top dO_i$$
  SRAM 内维护局部累加器 $\tilde{dV}_j$，内层循环结束后写回 HBM。

- $dQ_i \in \mathbb{R}^{B_r \times d}$：需要累加所有 key/value 列块的贡献。
  $$dQ_i \leftarrow dQ_i + \tau\, dS_{ij} K_j \in \mathbb{R}^{B_r \times d}$$
  每个 $(i,j)$ 块计算后立即更新，v2 中使用 atomic adds 支持序列长度维度的并行化。

- $dK_j \in \mathbb{R}^{B_c \times d}$：内层循环中累加所有 query 块的贡献。
  $$\tilde{dK}_j \leftarrow \tilde{dK}_j + \tau\, dS_{ij}^\top Q_i \in \mathbb{R}^{B_c \times d}$$
  SRAM 内维护局部累加器，内层循环结束后写回 HBM。

---

## 四、完整对比总结：v1 与 v2

**算法层面。** v1 在每轮内层循环中都执行完整的输出 rescaling；v2 改为维护 un-scaled 输出，仅在最后统一归一化，并只保存 logsumexp $L$ 而非分开保存 $m$ 和 $\ell$。这显著减少了非 matmul FLOPs。

**并行维度。** v1 仅在 batch 和 heads 维度并行；v2 额外增加了序列长度维度的并行化，将 query 行块（前向）和 key/value 列块（反向）分配到不同 thread block，大幅提升了长序列下的 SM 占用率。

**Warp 划分。** v1 采用 Split-K 策略，将 $K, V$ 切分到不同 warp，导致必须通过 shared memory 通信累加中间结果；v2 改为 Split-Q 策略，将 $Q$ 切分到不同 warp，$K, V$ 共享，warp 间无需通信，消除了 shared memory 读写瓶颈。

**理论峰值利用率。** v1 前向约为 30–50%，反向约为 25–35%；v2 前向可达 50–73%，反向可达 63%，整体接近 GEMM 的效率。

**相对加速。** v2 相比 v1 约有 2 倍加速；在端到端 GPT 训练中，v2 相比 v1 再快约 1.3 倍，单 A100 可达 225 TFLOPs/s。
