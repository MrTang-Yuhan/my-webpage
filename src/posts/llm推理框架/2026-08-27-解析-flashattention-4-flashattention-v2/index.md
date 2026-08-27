---
layout: post.njk
post_id: 2026-08-27-解析-flashattention-4-flashattention-v2
archive: llm推理框架
title: 解析 FlashAttention（4）：FlashAttention-v2 （未完成）
date: 2026-08-27
tags:
  - post
---


**论文链接：**
- FlashAttention v1：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022) —— https://arxiv.org/abs/2205.14135
- FlashAttention v2：*FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (arXiv 2023) —— https://arxiv.org/abs/2307.08691

---

## 符号约定与问题设定

为便于理解，全文采用统一的符号体系，并以 $N=4, d=3, B_r=2, B_c=2$ 作为贯穿所有推导的具体实例。

$N = 4$ 为序列长度，$d = 3$ 为特征维度，$B_r = 2$ 为每个 query 块的行数，$B_c = 2$ 为每个 key/value 块的行数。由此得到 query 块的数量 $T_r = \lceil N/B_r \rceil = 2$，key/value 块的数量 $T_c = \lceil N/B_c \rceil = 2$。

全局索引中，$n$ 用于 query 的全局行索引，$m$ 用于 key/value 的全局行索引，二者取值范围均为 $1,\dots,N$。特征维度索引为 $t = 1,\dots,d$。块索引中，$i = 1,\dots,T_r$ 为 query 块索引，$j = 1,\dots,T_c$ 为 key/value 块索引。块内索引中，$r = 1,\dots,B_r$ 为 query 块内行索引，$c = 1,\dots,B_c$ 为 key/value 块内行索引。

全局矩阵 $Q, K, V \in \mathbb{R}^{4 \times 3}$。按行分块后，$Q_i \in \mathbb{R}^{2 \times 3}$ 表示第 $i$ 个 query 块，例如 $Q_1 = \begin{bmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \end{bmatrix}$，其中 $\mathbf{q}_n \in \mathbb{R}^{1 \times 3}$ 为 $Q$ 的第 $n$ 行。$K_j, V_j \in \mathbb{R}^{2 \times 3}$ 表示第 $j$ 个 key/value 块，例如 $K_1 = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \end{bmatrix}$，$V_1 = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \end{bmatrix}$，其中 $\mathbf{k}_m, \mathbf{v}_m \in \mathbb{R}^{1 \times 3}$。

块级运算定义如下：$S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$ 为第 $i$ 个 query 块与第 $j$ 个 key 块的分数矩阵，其元素 $S_{ij}[r,c] = \mathbf{q}_{(i-1)B_r+r} \mathbf{k}_{(j-1)B_c+c}^\top$ 为标量内积。$P_{ij} \in \mathbb{R}^{B_r \times B_c}$ 为块 $(i,j)$ 对应的 softmax 概率子矩阵。$O_i \in \mathbb{R}^{B_r \times d}$ 为第 $i$ 个 query 块对应的输出块。

---

## 一、FlashAttention v1 核心回顾

标准 Attention 的核心矛盾在于必须将中间矩阵 $S, P \in \mathbb{R}^{N \times N}$ 写入 HBM，导致内存占用和 HBM 访问量均为 $O(N^2)$。v1 的解法是将 $Q, K, V$ 切分为足够小的块加载到 SRAM，通过 Online Softmax 维护全局统计量，增量地更新输出 $O$。

v1 论文中的 **Algorithm 1** 给出了完整的 Tiling 前向流程。其特点是每处理一个 $(K_j, V_j)$ 块，就必须对旧输出 $O_i$ 和新贡献分别进行 rescaling（乘以指数修正因子并除以新的全局指数和）。这意味着内层循环中包含了大量逐元素的乘除法。

v1 论文中的 **Algorithm 4** 描述了完整的反向分块算法。其关键洞察是 softmax 梯度的标量 $D_i$ 可以通过 $dO_i$ 与 $O_i$ 的点积得到，从而避免存储完整的 $P$ 矩阵。

---

## 二、FlashAttention-2 相比 v1 的三大核心变化

v2 论文（Section 3）指出，v1 虽然比标准实现快 2–4 倍，但前向仅达到理论峰值 FLOPs/s 的 30–50%，反向更是只有 25–35%。v2 通过算法微调、并行化扩展和 Warp 工作划分三个层面的改进，将利用率提升至 50–73%（A100），实现约 2 倍加速。

### 变化 1：算法微调 —— 减少非 Matmul FLOPs

现代 GPU 上，非矩阵乘法操作（逐元素乘、指数、除法等）与 Matmul 的吞吐量差距极大。以 A100 为例，FP16/BF16 Matmul 的理论峰值约为 312 TFLOPs/s，而 FP32 非 Matmul 仅约 19.5 TFLOPs/s。因此，减少非 matmul 操作、让 GPU 尽可能多地在 Tensor Core 上做矩阵乘，是提升利用率的关键。

v2 对此做了两处算法层面的调整：

**第一，延迟输出归一化。** v1 在每轮内层循环结束时都执行完整的 rescaling：先将旧输出 $O_i$ 乘以指数修正因子，再加上新块的贡献，最后除以新的全局指数和 $\ell_i^{\text{new}}$。v2 改为维护一个 **"un-scaled" 的输出** $\tilde{O}_i$，在内层循环中仅做累加和指数修正，直到所有 key/value 块处理完毕，才在最后统一除以 $\text{diag}(\ell_i^{(T_c)})^{-1}$ 得到正确输出。这大幅减少了循环内部的逐元素除法。

**第二，合并保存的统计量。** v1 在反向传播时需要分别保存行最大值 $m$ 和行指数和 $\ell$。v2 改为只保存 **logsumexp** $L = m + \log(\ell)$。这不仅减少了额外的内存占用，也简化了反向传播中的非 matmul 计算。

### 变化 2：并行化策略 —— 沿序列长度维度切分

v1 仅在 batch size 和 number of heads 两个维度上并行化。当序列很长时，batch size 和 head 数往往较小，导致 GPU 的 Streaming Multiprocessors (SMs) 占用率不足。

v2 增加了对序列长度维度的并行化：

**在前向传播中**，v1 的外层循环遍历 key/value 块 $K_j$，内层循环遍历 query 块 $Q_i$。v2 将循环顺序调换：外层循环遍历 query 的行块 $Q_i$，内层循环遍历 key/value 的列块 $K_j$。由于不同 query 行块之间的计算是完全独立的，可以将每个行块分配给一个独立的 thread block，无需任何跨 block 通信。

**在反向传播中**，类似地将外层循环（遍历 key/value 的列块 $K_j$）并行化。每个 thread block 负责一个列块，计算该列块对 $dK, dV$ 的贡献。由于 $dQ$ 需要累加所有列块的贡献，v2 使用 **atomic adds** 来协调不同 thread block 对 $dQ$ 的更新。

v2 论文中的 **Figure 2** 直观展示了这一并行化方案：左图（Forward）显示每个 Worker（thread block）负责注意力矩阵的一个行块；右图（Backward）显示每个 Worker 负责一个列块。

![](img/flash-atten-v2-fig2.png)

### 变化 3：Warp 级别工作划分 —— 避免 Split-K

v1 采用的是 **Split-K** 方案：将 $K$ 和 $V$ 切分到 4 个 warp，而 $Q$ 被所有 warp 共享。每个 warp 计算自己负责的 $K/V$ 切片与 $Q$ 的乘积，得到部分中间结果。由于最终输出需要累加所有 warp 的贡献，这些中间结果必须被写入 shared memory，经过同步后，再由其他 warp 读取并加和。这带来了大量的 shared memory 读写和同步开销。

v2 改为将 **$Q$ 切分到 4 个 warp**，而 $K$ 和 $V$ 被所有 warp 共享。每个 warp 负责计算自己的 $Q_i$ 切片与完整的 $K^\top$ 的乘积，然后直接与对应的 $V$ 切片相乘，得到该 warp 对应的输出切片。由于每个 warp 的输入 $Q_i$ 和输出 $O_i$ 切片是独立的，**warp 之间无需通信**，完全避免了 shared memory 上的中间结果读写。

v2 论文中的 **Figure 3** 清晰对比了两种划分策略：子图 (a) 展示了 FlashAttention v1 的 Split-K 方案（$K^T$ 和 $V$ 被 split 到 Warp 1-4，$Q$ 共享，需要通信累加）；子图 (b) 展示了 FlashAttention-2 的方案（$Q$ 被 split 到 Warp 1-4，$K^T$ 和 $V$ 共享，无通信）。

![](img/flash-atten-v2-fig3.png)

---

## 三、前置原理：Online Softmax 的数学本质

在深入 v2 的具体更新式之前，必须先理解 **Online Softmax** 的数学本质——这是所有 FlashAttention 系列算法的根基。

标准 Attention 对第 $n$ 个 query 行的输出为：
$$O_{n,t} = \sum_{m=1}^{N} P_{n,m} V_{m,t} = \frac{\sum_{m=1}^{N} \exp(S_{n,m}) V_{m,t}}{\sum_{m'=1}^{N} \exp(S_{n,m'})}$$

其中 $S_{n,m} = \mathbf{q}_n \mathbf{k}_m^\top$。由于 $N$ 可能很大（例如 32K），无法一次性将所有 $S_{n,1}, \dots, S_{n,N}$ 载入 SRAM，因此必须分块处理。

**核心矛盾：** Softmax 的分母 $\sum_{m=1}^N \exp(S_{n,m})$ 依赖于**整行**的 $S_{n,m}$。若仅看到前 $j$ 个 key 块，如何在不存储完整中间矩阵的前提下得到正确的最终输出？

**解决思路——动态定标（Rescaling）：**

假设已处理前 $j-1$ 个 key/value 块，并维护两个统计量：
- $m^{(j-1)}$：前 $j-1$ 个块中的**全局行最大值**
- $\ell^{(j-1)}$：前 $j-1$ 个块以 $m^{(j-1)}$ 为基准的**指数和**

即：
$$\ell^{(j-1)} = \sum_{t=1}^{j-1} \sum_{c=1}^{B_c} \exp(S_{n,(t-1)B_c+c} - m^{(j-1)})$$

其中 $S_{n,(t-1)B_c+c}$ 表示第 $n$ 个 query 与第 $t$ 个 key 块中第 $c$ 个 key 的内积。

现处理第 $j$ 个块，得到该块的局部最大值 $\tilde{m}$。新的全局最大值为 $m^{(j)} = \max(m^{(j-1)}, \tilde{m})$。

当基准从 $m^{(j-1)}$ 变为 $m^{(j)}$ 时，旧累积量必须按指数差重新定标：

$$\sum_{t=1}^{j-1} \sum_{c=1}^{B_c} \exp(S_{n,(t-1)B_c+c} - m^{(j)}) 
\\= \exp(m^{(j-1)} - m^{(j)}) \sum_{t=1}^{j-1} \sum_{c=1}^{B_c} \exp(S_{n,(t-1)B_c+c} - m^{(j-1)}) 
\\ = \exp(m^{(j-1)} - m^{(j)}) \cdot \ell^{(j-1)}$$

同理，对于未归一化的加权和 $O^{(j-1)} = \sum_{t=1}^{j-1} \sum_{c=1}^{B_c} \exp(S_{n,(t-1)B_c+c} - m^{(j-1)}) \mathbf{v}_{(t-1)B_c+c}$，重新定标后：
$$\sum_{t=1}^{j-1} \sum_{c=1}^{B_c} \exp(S_{n,(t-1)B_c+c} - m^{(j)}) \mathbf{v}_{(t-1)B_c+c} = \exp(m^{(j-1)} - m^{(j)}) \cdot O^{(j-1)}$$

然后加上新块的贡献 $\sum_{c=1}^{B_c} \exp(S_{n,(j-1)B_c+c} - m^{(j)}) \mathbf{v}_{(j-1)B_c+c}$，就得到了以新基准 $m^{(j)}$ 表示的累积量。这就是所有更新公式的来源。

---

## 四、FlashAttention-2 关键公式推导

以下推导基于上述符号体系，先以 $N=4, d=3, B_r=B_c=2$ 为例从元素级别展示 v2 的核心机制，再泛化至一般场景。

### 4.1 v2 前向传播：un-scaled 输出的增量更新

**具体例子：** $N=4, d=3, B_r=2, B_c=2$。输入矩阵为：
$$Q = \begin{bmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \\ \mathbf{q}_3 \\ \mathbf{q}_4 \end{bmatrix}, \quad K = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \\ \mathbf{k}_3 \\ \mathbf{k}_4 \end{bmatrix}, \quad V = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \\ \mathbf{v}_3 \\ \mathbf{v}_4 \end{bmatrix}$$

分块后：
$$Q_1 = \begin{bmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \end{bmatrix}, Q_2 = \begin{bmatrix} \mathbf{q}_3 \\ \mathbf{q}_4 \end{bmatrix}, \quad K_1 = \begin{bmatrix} \mathbf{k}_1 \\ \mathbf{k}_2 \end{bmatrix}, K_2 = \begin{bmatrix} \mathbf{k}_3 \\ \mathbf{k}_4 \end{bmatrix}, \quad V_1 = \begin{bmatrix} \mathbf{v}_1 \\ \mathbf{v}_2 \end{bmatrix}, V_2 = \begin{bmatrix} \mathbf{v}_3 \\ \mathbf{v}_4 \end{bmatrix}$$

以第 $i=1$ 个 query 块（即 $Q_1 = \begin{bmatrix} \mathbf{q}_1 \\ \mathbf{q}_2 \end{bmatrix}$）为例，展示其输出如何通过 v2 的 un-scaled 累加方式逐步构造。

**变量声明（针对第 $i=1$ 个 query 块）：**
- $S_{1j} = Q_1 K_j^\top \in \mathbb{R}^{2 \times 2}$：第 1 个 query 块与第 $j$ 个 key 块的内积矩阵。
- $m^{(j)} \in \mathbb{R}^{2}$：处理完前 $j$ 个 key 块后，当前全局行最大值（每行一个标量）。初始值 $m^{(0)} = \begin{bmatrix} -\infty \\ -\infty \end{bmatrix}$。
- $\ell^{(j)} \in \mathbb{R}^{2}$：处理完前 $j$ 个 key 块后，以 $m^{(j)}$ 为基准的全局行指数和。初始值 $\ell^{(0)} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$。
- $O^{(j)} \in \mathbb{R}^{2 \times 3}$：处理完前 $j$ 个 key 块后的 **un-scaled** 累积输出。初始值 $O^{(0)} = \mathbf{0}^{2 \times 3}$。

---

**第一步（$j=1$，处理 $K_1, V_1$，即 $\mathbf{k}_1, \mathbf{k}_2$ 和 $\mathbf{v}_1, \mathbf{v}_2$）：**

1. **计算局部分数：**
$$S_{11} = Q_1 K_1^\top = \begin{bmatrix} \mathbf{q}_1 \mathbf{k}_1^\top & \mathbf{q}_1 \mathbf{k}_2^\top \\ \mathbf{q}_2 \mathbf{k}_1^\top & \mathbf{q}_2 \mathbf{k}_2^\top \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

2. **更新全局行最大值：**
$$m^{(1)} = \max\left(m^{(0)},\ \text{rowmax}(S_{11})\right) = \text{rowmax}(S_{11}) = \begin{bmatrix} \max(S_{11}[1,1], S_{11}[1,2]) \\ \max(S_{11}[2,1], S_{11}[2,2]) \end{bmatrix} \in \mathbb{R}^{2}$$

此处 $\max$ 为逐元素取最大。由于此前未处理任何块，$m^{(0)} = -\infty$，故 $m^{(1)}$ 就是 $S_{11}$ 的每行最大值。

3. **计算局部平移指数矩阵：**
$$\tilde{P}_{11} = \exp\left(S_{11} - m^{(1)}\right) \in \mathbb{R}^{2 \times 2}$$

这里的减法是**逐行广播**：$S_{11}$ 的第 $r$ 行减去 $m^{(1)}$ 的第 $r$ 个元素。结果 $\tilde{P}_{11}[r,c] = \exp(S_{11}[r,c] - m^{(1)}[r])$。

此步骤的数学动机：为数值稳定性，softmax 计算时需减去行最大值。$\tilde{P}_{11}$ 的元素表示在已知当前全局最大值 $m^{(1)}$ 的前提下，每个分数相对于该最大值的指数值。

4. **更新全局行指数和：**
$$\ell^{(1)} = \exp\left(m^{(0)} - m^{(1)}\right) \odot \ell^{(0)} + \text{rowsum}\left(\tilde{P}_{11}\right) = \text{rowsum}\left(\tilde{P}_{11}\right) \in \mathbb{R}^{2}$$

其中 $\text{rowsum}(\tilde{P}_{11})$ 对 $\tilde{P}_{11}$ 每行求和，得到一个长度为 2 的列向量。

此步骤的数学动机：根据前置原理，旧指数和 $\ell^{(0)}=0$ 需按 $\exp(m^{(0)} - m^{(1)})$ 重新定标。由于 $m^{(0)} = -\infty$，该项为 0。新的指数和即为当前块每行指数值之和。

5. **更新 un-scaled 输出：**
$$O^{(1)} = \text{diag}\left(\exp\left(m^{(0)} - m^{(1)}\right)\right) O^{(0)} + \tilde{P}_{11} V_1 = \tilde{P}_{11} V_1 \in \mathbb{R}^{2 \times 3}$$

此步骤的数学动机：旧输出 $O^{(0)} = \mathbf{0}$ 需按 $\exp(m^{(0)} - m^{(1)})$ 重新定标以匹配新基准 $m^{(1)}$。由于 $O^{(0)} = \mathbf{0}$，该项消失。新输出即为当前块的平移指数矩阵与 $V_1$ 的乘积。

具体写出第 $r$ 行（$r=1$ 对应 $\mathbf{q}_1$，$r=2$ 对应 $\mathbf{q}_2$）：
$$O^{(1)}[r,:] = \sum_{c=1}^{2} \tilde{P}_{11}[r,c] \cdot \mathbf{v}_{(j-1)B_c+c} = \sum_{c=1}^{2} \exp\left(S_{11}[r,c] - m^{(1)}[r]\right) \mathbf{v}_c$$

当 $j=1$ 时，$(j-1)B_c+c = c$，故对应 $\mathbf{v}_1, \mathbf{v}_2$。

---

**第二步（$j=2$，处理 $K_2, V_2$，即 $\mathbf{k}_3, \mathbf{k}_4$ 和 $\mathbf{v}_3, \mathbf{v}_4$）：**

1. **计算局部分数：**
$$S_{12} = Q_1 K_2^\top = \begin{bmatrix} \mathbf{q}_1 \mathbf{k}_3^\top & \mathbf{q}_1 \mathbf{k}_4^\top \\ \mathbf{q}_2 \mathbf{k}_3^\top & \mathbf{q}_2 \mathbf{k}_4^\top \end{bmatrix} \in \mathbb{R}^{2 \times 2}$$

2. **更新全局行最大值：**
$$m^{(2)} = \max\left(m^{(1)},\ \text{rowmax}(S_{12})\right) \in \mathbb{R}^{2}$$

全局最大值必须在旧全局最大值 $m^{(1)}$ 和新块行最大值 $\text{rowmax}(S_{12})$ 之间逐元素取最大。

此处需分两种情况讨论，以展示 rescaling 的核心作用。

---

**情况 A：第 $r$ 行的全局最大值未变，即 $m^{(2)}[r] = m^{(1)}[r]$**

3. **计算局部平移指数：**
$$\tilde{P}_{12}[r,c] = \exp\left(S_{12}[r,c] - m^{(2)}[r]\right) = \exp\left(S_{12}[r,c] - m^{(1)}[r]\right), \quad c=1,2$$

4. **更新全局指数和：**
$$\ell^{(2)}[r] = \exp\left(m^{(1)}[r] - m^{(2)}[r]\right) \cdot \ell^{(1)}[r] + \sum_{c=1}^{2} \tilde{P}_{12}[r,c] = 1 \cdot \ell^{(1)}[r] + \sum_{c=1}^{2} \tilde{P}_{12}[r,c]$$

数学动机：由于基准未变（$m^{(2)}[r] = m^{(1)}[r]$），旧指数和 $\ell^{(1)}[r]$ 无需重新定标（乘数为 1）。新的全局指数和就是旧和加上新块的贡献。

5. **更新 un-scaled 输出：**
$$O^{(2)}[r,:] = \exp\left(m^{(1)}[r] - m^{(2)}[r]\right) \cdot O^{(1)}[r,:] + \sum_{c=1}^{2} \tilde{P}_{12}[r,c] \mathbf{v}_{2,c} = O^{(1)}[r,:] + \sum_{c=1}^{2} \tilde{P}_{12}[r,c] \mathbf{v}_{2,c}$$

其中 $\mathbf{v}_{2,1} = \mathbf{v}_3, \mathbf{v}_{2,2} = \mathbf{v}_4$。

数学动机：同理，基准未变，旧输出无需重新定标。新输出直接累加新块贡献。

6. **执行最终归一化（循环结束，$j=T_c=2$）：**
$$O_1[r,:] = \frac{1}{\ell^{(2)}[r]} O^{(2)}[r,:]$$

验证正确性：此时
$$O^{(2)}[r,:] = \sum_{c=1}^{2} \exp\left(S_{11}[r,c] - m^{(1)}[r]\right)\mathbf{v}_{1,c} + \sum_{c=1}^{2} \exp\left(S_{12}[r,c] - m^{(1)}[r]\right)\mathbf{v}_{2,c}$$

由于 $m^{(2)}[r] = m^{(1)}[r]$，上式可写为：
$$O^{(2)}[r,:] = \sum_{j=1}^{2} \sum_{c=1}^{2} \exp\left(S_{1j}[r,c] - m^{(2)}[r]\right)\mathbf{v}_{j,c}$$

除以 $\ell^{(2)}[r] = \sum_{j=1}^{2} \sum_{c=1}^{2} \exp(S_{1j}[r,c] - m^{(2)}[r])$，即得标准 softmax attention 输出。

---

**情况 B：第 $r$ 行的全局最大值更新，即 $m^{(2)}[r] > m^{(1)}[r]$**

设 $m^{(2)}[r] = \text{rowmax}(S_{12})[r]$，即新块中出现了更大的分数。

3. **计算局部平移指数：**
$$\tilde{P}_{12}[r,c] = \exp\left(S_{12}[r,c] - m^{(2)}[r]\right)$$

4. **更新全局指数和：**
$$\ell^{(2)}[r] = \exp\left(m^{(1)}[r] - m^{(2)}[r]\right) \cdot \ell^{(1)}[r] + \sum_{c=1}^{2} \tilde{P}_{12}[r,c]$$

数学动机：由于基准从 $m^{(1)}[r]$ 提升到了 $m^{(2)}[r]$，旧指数和 $\ell^{(1)}[r]$ 必须乘以 $\exp(m^{(1)}[r] - m^{(2)}[r]) < 1$ 进行重新定标（见前置原理推导），再加上新块以新基准计算的指数和。

5. **更新 un-scaled 输出：**
$$O^{(2)}[r,:] = \exp\left(m^{(1)}[r] - m^{(2)}[r]\right) \cdot O^{(1)}[r,:] + \sum_{c=1}^{2} \tilde{P}_{12}[r,c] \mathbf{v}_{2,c}$$

数学动机：旧输出 $O^{(1)}[r,:]$ 是以旧基准 $m^{(1)}[r]$ 累积的未归一化加权和：
$$O^{(1)}[r,:] = \sum_{c=1}^{2} \exp\left(S_{11}[r,c] - m^{(1)}[r]\right) \mathbf{v}_{1,c}$$

要将其转换为以新基准 $m^{(2)}[r]$ 表示，必须乘以 $\exp(m^{(1)}[r] - m^{(2)}[r])$：
$$\exp\left(m^{(1)}[r] - m^{(2)}[r]\right) O^{(1)}[r,:] = \sum_{c=1}^{2} \exp\left(S_{11}[r,c] - m^{(2)}[r]\right) \mathbf{v}_{1,c}$$

然后加上新块的贡献 $\sum_{c=1}^{2} \exp(S_{12}[r,c] - m^{(2)}[r]) \mathbf{v}_{2,c}$，即得以 $m^{(2)}[r]$ 为基准的完整未归一化加权和。

6. **执行最终归一化：**
$$O_1[r,:] = \frac{1}{\ell^{(2)}[r]} O^{(2)}[r,:]$$

验证正确性：此时分子为 $\sum_{j=1}^{2}\sum_{c=1}^{2} \exp(S_{1j}[r,c] - m^{(2)}[r]) \mathbf{v}_{j,c}$，分母为 $\sum_{j=1}^{2}\sum_{c=1}^{2} \exp(S_{1j}[r,c] - m^{(2)}[r])$，恰好是标准 softmax attention 的分子与分母。

---

**泛化至一般场景：**

设一般维度下，$Q, K, V \in \mathbb{R}^{N \times d}$，分块尺寸为 $B_r$ 和 $B_c$。对第 $i$ 个 query 块，定义：
- $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$
- $m_i^{(j)} \in \mathbb{R}^{B_r}$：全局行最大值，初始 $m_i^{(0)} = (-\infty)^{B_r}$
- $\ell_i^{(j)} \in \mathbb{R}^{B_r}$：全局行指数和，初始 $\ell_i^{(0)} = 0^{B_r}$
- $O_i^{(j)} \in \mathbb{R}^{B_r \times d}$：un-scaled 累积输出，初始 $O_i^{(0)} = \mathbf{0}^{B_r \times d}$

对 $j = 1, \dots, T_c$，依次执行：

1. $S_{ij} = Q_i K_j^\top$
2. $m_i^{(j)} = \max\left(m_i^{(j-1)},\ \text{rowmax}(S_{ij})\right)$
3. $\tilde{P}_{ij} = \exp\left(S_{ij} - m_i^{(j)}\right)$（逐行广播减法）
4. $\ell_i^{(j)} = \exp\left(m_i^{(j-1)} - m_i^{(j)}\right) \odot \ell_i^{(j-1)} + \text{rowsum}\left(\tilde{P}_{ij}\right)$
5. $O_i^{(j)} = \text{diag}\left(\exp\left(m_i^{(j-1)} - m_i^{(j)}\right)\right) O_i^{(j-1)} + \tilde{P}_{ij} V_j$

第 4、5 步的数学动机：当 $m_i^{(j)} > m_i^{(j-1)}$ 时，$\exp(m_i^{(j-1)} - m_i^{(j)}) < 1$，用于将旧累积量从旧基准 $m_i^{(j-1)}$ 重新定标到新基准 $m_i^{(j)}$；当两者相等时，乘数为 1，无需定标。这是前置原理中通用公式的直接应用。

6. 循环结束后，统一归一化：
$$O_i = \text{diag}\left(\ell_i^{(T_c)}\right)^{-1} O_i^{(T_c)}$$

v2 与 v1 的关键区别：v1 在第 5 步后会立即执行 $O_i^{(j)} \leftarrow \text{diag}(\ell_i^{(j)})^{-1} O_i^{(j)}$，即每轮都除以当前全局指数和；v2 则**延迟此除法**到循环结束后，从而将循环内的非 matmul 除法减少为仅 1 次。

---

### 4.2 v2 保存的统计量：logsumexp $L_i$

v2 不再分别保存 $m_i$ 和 $\ell_i$，而是保存 **logsumexp**：
$$L_i = m_i^{(T_c)} + \log\left(\ell_i^{(T_c)}\right) \in \mathbb{R}^{B_r}$$

**为何只保存 $L_i$ 即足够？**

从定义出发：设前向结束后，第 $i$ 个 query 块的最终全局最大值为 $m_i = m_i^{(T_c)}$，最终全局指数和为 $\ell_i = \ell_i^{(T_c)}$。则
$$L_i = m_i + \log(\ell_i)$$

**反向传播中恢复 $P_{ij}$：**

v2 Algorithm 2 第 11 行利用 $L_i$ 重计算概率。对任意 key 块 $j$，先重算分数 $S_{ij} = Q_i K_j^\top \in \mathbb{R}^{B_r \times B_c}$，则：

$$P_{ij} = \exp\left(S_{ij} - L_i\right)$$

此公式恢复正确 softmax 概率的推导如下：

将 $L_i$ 的定义代入：
$$P_{ij} = \exp\left(S_{ij} - m_i - \log(\ell_i)\right) = \exp\left(S_{ij} - m_i\right) \cdot \exp\left(-\log(\ell_i)\right) = \frac{\exp\left(S_{ij} - m_i\right)}{\ell_i}$$

这正是全局 softmax 概率的标准形式：分子为以最终全局最大值 $m_i$ 为基准的平移指数，分母为全局指数和 $\ell_i$。

**为何保存 $L_i$ 优于保存 $(m_i, \ell_i)$？**

第一，内存减半：只需存储一个长度为 $B_r$ 的向量，而非两个。第二，反向计算简化：恢复 $P_{ij}$ 时只需一次减法加指数，无需先分别读取 $m_i$ 和 $\ell_i$ 再做除法。

---

### 4.3 v2 反向传播：$D_i$ 的代数简化

在标准反向传播中，softmax 的 Jacobian 引入标量 $D_i$。以下详细推导为何 $D_i$ 可简化为 $dO_i$ 与 $O_i$ 的逐元素乘积的行和。

**背景：Attention 的前向与反向关系**

前向传播中，对第 $n$ 个全局 query 行：
$$O_{n,t} = \sum_{m=1}^{N} P_{n,m} V_{m,t}, \quad t=1,\dots,d$$

其中 $P_{n,m} = \frac{\exp(S_{n,m})}{\sum_{m'=1}^N \exp(S_{n,m'})}$ 为 softmax 概率，$S_{n,m} = \mathbf{q}_n \mathbf{k}_m^\top$。

反向传播中，已知损失对输出的梯度 $dO_{n,t} = \frac{\partial \mathcal{L}}{\partial O_{n,t}}$。

**第一步：求 $dP_{n,m} = \frac{\partial \mathcal{L}}{\partial P_{n,m}}$**

由于 $O_{n,t}$ 直接依赖于 $P_{n,m}$（且 $O_{n,t} = \sum_{m} P_{n,m} V_{m,t}$），由链式法则：
$$dP_{n,m} = \sum_{t=1}^{d} \frac{\partial \mathcal{L}}{\partial O_{n,t}} \frac{\partial O_{n,t}}{\partial P_{n,m}} = \sum_{t=1}^{d} dO_{n,t} V_{m,t}$$

物理意义：$dP_{n,m}$ 是损失函数通过输出 $O_n$ 的所有维度 $t$ 回传到概率 $P_{n,m}$ 的梯度之和。

**第二步：Softmax 的 Jacobian**

对于按行进行的 softmax，若 $P_{n,:} = \text{softmax}(S_{n,:})$，则：
$$dS_{n,m} = P_{n,m} \left(dP_{n,m} - \sum_{m'=1}^{N} P_{n,m'} dP_{n,m'}\right)$$

此式为 softmax 函数梯度的标准结论。softmax 的输出 $P_{n,m}$ 依赖于输入 $S_{n,m'}$（所有 $m'$），其 Jacobian 矩阵为 $\text{diag}(P_{n,:}) - P_{n,:} P_{n,:}^\top$。应用到梯度向量 $dP_{n,:}$ 上，即得上式。

定义：
$$D_n = \sum_{m'=1}^{N} P_{n,m'} dP_{n,m'}$$

这是 softmax 梯度的归一化修正项。

**第三步：将 $D_n$ 转换为 $dO_n$ 与 $O_n$ 的形式**

将第一步得到的 $dP_{n,m'} = \sum_{t=1}^{d} dO_{n,t} V_{m',t}$ 代入 $D_n$：

$$D_n = \sum_{m'=1}^{N} P_{n,m'} \left(\sum_{t=1}^{d} dO_{n,t} V_{m',t}\right)$$

交换求和顺序：
$$D_n = \sum_{t=1}^{d} dO_{n,t} \left(\sum_{m'=1}^{N} P_{n,m'} V_{m',t}\right)$$

注意到括号内正是 $O_{n,t}$ 的定义：
$$O_{n,t} = \sum_{m'=1}^{N} P_{n,m'} V_{m',t}$$

因此：
$$D_n = \sum_{t=1}^{d} dO_{n,t} O_{n,t}$$

**泛化至分块场景：**

对第 $i$ 个 query 块，$dO_i, O_i \in \mathbb{R}^{B_r \times d}$，则：
$$D_i = \text{rowsum}\left(dO_i \circ O_i\right) \in \mathbb{R}^{B_r}$$

其中 $\circ$ 为 Hadamard 积（逐元素相乘），$\text{rowsum}(\cdot)$ 对每行求和得到长度为 $B_r$ 的列向量。

关键意义：计算 $D_i$ **无需访问 $P$ 的任何元素**，仅需 $dO_i$ 与 $O_i$ 做点积，复杂度 $O(B_r d)$，完全在 SRAM 内完成。这正是 FlashAttention 反向传播避免存储 $O(N^2)$ 矩阵 $P$ 的核心代数技巧。

---

### 4.4 v2 反向传播：$dS_{ij}$ 的分块计算

**具体例子（$N=4, d=3, B_r=B_c=2$）：**

考虑第 $i=1$ 个 query 块和第 $j=1$ 个 key/value 块。

已知：
- $P_{11} \in \mathbb{R}^{2 \times 2}$：由前向保存的 $L_1$ 重算得到（见 4.2 节）。
- $dP_{11} = dO_1 V_1^\top \in \mathbb{R}^{2 \times 2}$：由链式法则 $dP_{ij} = dO_i V_j^\top$ 得到。
- $D_1 \in \mathbb{R}^{2}$：由 4.3 节公式计算得到。

则分块 softmax 梯度为：
$$dS_{11} = P_{11} \circ \left(dP_{11} - D_1 \mathbf{1}_{2}^\top\right) \in \mathbb{R}^{2 \times 2}$$

其中 $\mathbf{1}_{2}^\top = \begin{bmatrix} 1 & 1 \end{bmatrix} \in \mathbb{R}^{1 \times 2}$。

此公式的数学动机：这是 4.3 节 softmax Jacobian 的直接分块实现。$D_1 \mathbf{1}_{2}^\top$ 是外积，生成一个 $2 \times 2$ 矩阵，其第 $r$ 行全为 $D_1[r]$。这实现了对 $dP_{11}$ 的**逐行广播减法**：对第 $r$ 个 query，减去同一个标量 $D_1[r]$。然后逐元素乘以 $P_{11}$，得到 $dS_{11}$。

物理意义：$dS_{11}[r,c]$ 表示损失函数对分数 $S_{11}[r,c]$ 的梯度。由于 softmax 的归一化特性，增大某个分数不仅会提高对应概率，还会通过分母降低其他概率，因此需要减去修正项 $D_1[r]$。

**泛化至一般场景：**

对第 $i$ 个 query 块和第 $j$ 个 key/value 块：
$$dS_{ij} = P_{ij} \circ \left(dP_{ij} - D_i \mathbf{1}_{B_c}^\top\right) \in \mathbb{R}^{B_r \times B_c}$$

其中：
- $dP_{ij} = dO_i V_j^\top \in \mathbb{R}^{B_r \times B_c}$
- $D_i \in \mathbb{R}^{B_r}$
- $\mathbf{1}_{B_c}^\top \in \mathbb{R}^{1 \times B_c}$ 为全 1 行向量

---

### 4.5 v2 反向传播：$dQ_i$ 与 $dK_j$ 的分块累加

**背景：从分数矩阵到输入梯度的链式法则**

由 $S_{n,m} = \tau \cdot \mathbf{q}_n \mathbf{k}_m^\top$（其中 $\tau$ 为缩放常数，通常为 $1/\sqrt{d}$），链式法则给出：

$$d\mathbf{q}_n = \tau \sum_{m=1}^{N} dS_{n,m} \mathbf{k}_m \in \mathbb{R}^{1 \times d}$$
$$d\mathbf{k}_m = \tau \sum_{n=1}^{N} dS_{n,m} \mathbf{q}_n \in \mathbb{R}^{1 \times d}$$
$$d\mathbf{v}_m = \sum_{n=1}^{N} P_{n,m} d\mathbf{o}_n \in \mathbb{R}^{1 \times d}$$

此三式分别为 $S = \tau Q K^\top$ 和 $O = PV$ 的反向传播结果。$d\mathbf{q}_n$ 是所有 key 对第 $n$ 个 query 的梯度贡献之和；$d\mathbf{k}_m$ 是所有 query 对第 $m$ 个 key 的梯度贡献之和；$d\mathbf{v}_m$ 是所有 query 通过概率 $P_{n,m}$ 加权后对第 $m$ 个 value 的梯度贡献之和。

**具体例子（$N=4, d=3, B_r=B_c=2$）：**

在 v2 的分块算法中，外层循环遍历 $j$（key/value 块），内层循环遍历 $i$（query 块）。

**当 $j=1$ 时**（加载 $K_1, V_1$ 到 SRAM）：

内层循环 $i=1$：
- 重算 $P_{11} = \exp(S_{11} - L_1)$，计算 $dS_{11} = P_{11} \circ (dP_{11} - D_1 \mathbf{1}_2^\top)$（见 4.4 节）。
- 更新 $dQ_1 \leftarrow dQ_1 + \tau \cdot dS_{11} K_1$。
- 累加 $dK_1 \leftarrow dK_1 + \tau \cdot dS_{11}^\top Q_1$。
- 累加 $dV_1 \leftarrow dV_1 + P_{11}^\top dO_1$。

内层循环 $i=2$：
- 重算 $P_{21} = \exp(S_{21} - L_2)$，计算 $dS_{21} = P_{21} \circ (dP_{21} - D_2 \mathbf{1}_2^\top)$。
- 更新 $dQ_2 \leftarrow dQ_2 + \tau \cdot dS_{21} K_1$。
- 累加 $dK_1 \leftarrow dK_1 + \tau \cdot dS_{21}^\top Q_2$。
- 累加 $dV_1 \leftarrow dV_1 + P_{21}^\top dO_2$。

内层循环结束后，将 $dK_1, dV_1$ 写回 HBM。

$dQ_1$ 如此更新的数学动机：由 $d\mathbf{q}_n = \tau \sum_{m} dS_{n,m} \mathbf{k}_m$，对块内所有行同时计算即得矩阵乘法 $dS_{11} K_1$。由于 $dQ_1$ 需要累加所有 key/value 块（$j=1$ 和 $j=2$）的贡献，此处为累加更新。

$dK_1$ 如此更新的数学动机：由 $d\mathbf{k}_m = \tau \sum_{n} dS_{n,m} \mathbf{q}_n$，对块内所有行同时计算即得矩阵乘法 $dS_{11}^\top Q_1$。注意转置：$dS_{11}^\top \in \mathbb{R}^{2 \times 2}$，$Q_1 \in \mathbb{R}^{2 \times 3}$，乘积 $\in \mathbb{R}^{2 \times 3}$，恰好是 $dK_1$ 的尺寸。

$dV_1$ 如此累加的数学动机：由 $d\mathbf{v}_m = \sum_{n} P_{n,m} d\mathbf{o}_n$，对块内所有行同时计算。$P_{11}^\top dO_1$ 是第 1 个 query 块对 $dV_1$ 的贡献，$P_{21}^\top dO_2$ 是第 2 个 query 块的贡献。

**当 $j=2$ 时**（加载 $K_2, V_2$ 到 SRAM）：

重复上述过程：
- $dQ_1 \leftarrow dQ_1 + \tau \cdot dS_{12} K_2$
- $dQ_2 \leftarrow dQ_2 + \tau \cdot dS_{22} K_2$
- 累加 $dK_2 \leftarrow dK_2 + \tau \cdot dS_{12}^\top Q_1 + \tau \cdot dS_{22}^\top Q_2$
- 累加 $dV_2 \leftarrow dV_2 + P_{12}^\top dO_1 + P_{22}^\top dO_2$

**泛化至一般场景：**

- **$dV_j \in \mathbb{R}^{B_c \times d}$：** 内层循环中累加所有 query 块的贡献。
  $$dV_j = \sum_{i=1}^{T_r} P_{ij}^\top dO_i$$
  SRAM 内维护局部累加器，内层循环结束后写回 HBM。

- **$dQ_i \in \mathbb{R}^{B_r \times d}$：** 需要累加所有 key/value 列块的贡献。
  $$dQ_i \leftarrow dQ_i + \tau\, dS_{ij} K_j \in \mathbb{R}^{B_r \times d}$$
  每个 $(i,j)$ 块计算后立即更新。v2 中使用 atomic adds 支持序列长度维度的并行化（多个 thread block 可能同时更新同一个 $dQ_i$）。

- **$dK_j \in \mathbb{R}^{B_c \times d}$：** 内层循环中累加所有 query 块的贡献。
  $$dK_j \leftarrow dK_j + \tau\, dS_{ij}^\top Q_i \in \mathbb{R}^{B_c \times d}$$
  SRAM 内维护局部累加器，内层循环结束后写回 HBM。

---

## 五、完整对比总结：v1 与 v2

**算法层面。** v1 在每轮内层循环中都执行完整的输出 rescaling（除以当前 $\ell$）；v2 改为维护 un-scaled 输出，仅在最后统一归一化，并只保存 logsumexp $L$ 而非分开保存 $m$ 和 $\ell$。这显著减少了非 matmul FLOPs。

**并行维度。** v1 仅在 batch 和 heads 维度并行；v2 额外增加了序列长度维度的并行化，将 query 行块（前向）和 key/value 列块（反向）分配到不同 thread block，大幅提升了长序列下的 SM 占用率。

**Warp 划分。** v1 采用 Split-K 策略，将 $K, V$ 切分到不同 warp，导致必须通过 shared memory 通信累加中间结果；v2 改为 Split-Q 策略，将 $Q$ 切分到不同 warp，$K, V$ 共享，warp 间无需通信，消除了 shared memory 读写瓶颈。

**理论峰值利用率。** v1 前向约为 30–50%，反向约为 25–35%；v2 前向可达 50–73%，反向可达 63%，整体接近 GEMM 的效率。

**相对加速。** v2 相比 v1 约有 2 倍加速；在端到端 GPT 训练中，v2 相比 v1 再快约 1.3 倍，单 A100 可达 225 TFLOPs/s。
