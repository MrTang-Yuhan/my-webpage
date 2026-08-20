---
layout: post.njk
post_id: 2026-08-20-解析-flashattention-3-flashattention-v1-反向传播
archive: llm推理框架
title: 解析 FlashAttention（3）：FlashAttention-v1 反向传播
date: 2026-08-20
tags:
  - post
---


## 二、反向传播（未完成）

反向传播的目标是在不保存 $N \times N$ 矩阵 $\mathbf{S}$ 与 $\mathbf{P}$ 的前提下，由上游梯度 $\frac{\partial \mathcal{L}}{\partial \mathbf{O}}$ 计算 $\frac{\partial \mathcal{L}}{\partial \mathbf{Q}}$、$\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$、$\frac{\partial \mathcal{L}}{\partial \mathbf{V}}$。前向传播仅保存了 $\mathbf{O}$、$\mathbf{m}$、$\boldsymbol{\ell}$ 以及随机数种子 $\mathcal{R}$，反向传播必须在 SRAM 中重计算所需的局部 $\mathbf{P}_{ij}$。

### 2.1 符号定义

在反向传播中，符号与前向传播保持一致，同时引入损失函数 $\mathcal{L}$ 对各矩阵的梯度：

$\mathbf{P}$ 是 Attention 权重矩阵，定义为 $\mathbf{P} = \text{softmax}(\mathbf{Q}\mathbf{K}^\top) \in \mathbb{R}^{N \times N}$。它的第 $i$ 行第 $j$ 列元素 $P_{ij} = \exp(q_i^\top k_j) / L_i$，表示第 $i$ 个 query 对第 $j$ 个 key 的注意力权重。在标准 Attention 中，$\mathbf{P}$ 需要完整保存；但在 FlashAttention 中，$\mathbf{P}$ 不保存，反向时在 SRAM 中逐块重计算。

$\mathbf{V}$ 是 Value 矩阵，属于 $\mathbb{R}^{N \times d}$。它的第 $j$ 行 $\nu_j$ 是第 $j$ 个 token 的 value 向量。

$\mathbf{O}$ 是 Attention 输出矩阵，属于 $\mathbb{R}^{N \times d}$，且 $\mathbf{O} = \mathbf{P}\mathbf{V}$。前向传播结束时，$\mathbf{O}$ 被保存下来用于反向传播。

$\frac{\partial \mathcal{L}}{\partial \mathbf{O}} \in \mathbb{R}^{N \times d}$ 是损失函数 $\mathcal{L}$ 对 $\mathbf{O}$ 的梯度。它由神经网络的后续层（或损失函数）反向传播得到，是 FlashAttention 反向传播的输入。为书写简洁，记 $\mathbf{dO} = \frac{\partial \mathcal{L}}{\partial \mathbf{O}}$。

同理，记 $\mathbf{dV} = \frac{\partial \mathcal{L}}{\partial \mathbf{V}}$，$\mathbf{dQ} = \frac{\partial \mathcal{L}}{\partial \mathbf{Q}}$，$\mathbf{dK} = \frac{\partial \mathcal{L}}{\partial \mathbf{K}}$，$\mathbf{dP} = \frac{\partial \mathcal{L}}{\partial \mathbf{P}}$，$\mathbf{dS} = \frac{\partial \mathcal{L}}{\partial \mathbf{S}}$。

### 2.2 Algorithm 4 逐行详解

Algorithm 4 的输入包括：$\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{dO} \in \mathbb{R}^{N \times d}$ 存储在 HBM；$\boldsymbol{\ell}, \mathbf{m} \in \mathbb{R}^N$ 存储在 HBM（前向保存的 softmax 统计量）；SRAM 容量 $M$；softmax 缩放常数 $\tau$；mask 函数；dropout 概率 $p_{\text{drop}}$；前向保存的伪随机数生成器状态 $\mathcal{R}$。

**第 1 行**：将伪随机数生成器状态恢复为 $\mathcal{R}$

$$\text{Set RNG state to } \mathcal{R}$$

这一步确保反向传播中重新生成的 dropout mask 与前向传播完全一致，从而无需保存 $N \times N$ 的 dropout mask 矩阵。

**第 2 行**：设置块大小

$$B_c = \left\lceil \frac{M}{4d} \right\rceil, \quad B_r = \min\left(\left\lceil \frac{M}{4d} \right\rceil,\ d\right)$$

块大小设置与前向 Algorithm 1 完全一致。SRAM 需要同时容纳 $\mathbf{K}_j, \mathbf{V}_j, \mathbf{Q}_i, \mathbf{O}_i, \mathbf{dO}_i, \mathbf{dQ}_i$ 以及重计算的 $\mathbf{S}_{ij}, \mathbf{P}_{ij}$ 等。

**第 3 行**：输入矩阵分块

将 $\mathbf{Q}$ 沿行方向分为 $T_r = \lceil N / B_r \rceil$ 块 $\mathbf{Q}_1, \dots, \mathbf{Q}_{T_r}$，每块尺寸 $B_r \times d$。将 $\mathbf{K}$ 和 $\mathbf{V}$ 沿行方向分为 $T_c = \lceil N / B_c \rceil$ 块 $\mathbf{K}_1, \dots, \mathbf{K}_{T_c}$ 和 $\mathbf{V}_1, \dots, \mathbf{V}_{T_c}$，每块尺寸 $B_c \times d$。

**第 4 行**：输出与梯度分块

将 $\mathbf{O}$ 沿行方向分为 $T_r$ 块 $\mathbf{O}_1, \dots, \mathbf{O}_{T_r}$，每块 $B_r \times d$。将 $\mathbf{dO}$ 沿行方向分为 $T_r$ 块 $\mathbf{dO}_1, \dots, \mathbf{dO}_{T_r}$，每块 $B_r \times d$。将 $\boldsymbol{\ell}$ 分为 $T_r$ 块 $\ell_1, \dots, \ell_{T_r}$，每块 $B_r$。将 $\mathbf{m}$ 分为 $T_r$ 块 $m_1, \dots, m_{T_r}$，每块 $B_r$。

**第 5 行**：初始化梯度矩阵并分块

$$\mathbf{dQ} = \mathbf{0}_{N \times d} \in \mathbb{R}^{N \times d}, \quad \mathbf{dK} = \mathbf{0}_{N \times d} \in \mathbb{R}^{N \times d}, \quad \mathbf{dV} = \mathbf{0}_{N \times d} \in \mathbb{R}^{N \times d}$$

三者均存储在 HBM 中。将 $\mathbf{dQ}$ 沿行方向分为 $T_r$ 块 $\mathbf{dQ}_1, \dots, \mathbf{dQ}_{T_r}$，每块 $B_r \times d$。将 $\mathbf{dK}$ 和 $\mathbf{dV}$ 沿行方向分为 $T_c$ 块 $\mathbf{dK}_1, \dots, \mathbf{dK}_{T_c}$ 和 $\mathbf{dV}_1, \dots, \mathbf{dV}_{T_c}$，每块 $B_c \times d$。

**第 6 行**：外层循环开始

$$\text{for } j = 1 \text{ to } T_c \text{ do}$$

外层循环遍历 $\mathbf{K}$ 和 $\mathbf{V}$ 的分块。每轮迭代处理一个 $\mathbf{K}_j$ 和一个 $\mathbf{V}_j$，计算它们对 $\mathbf{dK}$ 和 $\mathbf{dV}$ 的贡献。

**第 7 行**：加载 $\mathbf{K}_j, \mathbf{V}_j$ 到 SRAM

将 $\mathbf{K}_j$（$B_c \times d$）和 $\mathbf{V}_j$（$B_c \times d$）从 HBM 加载到 on-chip SRAM。这一步在整个内层循环中只执行一次。

**第 8 行**：初始化局部梯度块

$$\tilde{\mathbf{dK}}_j = \mathbf{0}_{B_c \times d}, \quad \tilde{\mathbf{dV}}_j = \mathbf{0}_{B_c \times d}$$

在 SRAM 中为当前 $\mathbf{K}_j$ 和 $\mathbf{V}_j$ 对应的梯度累加器分配空间并初始化为零。

**第 9 行**：内层循环开始

$$\text{for } i = 1 \text{ to } T_r \text{ do}$$

内层循环遍历 $\mathbf{Q}$ 的分块。每轮迭代处理一个 $\mathbf{Q}_i$ 块，重计算对应的局部 $\mathbf{P}_{ij}$，并更新 $\mathbf{dQ}_i, \tilde{\mathbf{dK}}_j, \tilde{\mathbf{dV}}_j$。

**第 10 行**：加载 $\mathbf{Q}_i, \mathbf{O}_i, \mathbf{dO}_i, \mathbf{dQ}_i, \ell_i, m_i$ 到 SRAM

将 $\mathbf{Q}_i$（$B_r \times d$）、$\mathbf{O}_i$（$B_r \times d$）、$\mathbf{dO}_i$（$B_r \times d$）、$\mathbf{dQ}_i$（$B_r \times d$）、$\ell_i$（$B_r$）、$m_i$（$B_r$）从 HBM 加载到 SRAM。

**第 11 行**：在 SRAM 中重计算局部 score 矩阵

$$\mathbf{S}_{ij} = \tau \mathbf{Q}_i \mathbf{K}_j^\top \in \mathbb{R}^{B_r \times B_c}$$

与前向第 9 行完全一致，在 SRAM 中重新计算 $\mathbf{Q}_i$ 与 $\mathbf{K}_j$ 的 score。该块仅在 SRAM 中临时存在，**绝不写入 HBM**。

**第 12 行**：在 SRAM 中应用 mask

$$\mathbf{S}_{ij}^{\text{masked}} = \text{mask}(\mathbf{S}_{ij}) \in \mathbb{R}^{B_r \times B_c}$$

对 score 矩阵应用 mask（如 causal mask 或 padding mask），将需要屏蔽的位置设为 $-\infty$。

**第 13 行**：在 SRAM 中重计算概率矩阵 $\mathbf{P}_{ij}$

$$\mathbf{P}_{ij} = \text{diag}(\ell_i)^{-1} \exp(\mathbf{S}_{ij}^{\text{masked}} - m_i) \in \mathbb{R}^{B_r \times B_c}$$

这是反向传播的核心：利用前向保存的统计量 $(\ell_i, m_i)$ 在 SRAM 中精确恢复出概率矩阵 $\mathbf{P}_{ij}$，而无需从 HBM 读取巨大的 $N \times N$ 矩阵 $\mathbf{P}$。

推导如下：对第 $r$ 行（$1 \leq r \leq B_r$），全局 softmax 的第 $c$ 个元素为：

$$P_{i,r,c} = \frac{\exp(S_{i,r,c}^{\text{masked}} - m_{i,r})}{\ell_{i,r}}$$

其中 $m_{i,r}$ 是第 $i$ 块第 $r$ 行的全局最大值，$\ell_{i,r}$ 是对应的全局 EXP 和。矩阵形式即为 $\text{diag}(\ell_i)^{-1} \exp(\mathbf{S}_{ij}^{\text{masked}} - m_i)$，其中 $m_i$ 通过广播机制逐行相减。

**第 14 行**：在 SRAM 中重计算 dropout mask

$$\mathbf{Z}_{ij} \in \mathbb{R}^{B_r \times B_c}, \quad Z_{ij,r,c} = \begin{cases} \frac{1}{1-p_{\text{drop}}} & \text{with prob. } 1-p_{\text{drop}} \\ 0 & \text{with prob. } p_{\text{drop}} \end{cases}$$

利用恢复的随机数种子 $\mathcal{R}$，生成与前向完全相同的 dropout mask。每个元素以概率 $1-p_{\text{drop}}$ 取值为 $\frac{1}{1-p_{\text{drop}}}$（保证期望为 1），以概率 $p_{\text{drop}}$ 取值为 0。

**第 15 行**：在 SRAM 中应用 dropout

$$\mathbf{P}_{ij}^{\text{dropped}} = \mathbf{P}_{ij} \circ \mathbf{Z}_{ij} \in \mathbb{R}^{B_r \times B_c}$$

其中 $\circ$ 表示逐元素乘法（Hadamard 积）。这是前向 dropout 操作的精确重播。

**第 16 行**：在 SRAM 中累加 $\mathbf{dV}_j$

$$\tilde{\mathbf{dV}}_j \leftarrow \tilde{\mathbf{dV}}_j + (\mathbf{P}_{ij}^{\text{dropped}})^\top \mathbf{dO}_i \in \mathbb{R}^{B_c \times d}$$

数学原理：由前向关系 $\mathbf{O} = \mathbf{P}^{\text{dropped}} \mathbf{V}$，对 $\mathbf{V}$ 求导得 $\mathbf{dV} = (\mathbf{P}^{\text{dropped}})^\top \mathbf{dO}$。在分块形式下，$\mathbf{O}_i = \sum_{j'} \mathbf{P}_{ij'}^{\text{dropped}} \mathbf{V}_{j'}$，因此 $\mathbf{V}_j$ 对 $\mathbf{O}_i$ 的贡献为 $\mathbf{P}_{ij}^{\text{dropped}} \mathbf{V}_j$，其对 $\mathbf{dV}_j$ 的梯度贡献为 $(\mathbf{P}_{ij}^{\text{dropped}})^\top \mathbf{dO}_i$。遍历所有 $i$ 块后，即得到完整的 $\mathbf{dV}_j$。

**第 17 行**：在 SRAM 中计算 $\mathbf{dP}_{ij}^{\text{dropped}}$

$$\mathbf{dP}_{ij}^{\text{dropped}} = \mathbf{dO}_i \mathbf{V}_j^\top \in \mathbb{R}^{B_r \times B_c}$$

数学原理：由 $\mathbf{O}_i = \mathbf{P}_{ij}^{\text{dropped}} \mathbf{V}_j + \text{other blocks}$，对 $\mathbf{P}_{ij}^{\text{dropped}}$ 求导得 $\frac{\partial \mathcal{L}}{\partial \mathbf{P}_{ij}^{\text{dropped}}} = \mathbf{dO}_i \mathbf{V}_j^\top$。这是矩阵求导的链式法则：若 $\mathbf{O}_i = \mathbf{P}_{ij}^{\text{dropped}} \mathbf{V}_j$，则 $d\mathcal{L} = \text{tr}(\mathbf{dO}_i^\top d\mathbf{O}_i) = \text{tr}(\mathbf{dO}_i^\top d\mathbf{P}_{ij}^{\text{dropped}} \mathbf{V}_j) = \text{tr}((\mathbf{dO}_i \mathbf{V}_j^\top)^\top d\mathbf{P}_{ij}^{\text{dropped}})$。

**第 18 行**：在 SRAM 中还原 dropout 梯度

$$\mathbf{dP}_{ij} = \mathbf{dP}_{ij}^{\text{dropped}} \circ \mathbf{Z}_{ij} \in \mathbb{R}^{B_r \times B_c}$$

由于前向时 $\mathbf{P}_{ij}^{\text{dropped}} = \mathbf{P}_{ij} \circ \mathbf{Z}_{ij}$，反向传播需要乘回相同的 mask $\mathbf{Z}_{ij}$。注意 $\mathbf{Z}_{ij}$ 中非零元素为 $\frac{1}{1-p_{\text{drop}}}$，因此这一步同时完成了梯度缩放。

**第 19 行**：在 SRAM 中计算标量 $\mathbf{D}_i$

$$\mathbf{D}_i = \text{rowsum}(\mathbf{dO}_i \circ \mathbf{O}_i) \in \mathbb{R}^{B_r}$$

这一步是反向 softmax 梯度的核心简化。以下给出详细推导。

由 $\mathbf{P} = \text{softmax}(\mathbf{S})$，softmax 的 Jacobian 对第 $r$ 行给出：

$$dS_{i,r,c} = P_{i,r,c} \cdot dP_{i,r,c} - P_{i,r,c} \cdot \sum_{c'} P_{i,r,c'} \cdot dP_{i,r,c'} \quad (31)$$

定义标量 $D_{i,r} = \sum_{c'} P_{i,r,c'} \cdot dP_{i,r,c'}$，则：

$$dS_{i,r,c} = P_{i,r,c} (dP_{i,r,c} - D_{i,r}) \quad (32)$$

现在推导 $D_{i,r}$ 的简化计算。由第 17 行，$dP_{i,r,c} = \sum_{k=1}^{d} dO_{i,r,k} \cdot V_{j,c,k}$（即 $\mathbf{dO}_i \mathbf{V}_j^\top$ 的 $(r,c)$ 元素）。因此：

$$D_{i,r} = \sum_{c} P_{i,r,c} \cdot dP_{i,r,c} = \sum_{c} P_{i,r,c} \sum_{k} dO_{i,r,k} V_{j,c,k} \quad (33)$$

交换求和顺序：

$$D_{i,r} = \sum_{k} dO_{i,r,k} \sum_{c} P_{i,r,c} V_{j,c,k} \quad (34)$$

注意到 $\sum_{c} P_{i,r,c} V_{j,c,k}$ 是前向输出 $\mathbf{O}_i$ 的第 $(r,k)$ 元素 $O_{i,r,k}$（因为 $\mathbf{O} = \mathbf{P}\mathbf{V}$，且当前块 $\mathbf{P}_{ij}$ 只是 $\mathbf{P}$ 的一部分，但所有块的加权和构成完整输出）。因此：

$$D_{i,r} = \sum_{k} dO_{i,r,k} \cdot O_{i,r,k} = \text{rowsum}(\mathbf{dO}_i \circ \mathbf{O}_i)_r \quad (35)$$

即 $\mathbf{D}_i$ 的第 $r$ 个分量等于 $\mathbf{dO}_i$ 的第 $r$ 行与 $\mathbf{O}_i$ 的第 $r$ 行的逐元素乘积之和。这仅需两个长度为 $d$ 的向量点积，完全避免了对 $N$ 维向量 $\mathbf{P}_{i:}$ 的存储与遍历。

**第 20 行**：在 SRAM 中计算 $\mathbf{dS}_{ij}$

$$\mathbf{dS}_{ij} = \mathbf{P}_{ij} \circ (\mathbf{dP}_{ij} - \mathbf{D}_i) \in \mathbb{R}^{B_r \times B_c}$$

其中 $\mathbf{D}_i \in \mathbb{R}^{B_r}$ 通过广播机制逐行相减：对第 $r$ 行，$\mathbf{dP}_{ij,r,:} - D_{i,r}$，再逐元素乘以 $\mathbf{P}_{ij,r,:}$。这直接对应公式 (32) 的矩阵形式，是 softmax 梯度的分块实现。

**第 21 行**：在 SRAM 中更新 $\mathbf{dQ}_i$ 并写回 HBM

$$\mathbf{dQ}_i \leftarrow \mathbf{dQ}_i + \tau \mathbf{dS}_{ij} \mathbf{K}_j \in \mathbb{R}^{B_r \times d}$$

数学原理：由 $\mathbf{S}_{ij} = \tau \mathbf{Q}_i \mathbf{K}_j^\top$，对 $\mathbf{Q}_i$ 求导得 $\frac{\partial \mathcal{L}}{\partial \mathbf{Q}_i} = \tau \mathbf{dS}_{ij} \mathbf{K}_j$。由于 $\mathbf{Q}_i$ 可能参与多个块的计算（遍历所有 $j$），因此使用累加 $\leftarrow$ 而非赋值 $=$。计算完成后写回 HBM。

**第 22 行**：在 SRAM 中更新 $\tilde{\mathbf{dK}}_j$

$$\tilde{\mathbf{dK}}_j \leftarrow \tilde{\mathbf{dK}}_j + \tau \mathbf{dS}_{ij}^\top \mathbf{Q}_i \in \mathbb{R}^{B_c \times d}$$

数学原理：由 $\mathbf{S}_{ij} = \tau \mathbf{Q}_i \mathbf{K}_j^\top$，对 $\mathbf{K}_j$ 求导得 $\frac{\partial \mathcal{L}}{\partial \mathbf{K}_j} = \tau \mathbf{dS}_{ij}^\top \mathbf{Q}_i$。由于 $\mathbf{K}_j$ 可能参与多个 $\mathbf{Q}_i$ 块的计算（遍历所有 $i$），因此使用累加。注意 $\tilde{\mathbf{dK}}_j$ 暂存在 SRAM 中，待内层循环结束后再统一写回 HBM。

**第 23 行**：end for（内层循环结束）

**第 24 行**：将 $\tilde{\mathbf{dK}}_j, \tilde{\mathbf{dV}}_j$ 写回 HBM

$$\mathbf{dK}_j \leftarrow \tilde{\mathbf{dK}}_j, \quad \mathbf{dV}_j \leftarrow \tilde{\mathbf{dV}}_j$$

内层循环结束后，当前 $\mathbf{K}_j$ 和 $\mathbf{V}_j$ 对应的完整梯度已计算完毕，从 SRAM 写回 HBM。

**第 25 行**：end for（外层循环结束）

**第 26 行**：Return $\mathbf{dQ}, \mathbf{dK}, \mathbf{dV}$

最终返回三个梯度矩阵。虽然反向传播增加了约 $10\%$ 到 $15\%$ 的浮点运算量（由于重计算 $\mathbf{P}_{ij}$），但由于避免了从 HBM 读取巨大的 $N \times N$ 矩阵 $\mathbf{P}$，整体 wall-clock 时间仍显著快于标准实现。前向与反向的额外内存开销均从 $O(N^2)$ 降至 $O(N)$。
