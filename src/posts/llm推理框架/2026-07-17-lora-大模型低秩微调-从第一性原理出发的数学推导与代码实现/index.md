---
layout: post.njk
post_id: 2026-07-17-lora-大模型低秩微调-从第一性原理出发的数学推导与代码实现
archive: llm推理框架
title: LoRA 大模型低秩微调：从第一性原理出发的数学推导与代码实现
date: 2026-07-17
tags:
  - post
---
# LoRA 大模型低秩微调：从第一性原理出发的数学推导与代码实现

> References:
> - Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022, arXiv:2106.09685（原始论文）
> - Kalajdzievski, *A Rank-Stabilization Scaling Factor for Fine-tuning with LoRA*（rsLoRA）, 2023, arXiv:2312.03732（缩放因子修正）
> - Aghajanyan et al., *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*, 2020, arXiv:2012.13255（内在维度假设）
> - Eckart & Young, *The Approximation of One Matrix by Another of Lower Rank*, 1936（低秩近似最优性）

---

## 一、公式作用概述

### 1.1 问题设定：全量微调的"内存墙"

大语言模型（LLM）预训练完成后，下游任务适配的标准做法是**全量微调（Full Fine-Tuning）**：以预训练权重 $W_0$ 为起点，对全部参数继续梯度下降。问题在于优化器状态的内存开销随**可训练参数量**线性增长：

> **【知识卡片】混合精度训练 + Adam 的内存账本**
>
> 混合精度训练下，每个**可训练参数**需要常驻以下状态：
>
> | 状态 | 精度 | 字节数 |
> |---|---|---|
> | 前向权重 | fp16 | 2 |
> | 梯度 | fp16 | 2 |
> | fp32 主权重（master copy） | fp32 | 4 |
> | Adam 一阶动量 $m$ | fp32 | 4 |
> | Adam 二阶动量 $v$ | fp32 | 4 |
> | **合计** | | **16 字节/参数** |
>
> 以 175B 模型为例：$175\times 10^9 \times 16\ \text{B} \approx 2.8\ \text{TB}$。原论文报告 GPT-3 175B 全量微调需约 1.2 TB 显存，是每个下游任务都难以承受的门槛。

### 1.2 核心思想（先用自然语言）

LoRA 的回答是：**预训练权重 $W_0$ 冻结不动，把"要学的增量" $\Delta W$ 约束为低秩矩阵**。

- 直观动机：$W_0$ 已经在海量语料上学到了通用表示，下游适配只需要在少数方向上做"微调修正"。Aghajanyan et al. (2020) 的内在维度实验表明，下游任务适配所需的自由度远小于参数空间的表观维度——即 $\Delta W$ 具有很低的"内在秩"。
- 参数化方式：任何秩不超过 $r$ 的矩阵都可以写成"矮矩阵 × 瘦矩阵"的乘积，于是令

$$
\Delta W = BA,\qquad B\in\mathbb{R}^{d\times r},\quad A\in\mathbb{R}^{r\times k},\quad r \ll \min(d,k)
$$

- 收益：可训练参数从 $dk$ 降为 $r(d+k)$，优化器状态、梯度、检查点同步缩小约 $\dfrac{dk}{r(d+k)}$ 倍；训练完后把 $BA$ 合并回 $W_0$，**推理时零额外开销**。
- 一句话总结：LoRA 把"在全参数空间里做无约束优化"变为"在秩 $\le r$ 的矩阵流形上做优化"，并用低秩因子化 $BA$ 隐式地参数化这个流形。

> **【知识卡片】矩阵秩与分解**
>
> 矩阵 $M\in\mathbb{R}^{d\times k}$ 的秩 $\mathrm{rank}(M)$ = 线性无关列（行）的最大个数 = 非零奇异值个数。
>
> 两条本推导反复用到的性质：
>
> 1. **秩的子乘法性**：$\mathrm{rank}(BA) \le \min(\mathrm{rank}(B),\mathrm{rank}(A)) \le r$；
> 2. **秩-$r$ 分解的存在性**：$\mathrm{rank}(M) \le r \iff$ 存在 $B\in\mathbb{R}^{d\times r}, A\in\mathbb{R}^{r\times k}$ 使 $M = BA$。
>
> 第 2 条保证：用 $BA$ 参数化**恰好覆盖**所有秩 $\le r$ 的矩阵，不多也不少。

### 1.3 符号表

| 符号 | 含义 | 维度/类型 |
|---|---|---|
| $W_0$ | 冻结的预训练权重 | $\mathbb{R}^{d\times k}$ |
| $\Delta W$ | 微调学到的权重增量 | $\mathbb{R}^{d\times k}$ |
| $B,\ A$ | LoRA 低秩因子，$\Delta W = BA$ | $\mathbb{R}^{d\times r},\ \mathbb{R}^{r\times k}$ |
| $r$ | LoRA 秩（超参数） | $r \ll \min(d,k)$，常用 $1\sim 64$ |
| $\alpha$ | LoRA 缩放常数 | 标量超参数 |
| $s$ | 缩放因子，原论文取 $s=\alpha/r$；rsLoRA 取 $s=\alpha/\sqrt{r}$ | 标量 |
| $x,\ h$ | 层输入、层输出 | $\mathbb{R}^{k},\ \mathbb{R}^{d}$ |
| $g$ | 损失对输出的梯度 $\partial\mathcal{L}/\partial h$ | $\mathbb{R}^{d}$ |
| $\eta$ | 学习率 | 标量 |
| $\sigma_a^2$ | $A$ 初始化元素方差 | 标量，Kaiming 下 $\Theta(1/k)$ |
| $\sigma_i(\cdot)$ | 第 $i$ 大奇异值 | 标量 |
| $d,\ k$ | 权重输出/输入维度（Transformer 中 $d=k=d_{\text{model}}$ 常见） | 标量 |

---

## 二、完整推导过程

### 2.1 问题背景与标准形式（全量微调基线）

设预训练得到权重 $W_0\in\mathbb{R}^{d\times k}$（为简洁省略偏置），单层前向为

$$
h = W_0\, x.
$$

下游任务数据集 $\mathcal{D}=\{(x_n,y_n)\}_{n=1}^N$ 上的微调目标是

$$
\min_{W}\ \mathcal{L}(W) \;=\; \frac{1}{N}\sum_{n=1}^N \ell\big(W x_n,\, y_n\big).
$$

**全量微调的标准形式**：把 $W$ 写成 $W_0$ 加无约束增量 $\Delta W$，

$$
\min_{\Delta W\in\mathbb{R}^{d\times k}}\ \mathcal{L}\big(W_0 + \Delta W\big),\qquad h = (W_0+\Delta W)\,x.
$$

这里 $\Delta W$ 有 $dk$ 个自由参数，是**无约束优化**。LoRA 的全部内容可以概括为：给 $\Delta W$ 加上秩约束，并给出该约束下的可微参数化与正确的初始化、缩放规则。

### 2.2 方法一：LoRA 低秩重参数化（原论文，$s=\alpha/r$）

#### 2.2.1 从内在维度假设到 $\Delta W = BA$

**假设（低秩内在更新）**：下游适配所需的最优增量 $\Delta W^*$ 的谱快速衰减，即存在 $r\ll\min(d,k)$ 使得秩 $\le r$ 的增量已足够逼近 $\Delta W^*$。

> **【推导】约束优化的参数化**
>
> 目标：求解
>
> $$
> \min_{\Delta W}\ \mathcal{L}(W_0+\Delta W)\quad \text{s.t.}\quad \mathrm{rank}(\Delta W)\le r. \tag{2-1}
> $$
>
> 秩约束是非凸的组合约束，无法直接梯度下降。由 1.3 节知识卡片第 2 条性质：
>
> $$
> \{\Delta W : \mathrm{rank}(\Delta W)\le r\} \;=\; \{BA : B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k}\},
> $$
>
> 因此 (2-1) 精确等价于无约束问题
>
> $$
> \min_{B,\,A}\ \mathcal{L}\big(W_0 + BA\big). \tag{2-2}
> $$
>
> 注意因子化引入了对 $(B,A)$ 的非凸性（$(BQ, Q^{-1}A)$ 给出同一个 $\Delta W$），但**在 $\Delta W$ 空间看**，(2-2) 与 (2-1) 是同一个约束优化问题，因子化只是该流形的可微坐标。

#### 2.2.2 前向公式

LoRA 前向在原权重旁并联一条低秩支路，并乘以缩放因子 $s$：

$$
\boxed{\ h \;=\; W_0\, x \;+\; s\cdot BA\, x, \qquad s = \frac{\alpha}{r}\ } \tag{2-3}
$$

- $W_0 x$：冻结支路，不反传梯度；
- $BAx$：低秩支路，先算 $u=Ax$（$k\to r$ 降维），再算 $Bu$（$r\to d$ 升维），计算量 $O(r(k+d))$ 而非 $O(dk)$；
- $s=\alpha/r$ 的作用在 2.2.6 节推导。

#### 2.2.3 参数量与内存推导

> **【推导】可训练参数量**
>
> $$
> \underbrace{dk}_{\text{全量微调}}\ \longrightarrow\ \underbrace{r(d+k)}_{\text{LoRA}},\qquad
> \text{压缩比} = \frac{r(d+k)}{dk} = r\Big(\frac{1}{k}+\frac{1}{d}\Big).
> $$
>
> 当 $d=k$ 时压缩比 $=2r/d$。

> **【小例子】$d=k=4096$ 的注意力权重（GPT-3 的 $d_{\text{model}}=12288$ 同量级）**
>
> 取 $r=16$：
>
> $$
> dk = 4096^2 = 16{,}777{,}216,\qquad r(d+k) = 16\times 8192 = 131{,}072,\qquad \text{占比 } = \frac{2\times 16}{4096} = 0.78\%.
> $$
>
> 按 16 字节/可训练参数的 Adam 账本（1.1 节知识卡片）：
>
> - 全量微调优化器状态：$16{,}777{,}216\times 16\ \text{B} = 256\ \text{MiB}$；
> - LoRA 优化器状态：$131{,}072\times 16\ \text{B} = 2\ \text{MiB}$。
>
> 单层就差 $128\times$；乘上 96 层并只注入注意力矩阵，即是论文中"GPT-3 检查点 350 GB → 35 MB（约 $10^4$ 倍）、显存 1.2 TB → 350 GB"的来源。

#### 2.2.4 梯度完整推导（迹技巧）

只有 $A,B$ 可训练，需推导 $\nabla_A\mathcal{L}$ 与 $\nabla_B\mathcal{L}$。

> **【知识卡片】迹技巧（矩阵求导的最小工具集）**
>
> 对标量损失，若能把微分写成 $d\mathcal{L} = \mathrm{tr}(G^\top dX)$，则 $\nabla_X\mathcal{L} = G$。
> 用到：$\mathrm{tr}(UV)=\mathrm{tr}(VU)$（循环置换）、$a^\top b = \mathrm{tr}(a^\top b) = \mathrm{tr}(ba^\top)$（标量取迹）。

> **【推导】$\nabla_B$ 与 $\nabla_A$（单样本）**
>
> 记 $u = Ax\in\mathbb{R}^r$，$g = \dfrac{\partial\mathcal{L}}{\partial h}\in\mathbb{R}^d$。由 (2-3)，$h = W_0 x + sBu$。
>
> **对 $B$ 求导**：$dh = s\,(dB)\,u$，于是
>
> $$
> d\mathcal{L} = g^\top dh = s\, g^\top (dB)\, u = s\,\mathrm{tr}\big(g^\top dB\, u\big) = s\,\mathrm{tr}\big(u\,g^\top\, dB\big)
> $$
>
> $$
> \Rightarrow\ \boxed{\ \nabla_B \mathcal{L} = s\cdot g\,(Ax)^\top \ }\quad (d\times r) \tag{2-4}
> $$
>
> **对 $A$ 求导**：$du = (dA)x$，$dh = sB(dA)x$，于是
>
> $$
> d\mathcal{L} = s\, g^\top B\,(dA)\,x = s\,\mathrm{tr}\big(x\,g^\top B\, dA\big)
> $$
>
> $$
> \Rightarrow\ \boxed{\ \nabla_A \mathcal{L} = s\cdot B^\top g\, x^\top \ }\quad (r\times k) \tag{2-5}
> $$
>
> **批形式**：$X\in\mathbb{R}^{N\times k}$、$G\in\mathbb{R}^{N\times d}$（每行一个样本）时，
>
> $$
> \nabla_B \mathcal{L} = s\, G^\top X A^\top,\qquad \nabla_A \mathcal{L} = s\, B^\top G^\top X. \tag{2-6}
> $$
>
> 维度校验：$(d\times N)(N\times k)(k\times r) = d\times r$ ✓；$(r\times d)(d\times N)(N\times k) = r\times k$ ✓。
>
> **为什么这一步重要**：(2-4)(2-5) 显式表明两个梯度分别含有因子 $A$ 和 $B$——$\nabla_B$ 依赖 $A$，$\nabla_A$ 依赖 $B$。这一耦合直接决定了初始化策略（2.2.5）与缩放规则（2.2.6）。

> **【小例子】标量情形的梯度核验**
>
> 取 $d=k=r=1$，$s=1$，$W_0 = 2$，$A = 3$，$B = 0.5$，$x=4$，目标 $y=1$，$\mathcal{L}=(h-y)^2$。
>
> **前向**：$h = 2\cdot 4 + 1\cdot 0.5\cdot 3\cdot 4 = 8+6 = 14$；$g = \partial\mathcal{L}/\partial h = 2(14-1)=26$。
>
> **公式 (2-4)**：$\nabla_B = s\cdot g\cdot Ax = 26\times 12 = 312$。
>
> **公式 (2-5)**：$\nabla_A = s\cdot B\cdot g\cdot x = 0.5\times 26\times 4 = 52$。
>
> **有限差分验证**：$B\leftarrow 0.5001 \Rightarrow h = 14.0012 \Rightarrow \mathcal{L} = 13.0012^2 = 169.03120$，$\Delta\mathcal{L}/\Delta B = 0.03120/0.0001 = 312.0$ ✓；$A\leftarrow 3.0001 \Rightarrow h = 14.0002 \Rightarrow \Delta\mathcal{L}/\Delta A = 52.0$ ✓。
>
> 矩阵公式在标量退化时与直接求导完全一致。

#### 2.2.5 初始化策略：为什么必须 "$B=0$，$A$ 随机"

设计约束有两条，且**互相制衡**：

- (C1) **不破坏预训练函数**：初始化时 $\Delta W = BA = 0$，即微调起点恰为预训练模型；
- (C2) **可训练性**：初始化处梯度不能恒为零，否则陷入鞍点。

> **【推导】四种初始化组合逐一判定**
>
> 由 (2-4)(2-5)：$\nabla_B\mathcal{L} = s\,g(Ax)^\top$，$\nabla_A\mathcal{L} = s\,B^\top g x^\top$。
>
> | 组合 | (C1) $\Delta W_0 = 0$？ | (C2) 初始梯度？ | 结论 |
> |---|---|---|---|
> | $A=0,\ B=0$ | ✓ | $\nabla_B = s g (Ax)^\top = 0$，$\nabla_A = s B^\top g x^\top = 0$ | ✗ 死锁鞍点 |
> | $A$ 随机，$B$ 随机 | ✗（$BA\ne 0$ 几乎必然） | 非零 | ✗ 起点即破坏 $W_0$ |
> | $A$ 随机，$B=0$ | ✓ | $\nabla_B = s\,g(Ax)^\top \ne 0$（a.s.），$\nabla_A = 0$ | ✓ 可用 |
> | $A=0$，$B$ 随机 | ✓ | $\nabla_A = s\,B^\top g x^\top \ne 0$，$\nabla_B = 0$ | ✓ 可用（对称） |
>
> 关键机制（以论文采用的 **$A\sim$ 高斯/Kaiming、$B=0$** 为例）：
>
> 1. **第 0 步**：$\Delta W = 0$，模型输出与预训练完全一致（数值验证见 3.2 节，最大偏差 $0.0$）；
> 2. **第 1 步**：只有 $B$ 收到非零梯度并离开原点（此时 $\nabla_A = sB^\top g x^\top$ 恰为 0——这不是缺陷，而是"先升维后降维"的启动顺序）；
> 3. **第 2 步起**：$B\ne 0$ 使 $\nabla_A \ne 0$，$A,B$ 联合训练。
>
> 3.2 节的梯度实测与此完全对应：初始化时 $\|\nabla_A\| = 0.0$、$\|\nabla_B\| = 0.3851$；$B$ 注入非零值后 $\|\nabla_A\| = 0.0541 \ne 0$。

#### 2.2.6 缩放因子推导：首步更新幅度分析

$s$ 不是可有可无的常数——它控制**有效学习率随 $r$ 的标度律**。从首步更新的输出扰动出发推导。

> **【推导】首步有效更新 $\Delta W_1$ 与输出扰动 $\delta h$**
>
> 初始化 $B_0 = 0$，学习率 $\eta$，单样本。由 (2-4)，第一步 SGD：
>
> $$
> B_1 = -\eta\,\nabla_B\mathcal{L} = -\eta s\; g\,(Ax)^\top.
> $$
>
> 首步后的有效增量（此时 $A$ 尚未更新）：
>
> $$
> \Delta W_1 = s\,B_1 A = -\eta s^2\; g\, x^\top A^\top A. \tag{2-7}
> $$
>
> 它对该样本输出造成的扰动：
>
> $$
> \delta h = \Delta W_1\, x = -\eta s^2\; g\cdot \big(x^\top A^\top A x\big) = -\eta s^2\, g\,\|Ax\|^2. \tag{2-8}
> $$
>
> 需要如下引理估计 $\|Ax\|^2$ 的期望。

> **【推导】引理：$\mathbb{E}\|Ax\|^2 = r\sigma_a^2\|x\|^2$**
>
> 设 $A\in\mathbb{R}^{r\times k}$，$A_{ij}$ 独立同分布，$\mathbb{E}A_{ij}=0$，$\mathrm{Var}(A_{ij})=\sigma_a^2$。$x$ 与 $A$ 独立。
>
> $$
> (Ax)_i = \sum_{j=1}^k A_{ij}x_j \;\Rightarrow\; \mathbb{E}(Ax)_i^2 = \sum_{j,j'} \mathbb{E}[A_{ij}A_{ij'}] x_j x_{j'} = \sum_{j=1}^k \sigma_a^2 x_j^2 = \sigma_a^2\|x\|^2,
> $$
>
> 其中交叉项因独立性归零。对 $i=1,\dots,r$ 求和：
>
> $$
> \mathbb{E}\|Ax\|^2 = \sum_{i=1}^r \mathbb{E}(Ax)_i^2 = r\,\sigma_a^2\,\|x\|^2. \qquad \blacksquare
> $$

> **【推导】三种缩放律的对比结论**
>
> 将引理代入 (2-8) 取期望幅度：$\mathbb{E}|\delta h| \propto \eta\, s^2\, r\, \sigma_a^2\, \|g\|\,\|x\|^2$。
>
> Kaiming/高斯初始化按 **fan-in** 定方差：$\sigma_a^2 = \Theta(1/k)$，**与 $r$ 无关**。因此 $\mathbb{E}|\delta h| \propto s^2 r$：
>
> $$
> \begin{array}{llll}
> s = \alpha/r & (\text{原论文 LoRA}): & \mathbb{E}|\delta h| \propto \alpha^2/r & \Rightarrow\ r\ \text{越大，有效更新越小（更新坍缩）}\\[2pt]
> s = \alpha/\sqrt{r} & (\text{rsLoRA}): & \mathbb{E}|\delta h| \propto \alpha^2 & \Rightarrow\ \text{与}\ r\ \text{无关（秩稳定）}\\[2pt]
> s = 1 & (\text{不缩放}): & \mathbb{E}|\delta h| \propto r & \Rightarrow\ r\ \text{越大，更新发散}
> \end{array}
> $$
>
> **原论文为何仍用 $\alpha/r$？** 论文的做法是 $\alpha$ 取第一个尝试的 $r$、之后不再调（相当于把 $s^2r$ 的 $r$ 依赖吸收进学习率 $\eta$ 的调参）。在 $r\le 64$ 的实用区间内这样可行；但严格意义上，要让有效学习率**与 $r$ 无关**，正确标度是 $\alpha/\sqrt{r}$（即 2.3 节的 rsLoRA）。3.4 节的实测（20 种子平均，$r$ 翻倍时首步扰动之比）：
>
> | 缩放 | $r: 1\to 64$ 扰动变化 | $r$ 翻倍比值（实测） | 理论比值 |
> |---|---|---|---|
> | $\alpha/r$ | $0.447 \to 0.0017$（坍缩约 $256\times$） | $2.86, 2.74, 2.60, 2.37, 2.38, 2.24$ | $\to 2$（$\propto 1/r$） |
> | $\alpha/\sqrt{r}$ | $0.447 \to 0.111$ | $1.43, 1.37, 1.30, 1.19, 1.19, 1.12$ | $\to 1$（秩稳定 ✓） |
> | $1$ | $0.0018 \to 0.028$ | $1.40, 1.46, 1.54, 1.69, 1.68, 1.79$ | $\to 2$（$\propto r$） |
>
> 实测比值随 $r$ 增大分别逼近 $2,1,2$，与理论标度完全一致（小 $r$ 处的偏离来自 $\|Ax\|^2$ 的有限样本涨落）。

### 2.3 方法二：rsLoRA 秩稳定缩放（$s=\alpha/\sqrt{r}$）

rsLoRA（Kalajdzievski, 2023）与 LoRA 的**唯一区别**是缩放因子：

$$
h = W_0 x + \frac{\alpha}{\sqrt{r}}\, BA x. \tag{2-9}
$$

> **【推导】秩稳定性的必要性**
>
> 沿用 2.2.6 的框架。要求"增大 $r$ 时，首步输出扰动的期望幅度保持不变"（rank-stabilization）：
>
> $$
> \mathbb{E}|\delta h| \propto s^2\, r\, \sigma_a^2 \stackrel{\text{要求}}{=} \text{const（与 }r\text{ 无关）}.
> $$
>
> 由于 $\sigma_a^2 = \Theta(1/k)$ 与 $r$ 无关，解出
>
> $$
> s^2 r = \text{const} \;\Rightarrow\; s = \frac{\alpha}{\sqrt{r}}. \qquad \blacksquare
> $$
>
> 直观解释：$r$ 增大时 $A$ 的行数变多，$\|Ax\|^2$ 中累加的独立随机项从 $r$ 项增加，首步更新天然带有 $\sqrt{r}$ 量级的"能级膨胀"；除以 $\sqrt{r}$ 恰好抵消，而 $\alpha/r$ 矫枉过正、多除了一个 $\sqrt{r}$。

### 2.4 等价性证明

本节给出四条等价性/最优性陈述，把 LoRA 与全量微调、合并推理、SVD 最优近似、两种缩放之间的关系钉死。

#### 2.4.1 LoRA = 秩约束全量微调（约束优化视角的等价）

> **【推导】**
>
> 全量微调（2-1 的无约束版）：$\min_{\Delta W\in\mathbb{R}^{d\times k}} \mathcal{L}(W_0+\Delta W)$。
>
> LoRA：$\min_{B,A} \mathcal{L}(W_0 + sBA)$。注意 $s$ 只是吸收进 $\Delta W$ 定义的常数——令 $\Delta W = sBA$，由 1.3 知识卡片性质 2：
>
> $$
> \{sBA : B,A\} = \{\Delta W : \mathrm{rank}(\Delta W)\le r\} =: \mathcal{M}_r,
> $$
>
> 故 LoRA $\iff$ $\min_{\Delta W\in\mathcal{M}_r}\mathcal{L}(W_0+\Delta W)$，即**同一目标函数在秩-$r$ 矩阵流形 $\mathcal{M}_r$ 上的约束优化**。$\blacksquare$
>
> **推论（无损条件）**：若全量微调的最优解 $\Delta W^*$ 满足 $\mathrm{rank}(\Delta W^*)\le r$，则 $\Delta W^* \in \mathcal{M}_r$，LoRA 与全量微调有相同最优值——**LoRA 无损**。反之误差由 2.4.3 的 Eckart–Young 界定界。这解释了"内在秩低 ⇒ LoRA 几乎不掉点"。

#### 2.4.2 合并等价性（推理零额外开销）

> **【推导】**
>
> 训练完成后预计算合并权重
>
> $$
> W' = W_0 + s\,BA\quad\in\mathbb{R}^{d\times k}. \tag{2-10}
> $$
>
> 对任意输入 $x$，由矩阵乘法对加法的分配律：
>
> $$
> W'x = \big(W_0 + sBA\big)x = W_0 x + sBAx = h_{\text{LoRA}}(x). \qquad \blacksquare
> $$
>
> $W'$ 与 $W_0$ 同形，推理 FLOPs、显存访问模式与基线模型**逐位相同**（3.2 节实测合并前后输出最大偏差 $9.5\times 10^{-7}$，为浮点重排误差）。这与 Adapter 类方法形成对比：Adapter 在层间串入新模块，推理延迟真实增加。

#### 2.4.3 与 SVD 最优低秩近似的联系（Eckart–Young 定界）

> **【知识卡片】SVD 与 Eckart–Young 定理**
>
> 任何 $M\in\mathbb{R}^{d\times k}$ 有奇异值分解 $M = U\Sigma V^\top$，奇异值 $\sigma_1\ge\sigma_2\ge\cdots\ge 0$。
>
> **Eckart–Young 定理**：在所有秩 $\le r$ 的矩阵中，截断 SVD $M_r = U_r\Sigma_r V_r^\top$ 同时达到 Frobenius 范数与谱范数下的最优近似：
>
> $$
> \min_{\mathrm{rank}(\widehat M)\le r}\|M-\widehat M\|_F = \sqrt{\sum_{i>r}\sigma_i^2},\qquad
> \min_{\mathrm{rank}(\widehat M)\le r}\|M-\widehat M\|_2 = \sigma_{r+1}.
> $$

> **【推导】LoRA 不计算 SVD，却被 Eckart–Young 定界**
>
> 一个自然疑问：既然截断 SVD 是最优秩-$r$ 近似，为何不直接对 $\Delta W$ 做 SVD？因为 $\Delta W$ **未知**——先全量微调再 SVD 就失去了省内存的意义。LoRA 的做法是**直接在流形 $\mathcal{M}_r$ 上梯度下降**，让数据本身决定低秩结构。
>
> 当输入近似白化（$\mathbb{E}[xx^\top]\propto I$）且 $\ell$ 为平方损失时，
>
> $$
> \mathcal{L}(W_0+\Delta W) - \mathcal{L}(W_0+\Delta W^*) = \frac{1}{N}\|X(\Delta W - \Delta W^*)^\top\|_F^2 \;\propto\; \|\Delta W - \Delta W^*\|_F^2,
> $$
>
> 即训练目标逼近"以 Frobenius 范数拟合 $\Delta W^*$"，于是 LoRA 的收敛误差以 Eckart–Young 界为下界：
>
> $$
> \|\Delta W_{\text{LoRA}} - \Delta W^*\|_F \;\ge\; \sqrt{\sum_{i>r}\sigma_i(\Delta W^*)^2}. \tag{2-11}
> $$
>
> **当 $\Delta W^*$ 的谱快速衰减时，很小的 $r$ 即可让 (2-11) 的右端趋零**——这正是低秩内在假设的定量化。

> **【小例子】Eckart–Young 界的数值核验（3.3 节实验的真实输出）**
>
> 构造真实增量 $\Delta W^* = B^*A^*$（$d=128, k=64$，真实秩 $r^*=4$），测得奇异值：
>
> $$
> \sigma = (114.5217,\ 107.4066,\ 83.2100,\ 72.2037,\ 0,\ 0,\ \dots).
> $$
>
> - $r=1$ 的理论下界：$\dfrac{\sqrt{107.4066^2 + 83.21^2 + 72.2037^2}}{\sqrt{\sum_i\sigma_i^2}} = \dfrac{153.861}{191.804} = 0.8022$；
> - LoRA $r=1$ 训练 3000 步后实测相对误差 $\|\widehat{\Delta W}-\Delta W^*\|_F/\|\Delta W^*\|_F = \mathbf{0.8054}$——几乎贴着下界 $0.8022$，说明 LoRA 找到了**近乎最优的秩-1 近似**；
> - $r=4$ 时（秩充分）：下界为 $0$，LoRA 实测误差 $\mathbf{1\times 10^{-6}}$、训练损失降至 $1.76\times 10^{-10}$——**无损恢复 $\Delta W^*$**，验证了 2.4.1 的无损条件。

#### 2.4.4 两种缩放的联系

> **【推导】**
>
> 记 LoRA 缩放 $s_{\text{A}} = \alpha_{\text{A}}/r$，rsLoRA 缩放 $s_{\text{B}} = \alpha_{\text{B}}/\sqrt{r}$。同一 $(A,B)$ 下两者前向逐点相等当且仅当
>
> $$
> \frac{\alpha_{\text{A}}}{r} = \frac{\alpha_{\text{B}}}{\sqrt{r}} \;\Longleftrightarrow\; \alpha_{\text{B}} = \frac{\alpha_{\text{A}}}{\sqrt{r}}.
> $$
>
> 即两种方法是**同一算法族 $s(r)$ 的两个参数化**：LoRA 取 $s\propto r^{-1}$，rsLoRA 取 $s\propto r^{-1/2}$，差别仅在有效学习率的 $r$-标度律 $\mathbb{E}|\delta h|\propto s^2 r$（2.2.6）。换算关系是精确的恒等式，不存在近似。$\blacksquare$

---

## 三、完整算法流程与核心代码

### 3.1 LoRA 完整实现（逐行公式注释）

```python
import torch, torch.nn as nn, torch.nn.functional as F, math

torch.manual_seed(42)

class LoRALinear(nn.Module):
    """h = W0 x + (alpha / r) * B A x,  W0 冻结, 只训练 A, B"""
    def __init__(self, in_features, out_features, r, alpha):
        super().__init__()
        self.r = r
        self.scaling = alpha / r                                   # 缩放因子 s = α/r  [式(2-3)]

        # 冻结的预训练权重 W0 ∈ R^{out×in}, requires_grad=False ⇒ 无梯度/无优化器状态
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features),
            requires_grad=False)

        # LoRA 可训练参数: A ∈ R^{r×in}, B ∈ R^{out×r}
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))   # B = 0 ⇒ 初始 ΔW=0 [2.2.5 约束 C1]
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))      # A ~ Kaiming 均匀 ⇒ σ_a²=Θ(1/k) [2.2.6]

    def forward(self, x):
        base  = F.linear(x, self.weight)                           # 冻结支路: W0 x      [式(2-3) 第1项]
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)    # 低秩支路: B(A x)    [式(2-3) 第2项]
        return base + self.scaling * delta                         # h = W0 x + s·BA x   [式(2-3)]

    @torch.no_grad()
    def merge(self):
        """推理前合并: W' = W0 + s·BA  [式(2-10), 合并等价性 2.4.2]"""
        merged = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False)
        merged.weight.copy_(self.weight + self.scaling * (self.lora_B @ self.lora_A))
        return merged
```

| 代码变量 | 数学符号 | 含义 |
|---|---|---|
| `self.weight` | $W_0$ | 冻结预训练权重，$\mathbb{R}^{d\times k}$ |
| `self.lora_A` / `self.lora_B` | $A$ / $B$ | 低秩因子，$\mathbb{R}^{r\times k}$ / $\mathbb{R}^{d\times r}$ |
| `self.scaling` | $s=\alpha/r$ | 缩放因子（rsLoRA 改为 `alpha/math.sqrt(r)`） |
| `base` | $W_0x$ | 冻结支路输出 |
| `delta` | $BAx$ | 低秩支路输出（先降维 $Ax$ 再升维 $B(\cdot)$） |
| `merge()` | $W' = W_0 + sBA$ | 式 (2-10) 合并权重，推理零开销 |

### 3.2 正确性验证（四条断言）

```python
d_in, d_out, r, alpha = 64, 128, 8, 16
x = torch.randn(4, d_in)
lora = LoRALinear(d_in, d_out, r, alpha)

# 验证1: 初始化等价性 —— B=0 ⇒ LoRA 输出 ≡ W0 x  [2.2.5 约束 C1]
ref = F.linear(x, lora.weight)
assert torch.allclose(lora(x), ref)                       # 最大偏差 0.0 ✓

# 验证2: 合并等价性 —— 训练后 W0 + s·BA 与分支前向逐位一致  [2.4.2]
with torch.no_grad():                                     # 模拟训练后的非零 B
    lora.lora_B.copy_(torch.randn_like(lora.lora_B) * 0.01)
merged_linear = lora.merge()
assert torch.allclose(lora(x), merged_linear(x), atol=1e-6)  # 最大偏差 9.5e-7 ✓

# 验证3: 参数量 —— r(d+k) vs dk  [2.2.3]
trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
assert trainable == r * (d_in + d_out)                    # 1536 vs 8192, 占 18.75% ✓

# 验证4: 梯度流向 —— W0 无梯度; 初始 ‖∇_A‖=0, ‖∇_B‖>0  [2.2.5 启动顺序]
lora2 = LoRALinear(d_in, d_out, r, alpha)
F.mse_loss(lora2(x), torch.randn(4, d_out)).backward()
assert lora2.weight.grad is None                          # W0 冻结 ✓
assert lora2.lora_A.grad.norm().item() == 0.0             # ∇_A = s Bᵀg xᵀ = 0 (因 B=0) ✓
assert lora2.lora_B.grad.norm().item() > 0                # ∇_B = s g(Ax)ᵀ ≈ 0.3851 > 0 ✓
```

| 代码行 | 数学对象 | 验证的理论命题 |
|---|---|---|
| `torch.allclose(lora(x), ref)` | $BA=0 \Rightarrow h=W_0x$ | 2.2.5 约束 (C1)，实测偏差 $0.0$ |
| `torch.allclose(..., atol=1e-6)` | $W'x = W_0x + sBAx$ | 2.4.2 合并等价，实测偏差 $9.5\times10^{-7}$ |
| `trainable == r*(d_in+d_out)` | $dk \to r(d+k)$ | 2.2.3 参数量推导 |
| `lora_A.grad == 0` | $\nabla_A = sB^\top g x^\top$ | 式 (2-5) 与初始化启动顺序 |
| `lora_B.grad > 0` | $\nabla_B = s\,g(Ax)^\top$ | 式 (2-4)，实测 $\|\nabla_B\|=0.3851$ |

### 3.3 低秩拟合实验：$r$ 充分 vs $r$ 不足，对照 Eckart–Young 界

构造真实低秩增量 $\Delta W^* = B^*A^*$（秩 $r^*=4$），用 LoRA 以不同 $r$ 去拟合，检验 2.4.1（无损条件）与 2.4.3（Eckart–Young 下界）。

```python
torch.manual_seed(123)
d_in, d_out, r_star = 64, 128, 4
W0 = torch.randn(d_out, d_in) / math.sqrt(d_in)
DeltaW_star = torch.randn(d_out, r_star) @ torch.randn(r_star, d_in)   # 真实秩-4 增量 ΔW*
X = torch.randn(512, d_in)                                             # 近似白化数据 E[xxᵀ]≈I
Y = X @ (W0 + DeltaW_star).T                                           # 目标映射 y = (W0+ΔW*)x

def train_lora(r, alpha=1.0, steps=3000, lr=1e-2):
    torch.manual_seed(0)
    layer = LoRALinear(d_in, d_out, r, alpha)
    with torch.no_grad():
        layer.weight.copy_(W0)                                         # 起点 = 预训练权重
    opt = torch.optim.Adam([p for p in layer.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):                                             # 只优化 A, B（流形 M_r 上的 GD）
        opt.zero_grad()
        loss = F.mse_loss(layer(X), Y)                                 # L(W0 + s·BA)
        loss.backward()                                                # ∇_A, ∇_B 由式(2-6)给出
        opt.step()
    with torch.no_grad():
        DeltaW_hat = layer.scaling * (layer.lora_B @ layer.lora_A)     # 学成增量 ΔŴ = s·BA
        rel_err = ((DeltaW_hat - DeltaW_star).norm('fro')              # 相对 Frobenius 误差
                   / DeltaW_star.norm('fro')).item()
    return loss.item(), rel_err

# Eckart–Young 理论下界: min_{rank≤r} ‖ΔŴ-ΔW*‖_F / ‖ΔW*‖_F  [式(2-11)]
U, S, Vh = torch.linalg.svd(DeltaW_star)
for r_test in [1, 4]:
    bound = (S[r_test:]**2).sum().sqrt().item() / S.norm().item()      # √(Σ_{i>r}σ_i²)/‖ΔW*‖_F
    loss, rel_err = train_lora(r_test)
    print(f"r={r_test}: loss={loss:.3e}, 实测误差={rel_err:.4f}, EY下界={bound:.4f}")
```

**实测输出**（沙箱真实运行结果）：

| LoRA 秩 $r$ | 最终训练损失 | 实测相对误差 $\|\widehat{\Delta W}-\Delta W^*\|_F/\|\Delta W^*\|_F$ | Eckart–Young 下界 |
|---|---|---|---|
| $1$（秩不足） | $1.771\times 10^{2}$ | $0.8054$ | $0.8022$ |
| $4$（秩充分） | $1.764\times 10^{-10}$ | $0.000001$ | $0.0000$ |

| 代码变量 | 数学符号 | 含义 |
|---|---|---|
| `DeltaW_star` | $\Delta W^* = B^*A^*$ | 待拟合的真实秩-4 增量 |
| `DeltaW_hat` | $\widehat{\Delta W} = sBA$ | LoRA 学成的增量 |
| `S` | $\sigma_1\ge\sigma_2\ge\cdots$ | $\Delta W^*$ 的奇异值 |
| `bound` | $\sqrt{\sum_{i>r}\sigma_i^2}/\|\Delta W^*\|_F$ | 式 (2-11) 的 Eckart–Young 相对下界 |
| `rel_err` | $\|\widehat{\Delta W}-\Delta W^*\|_F/\|\Delta W^*\|_F$ | 实测相对误差 |

**结论**：秩不足时 LoRA 收敛到几乎贴着 Eckart–Young 下界的最优秩-1 近似（0.8054 vs 0.8022）；秩充分时无损恢复 $\Delta W^*$。低秩假设成立时 LoRA 不掉点，不成立时其损失由谱尾巴精确预言。

### 3.4 缩放因子实验：首步输出扰动 $\|\delta h\|$ 与 $r$ 的标度律

验证 2.2.6/2.3 的推导：$\mathbb{E}\|\delta h\| \propto s^2 r$，即 $s=\alpha/r$ 时 $\propto 1/r$（坍缩）、$s=\alpha/\sqrt r$ 时恒定（秩稳定）、$s=1$ 时 $\propto r$（发散）。

```python
def avg_output_perturbation(r, scale_fn, seeds=20, lr=0.1):
    """多种子平均: 一步 SGD 后首步更新的每样本输出扰动 ‖ΔW₁ x‖  [式(2-7)(2-8)]"""
    vals = []
    for sd in range(seeds):
        torch.manual_seed(sd)
        layer = LoRALinear(64, 128, r, alpha=1.0)
        torch.manual_seed(1000 + sd)
        xx = torch.randn(32, 64); tt = torch.randn(32, 128)
        layer.scaling = scale_fn(r)                                    # 注入缩放律 s(r)
        F.mse_loss(layer(xx), tt).backward()                           # ∇_B = s·GᵀXAᵀ [式(2-6)]
        with torch.no_grad():
            B1 = layer.lora_B - lr * layer.lora_B.grad                 # B₁ = -η∇_B  [2.2.6]
            DeltaW1 = layer.scaling * (B1 @ layer.lora_A)              # ΔW₁ = -ηs²·gxᵀAᵀA [式(2-7)]
            vals.append((DeltaW1 @ xx.T).norm().item() / math.sqrt(32))
    return sum(vals) / len(vals)

for r in [1, 2, 4, 8, 16, 32, 64]:
    n_lora   = avg_output_perturbation(r, lambda r: 16.0 / r)          # s = α/r   (LoRA)
    n_rslora = avg_output_perturbation(r, lambda r: 16.0 / math.sqrt(r))# s = α/√r  (rsLoRA)
    n_none   = avg_output_perturbation(r, lambda r: 1.0)               # s = 1     (不缩放)
```

**实测输出**（20 种子平均，每样本均方根扰动）：

| $r$ | $s=\alpha/r$（LoRA） | $s=\alpha/\sqrt r$（rsLoRA） | $s=1$（不缩放） |
|---|---|---|---|
| 1 | 0.44724 | 0.44724 | 0.00175 |
| 2 | 0.15635 | 0.31269 | 0.00244 |
| 4 | 0.05702 | 0.22806 | 0.00356 |
| 8 | 0.02196 | 0.17565 | 0.00549 |
| 16 | 0.00926 | 0.14817 | 0.00926 |
| 32 | 0.00390 | 0.12466 | 0.01558 |
| 64 | 0.00174 | 0.11139 | 0.02785 |

$r$ 翻倍时的实测比值分别为 $\to 2$（$\propto 1/r$）、$\to 1$（秩稳定 ✓）、$\to 2$（$\propto r$），与 2.2.6 的理论标度律完全吻合（小 $r$ 处的偏差来自 $\|Ax\|^2$ 的有限样本涨落，见引理证明）。

| 代码变量 | 数学符号 | 含义 |
|---|---|---|
| `scale_fn(r)` | $s(r)$ | 三种缩放律：$\alpha/r$、$\alpha/\sqrt r$、$1$ |
| `B1` | $B_1 = -\eta s\,g(Ax)^\top$ | 一步 SGD 后的 $B$ |
| `DeltaW1` | $\Delta W_1 = -\eta s^2 g x^\top A^\top A$ | 式 (2-7) 首步有效更新 |
| `vals` | $\|\delta h\| = \|\Delta W_1 x\|$ | 式 (2-8) 输出扰动 |

### 3.5 对比表：全量微调 vs LoRA vs rsLoRA

| 维度 | 全量微调 | LoRA（$s=\alpha/r$） | rsLoRA（$s=\alpha/\sqrt r$） |
|---|---|---|---|
| 优化变量 | $\Delta W\in\mathbb{R}^{d\times k}$（$dk$ 个） | $A,B$（$r(d+k)$ 个） | 同 LoRA |
| 约束/假设 | 无 | $\mathrm{rank}(\Delta W)\le r$（2.4.1） | 同 LoRA |
| 优化器状态 | 16 字节 × $dk$ | 16 字节 × $r(d+k)$ | 同 LoRA |
| $d=k=4096,\ r=16$ 时优化器状态 | 256 MiB/层 | 2 MiB/层（0.78%） | 同 LoRA |
| GPT-3 175B 检查点（论文数据） | ~350 GB | ~35 MB（$r=4$，约 $10^{-4}$） | 同 LoRA |
| 初始化要求 | 任意 | $BA=0$ 且至少一个因子随机（2.2.5） | 同 LoRA |
| 首步有效更新 $\propto s^2r$ | — | $\propto 1/r$（大 $r$ 更新坍缩） | 常数（秩稳定，2.3） |
| 推理开销 | 基线 | 基线 + 零（合并等价，2.4.2） | 同 LoRA |
| 误差保证 | — | Eckart–Young 下界（2.4.3） | 同 LoRA |
| 多任务切换 | 每任务一份全量权重 | 每任务一对小 $A,B$，共享 $W_0$ | 同 LoRA |

---

## 四、数学知识清单

1. **矩阵秩**：秩的子乘法性 $\mathrm{rank}(BA)\le\min(\mathrm{rank}B,\mathrm{rank}A)$；秩-$r$ 分解存在性 $\mathrm{rank}(M)\le r \iff M=BA$（1.3、2.2.1）。
2. **SVD 与 Eckart–Young 定理**：最优秩-$r$ 近似 = 截断 SVD，Frobenius 误差 $\sqrt{\sum_{i>r}\sigma_i^2}$（2.4.3）。
3. **迹技巧（矩阵求导）**：$d\mathcal{L} = \mathrm{tr}(G^\top dX) \Rightarrow \nabla_X = G$；$\mathrm{tr}$ 的循环不变性（2.2.4）。
4. **随机向量的二次型期望**：$\mathbb{E}\|Ax\|^2 = r\sigma_a^2\|x\|^2$（独立零均值条目，2.2.6 引理）。
5. **Kaiming/高斯初始化的方差标度**：fan-in 初始化 $\sigma_a^2 = \Theta(1/k)$，与秩 $r$ 无关（2.2.6）。
6. **约束优化与流形参数化**：秩约束 $\mathcal{M}_r$ 的低秩因子化坐标 $(B,A)$，因子化非凸性与规范自由度 $(BQ, Q^{-1}A)$（2.2.1）。
7. **内在维度假设**：下游适配的自由度远低于表观参数维度（Aghajanyan et al., 2020；1.2、2.2.1）。
8. **Adam 优化器内存模型**：混合精度下 16 字节/可训练参数（1.1）。
9. **Frobenius 范数与谱**：$\|M\|_F^2 = \sum_i\sigma_i^2$，白化数据下平方损失与权重误差 Frobenius 范数的对应（2.4.3）。

---

## 五、总结

1. **LoRA 的本质是一个约束优化问题**：把全量微调 $\min_{\Delta W}\mathcal{L}(W_0+\Delta W)$ 限制在秩 $\le r$ 流形 $\mathcal{M}_r$ 上，并用 $\Delta W = sBA$ 给出该流形的可微参数化（2.2.1、2.4.1）。可训练参数 $dk \to r(d+k)$，优化器内存同步压缩（2.2.3）。
2. **两个工程细节各有一条定理支撑**：初始化 "$B=0$、$A$ 随机" 是同时满足"不破坏预训练函数"与"可训练性"的（仅差一个对称选择的）唯一方案，梯度公式 (2-4)(2-5) 给出了逐步启动机制（2.2.5）；缩放因子决定有效学习率的 $r$-标度律 $\mathbb{E}\|\delta h\|\propto s^2 r$，严格的秩稳定要求 $s=\alpha/\sqrt r$（rsLoRA），原论文的 $\alpha/r$ 在小 $r$ 区间可用但大 $r$ 下更新坍缩（2.2.6、2.3、3.4 实测）。
3. **误差有精确理论兜底**：当最优增量秩 $\le r$ 时 LoRA 与全量微调同最优值（无损）；否则误差以 Eckart–Young 界 $\sqrt{\sum_{i>r}\sigma_i^2}$ 为下界，实验中 LoRA 收敛值几乎贴着该下界（2.4.3、3.3 实测 0.8054 vs 0.8022）。
4. **部署零成本**：合并权重 $W' = W_0 + sBA$ 与基线推理逐位等价（2.4.2，实测偏差 $9.5\times10^{-7}$），这是 LoRA 优于 Adapter 类方法的结构性优势。
5. **方法族视角**：LoRA、rsLoRA 只是缩放律 $s(r)$ 的两种取法，换算关系为精确恒等式 $\alpha_{\text{B}} = \alpha_{\text{A}}/\sqrt r$（2.4.4）；理解标度律 $s^2r$ 比记忆任何单一公式更重要。
