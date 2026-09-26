---
layout: post.njk
post_id: 2026-09-26-qwen-3-5-架构解析-1-从-softmax-attention-linear-attention-deltanet-gated-deltanet-演化
archive: 大模型架构
title: Qwen 3.5 架构解析 (1)：整体分析 Gated DeltaNet 与 Sparse MoE
date: 2026-09-26
updated: 2026-09-26
tags:
  - post
---
## 1. 总体介绍 Qwen 

### 1.1 先看两张图

**官方 Qwen3-Next 架构图**展示了三条主线：**Gated DeltaNet** 作为线性 token mixer、**Gated Softmax Attention** 作为精确的全注意力 token mixer，以及每个 block 后面的 **Mixture of Experts**。图中右下角的 Gated Delta Rule 负责跨时间的递归状态更新，右上角的 Gated Softmax Attention 负责显式的 token-to-token 交互。

![官方架构图](img/qwen3_next_architecture.png)

下面进行详细的推导。


---

## 第一部分：Linear Attention 公式 (5) 的推导

### 1.1 起点：核化后的因果注意力

Linear Attention 论文（Katharopoulos et al., 2020）把标准 softmax 替换成核函数 $\kappa(q,k)=\phi(q)^\top\phi(k)$。为了书写简洁，论文用 $\tilde{q}_t:=\phi(q_t)$、$\tilde{k}_i:=\phi(k_i)$ 表示特征映射后的向量，得到论文中的公式 (3)：

$$o_t=\frac{\displaystyle\sum_{i=1}^{t}(\tilde{q}_t^\top\tilde{k}_i)\,v_i}{\displaystyle\sum_{i=1}^{t}\tilde{q}_t^\top\tilde{k}_i}. \tag{3}$$

注意分子里的 $(\tilde{q}_t^\top\tilde{k}_i)$ 是一个**标量**（1×1），$v_i$ 是一个列向量。

### 1.2 分子：结合律如何产生矩阵 $W_t$

我们先看分子，目标是把 $\tilde{q}_t$ 从求和号里**提取**出来。

**第一步**：标量乘法可交换顺序
$$(\tilde{q}_t^\top\tilde{k}_i)\,v_i=v_i\,(\tilde{q}_t^\top\tilde{k}_i).$$

**第二步**：标量等于自身的转置
$$\tilde{q}_t^\top\tilde{k}_i=(\tilde{q}_t^\top\tilde{k}_i)^\top=\tilde{k}_i^\top\tilde{q}_t.$$

于是分子变成：
$$\sum_{i=1}^{t}v_i\,(\tilde{k}_i^\top\tilde{q}_t).$$

**第三步**：利用矩阵乘法结合律
$v_i$ 是 $d_v\times 1$ 的列向量，$\tilde{k}_i^\top\tilde{q}_t$ 是标量。一个列向量乘标量，等于先把列向量和标量里的"行向量部分"结合：

$$v_i\,(\tilde{k}_i^\top\tilde{q}_t)=\underbrace{(v_i\tilde{k}_i^\top)}_{d_v\times d_\phi}\,\tilde{q}_t.$$

这里 $v_i\tilde{k}_i^\top$ 是**外积**（列×行=矩阵），维度为 $d_v\times d_\phi$。

**第四步**：求和号只作用于与 $i$ 有关的部分
$$\sum_{i=1}^{t}(v_i\tilde{k}_i^\top)\tilde{q}_t=\left(\sum_{i=1}^{t}v_i\tilde{k}_i^\top\right)\tilde{q}_t.$$

论文定义：
$$W_t:=\sum_{i=1}^{t}v_i\tilde{k}_i^\top\quad\in\mathbb{R}^{d_v\times d_\phi}. \tag{4}$$

于是分子严格等于：
$$\boxed{W_t\tilde{q}_t}.$$

### 1.3 分母：结合律产生向量 $z_t$

分母更简单，全是标量求和：
$$\sum_{i=1}^{t}\tilde{q}_t^\top\tilde{k}_i=\tilde{q}_t^\top\left(\sum_{i=1}^{t}\tilde{k}_i\right).$$

论文定义：
$$z_t:=\sum_{i=1}^{t}\tilde{k}_i\quad\in\mathbb{R}^{d_\phi}.$$

于是分母等于：
$$\tilde{q}_t^\top z_t=z_t^\top\tilde{q}_t$$
（最后一步因为标量转置不变）。

### 1.4 公式 (5) 的诞生

把分子分母合起来，即得论文公式 (5)：

$$\boxed{o_t=\frac{W_t\tilde{q}_t}{z_t^\top\tilde{q}_t}}. \tag{5}$$

**关键认知**：这一步没有任何近似，纯粹是**结合律**的代数变形。它把"每个位置都要和历史上所有位置两两配对"的 $O(N^2)$ 计算，变成了"先累加一个公共矩阵 $W_t$，再乘以当前查询"的 $O(N)$ 计算。

> **参考文献**：Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, ICML 2020.  
> [arXiv](https://arxiv.org/abs/2006.16236) / [PMLR](https://proceedings.mlr.press/v119/katharopoulos20a.html)

---

## 第二部分：Delta Rule 的梯度推导（指标法）

你贴出的图片里要求梯度 $\nabla_S\mathcal{L}$。很多同学（包括我之前笔记里的写法）容易在这里搞混**矩阵的左右乘**。下面用**指标法**——把矩阵拆开写成带下标的元素——一步步算，这是推导矩阵梯度最不容易出错的方法。

### 2.1 统一符号：采用 Fast Weight 视角

为了和 DeltaNet 原始论文（Schlag et al., 2021）一致，我们引入 **fast weight matrix** $W\in\mathbb{R}^{d\times d}$。它的作用是把输入 key 直接映射到 value：

$$\hat{v}=Wk.$$

（如果你更熟悉 Linear Attention 里的 $S$，两者的关系很简单：$W=S^\top$，只是转置关系。）

在第 $t$ 步，模型看到输入 $k_t\in\mathbb{R}^d$ 和目标输出 $v_t\in\mathbb{R}^d$，定义平方损失：

$$\mathcal{L}(W)=\frac{1}{2}\|\hat{v}_t-v_t\|^2=\frac{1}{2}(Wk_t-v_t)^\top(Wk_t-v_t). \tag{2}$$

### 2.2 把损失写成"带下标"的元素形式

设 $W$ 的第 $a$ 行第 $b$ 列元素为 $W_{ab}$，向量 $k_t$ 的第 $b$ 个元素为 $(k_t)_b$，$v_t$ 的第 $a$ 个元素为 $(v_t)_a$。

预测输出的第 $a$ 个分量：
$$(\hat{v}_t)_a=\sum_{b=1}^{d}W_{ab}(k_t)_b.$$

损失函数展开：
$$\mathcal{L}=\frac{1}{2}\sum_{a=1}^{d}\left[\underbrace{\left(\sum_{b=1}^{d}W_{ab}(k_t)_b\right)}_{(\hat{v}_t)_a}-(v_t)_a\right]^2.$$

### 2.3 对 $W_{mn}$ 求偏导数

我们要算的是 $\frac{\partial\mathcal{L}}{\partial W_{mn}}$（第 $m$ 行第 $n$ 列元素的偏导）。

根据链式法则：
$$\frac{\partial\mathcal{L}}{\partial W_{mn}}=\sum_{a=1}^{d}\left[(\hat{v}_t)_a-(v_t)_a\right]\cdot\frac{\partial(\hat{v}_t)_a}{\partial W_{mn}}.$$

先看内层偏导 $\frac{\partial(\hat{v}_t)_a}{\partial W_{mn}}$：
$$(\hat{v}_t)_a=\sum_{b}W_{ab}(k_t)_b.$$

对 $W_{mn}$ 求导时，只有当 $a=m$ 且 $b=n$ 时该项才不为零，因此：
$$\frac{\partial(\hat{v}_t)_a}{\partial W_{mn}}=\begin{cases}(k_t)_n,&a=m,\\0,&a\neq m.\end{cases}$$

代回链式法则，求和号里只剩 $a=m$ 这一项：
$$\frac{\partial\mathcal{L}}{\partial W_{mn}}=\left[(\hat{v}_t)_m-(v_t)_m\right]\cdot(k_t)_n=(Wk_t-v_t)_m\cdot(k_t)_n.$$

### 2.4 从元素回到矩阵

我们发现，梯度矩阵的第 $(m,n)$ 个元素等于：
$$G_{mn}=(Wk_t-v_t)_m\cdot(k_t)_n.$$

这正是**外积** $(Wk_t-v_t)\,k_t^\top$ 的第 $(m,n)$ 个元素。因为：
- $(Wk_t-v_t)$ 是 $d\times 1$ 列向量，第 $m$ 个元素是 $(Wk_t-v_t)_m$；
- $k_t^\top$ 是 $1\times d$ 行向量，第 $n$ 个元素是 $(k_t)_n$；
- 两者相乘得到 $d\times d$ 矩阵，第 $(m,n)$ 个元素正好是上面的乘积。

因此：
$$\boxed{\nabla_W\mathcal{L}=(Wk_t-v_t)\,k_t^\top}. \tag{梯度公式}$$

**维度检查**：$W$ 是 $d\times d$，$(Wk_t-v_t)$ 是 $d\times 1$，$k_t^\top$ 是 $1\times d$，外积结果恰好是 $d\times d$，与 $W$ 同维度。**维度一致，说明公式正确。**

### 2.5 SGD 更新即得 Delta Rule

梯度下降 $W_t=W_{t-1}-\beta_t\nabla_W\mathcal{L}$，代入：

$$W_t=W_{t-1}-\beta_t(W_{t-1}k_t-v_t)k_t^\top.$$

把括号拆开：
$$W_t=W_{t-1}-\beta_t W_{t-1}k_t k_t^\top+\beta_t v_t k_t^\top.$$

提取公因式 $W_{t-1}$：
$$\boxed{W_t=W_{t-1}\big(I-\beta_t k_t k_t^\top\big)+\beta_t v_t k_t^\top}. \tag{DeltaNet}$$

这就是 DeltaNet 核心更新方程的完整来源：**它等价于对线性预测损失做一步随机梯度下降。**

> **参考文献**：Schlag et al., *Linear Transformers are Secretly Fast Weight Programmers*, NeurIPS 2021.  
> [arXiv](https://arxiv.org/abs/2102.11174)

---

## 第三部分：Gated DeltaNet 状态方程的构造

Gated DeltaNet（Yang et al., 2024）的方程不是"凭空设计"的，而是遵循一个非常清晰的原则：**先全局遗忘，再局部纠错**。下面展示它的两步构造法。

### 3.1 第一步：输入依赖的全局遗忘

引入门控向量 $\alpha_t\in(0,1)^d$，它由当前输入经一个小型投影网络 + Sigmoid 生成。定义对角衰减矩阵：

$$\Lambda_t:=\operatorname{Diag}(\alpha_t)\in\mathbb{R}^{d\times d}.$$

对历史状态做逐行（逐通道）的指数式衰减：
$$W'_{t-1}=\Lambda_t\,W_{t-1}=\operatorname{Diag}(\alpha_t)\,W_{t-1}.$$

**语义**：如果某个 $\alpha_t$ 的分量接近 0，则状态矩阵对应行的信息被大幅冲刷；接近 1 则基本保留。这个遗忘速率是数据驱动的。

### 3.2 第二步：基于衰减状态做 Delta Rule

遗忘后，我们用**已经衰减过的旧状态** $W'_{t-1}$ 来做检索和纠错：

1. **检索**：$\hat{v}_t=W'_{t-1}k_t=\Lambda_t W_{t-1}k_t$；
2. **误差**：$e_t=v_t-\hat{v}_t=v_t-\Lambda_t W_{t-1}k_t$；
3. **修正**：按 delta rule 写入误差。

更新方程：
$$W_t=W'_{t-1}+\beta_t\,e_t k_t^\top.$$

### 3.3 完整方程

把两步合起来，得到 Gated DeltaNet 的核心递推：

$$\boxed{W_t=\operatorname{Diag}(\alpha_t)\,W_{t-1}+\beta_t\left(v_t-\operatorname{Diag}(\alpha_t)\,W_{t-1}k_t\right)k_t^\top}. \tag{Gated DeltaNet}$$

**各项的严格语义**：

| 项 | 数学作用 |
|---|---|
| $\operatorname{Diag}(\alpha_t)W_{t-1}$ | **全局遗忘**：历史状态按输入依赖的速率逐行衰减 |
| $\operatorname{Diag}(\alpha_t)W_{t-1}k_t$ | **衰减后检索**：基于"遗忘后的记忆"做预测，保证时序因果性 |
| $\beta_t(v_t-\dots)k_t^\top$ | **定向纠错**：只沿 $k_t$ 方向修正误差，不扰动其他记忆 |

### 3.4 与 Linear Attention / DeltaNet 的关系

- 令 $\alpha_t=\mathbf{1}$（无遗忘）：退化为 **DeltaNet**；
- 令 $\alpha_t=\mathbf{1}$ 且 $\beta_t\to 0$：退化为纯累加的 **Linear Attention**；
- 令 $\beta_t\to 0$（无纠错）：退化为只有门控衰减的 **Mamba2/GLA** 风格。

> **参考文献**：Yang et al., *Gated Delta Networks: Improving Mamba2 with Delta Rule*, ICLR 2025.  
> [arXiv](https://arxiv.org/abs/2412.06464) / [OpenReview](https://openreview.net/forum?id=r8H7xhYPwz) / [Code](https://github.com/NVlabs/GatedDeltaNet)

---

## 第四部分：Sparse MoE 的详细推导

Sparse Mixture-of-Experts（Sparse MoE）与上面三条线解决的是不同问题：它不改造 Attention/SSM，而是把**前馈网络（FFN）**拆成多个"专家"，让每个 token 只激活极少数专家，从而在不增加推理计算的前提下扩大模型容量。

### 4.1 问题设定与路由机制

设我们有 $N$ 个专家网络 $E_1,\dots,E_N$，每个都是标准 FFN（通常比原始 dense FFN 小很多）。给定输入 $x\in\mathbb{R}^d$：

**Step 1：计算路由分数**
通过一个可学习的线性门控（router）：
$$h=W_g\,x\quad\in\mathbb{R}^N,$$
其中 $W_g\in\mathbb{R}^{N\times d}$。

**Step 2：Softmax 归一化**
$$g(x)=\operatorname{Softmax}(h)\in\mathbb{R}^N,\quad g(x)_i=\frac{e^{h_i}}{\sum_{j=1}^N e^{h_j}}.$$

**Step 3：Top-k 稀疏选择**
只保留分数最大的 $k$ 个专家（通常 $k=1$ 或 $2$）：
$$\mathcal{T}=\operatorname{TopK}(\{g(x)_i\}_{i=1}^N,\,k).$$

对选中的专家重新归一化权重：
$$\tilde{g}(x)_i=\frac{g(x)_i}{\sum_{j\in\mathcal{T}}g(x)_j},\quad i\in\mathcal{T}.$$

**Step 4：加权聚合输出**
$$y=\sum_{i\in\mathcal{T}}\tilde{g}(x)_i\cdot E_i(x). \tag{MoE输出}$$

### 4.2 Switch Transformer：$k=1$ 的极简形态

Switch Transformer（Fedus et al., 2021）的极端设定是 $k=1$：每个 token 只送给**一个**专家。此时：

$$y=E_{i^*}(x),\quad i^*=\underset{i}{\arg\max}\,g(x)_i.$$

但 $\arg\max$ 不可导，训练时仍用 softmax 采样。Switch Transformer 证明了：即使只用一个专家，只要把专家数 $N$ 拉大、配合负载均衡损失，效果依然很好，且推理极快。

### 4.3 负载均衡辅助损失（Auxiliary Loss）

如果不加约束，路由器会"偷懒"——总是把 token 送给少数几个学得好的专家，其他专家永远不被训练（**专家坍塌**，expert collapse）。

**定义**（在一个 batch 内统计）：
- $f_i=\frac{\text{路由给专家 }i\text{ 的 token 数}}{\text{batch 总 token 数}}$，这是**实际频率**；
- $P_i=\frac{1}{B}\sum_{x\in\text{batch}}g(x)_i$，这是路由器对专家 $i$ 的**平均分配概率**。

Switch Transformer 引入辅助损失：
$$\boxed{L_{\mathrm{aux}}=\alpha\cdot N\cdot\sum_{i=1}^{N}f_i\cdot P_i}. \tag{负载均衡损失}$$

**直观解释**：
- 如果专家 $i$ 接收了很多 token（$f_i$ 大），同时路由器平均也给它高概率（$P_i$ 大），则 $f_i P_i$ 乘积大，损失惩罚高；
- 最小化这个损失会迫使 $f_i$ 和 $P_i$ 趋向均匀分布 $1/N$；
- $\alpha$ 是超参数（通常取 0.01 左右），保证主任务损失不受干扰。

**总训练损失**：
$$L_{\text{total}}=L_{\text{LM}}+\alpha\cdot N\sum_{i=1}^{N}f_i P_i.$$

### 4.4 容量因子（Capacity Factor）

为防止某个专家在单步内被塞入过多 token，Switch Transformer 引入**容量因子** $C$：每个专家每轮最多处理 $C\cdot\frac{B}{N}$ 个 token（$B$ 为 batch 中 token 总数）。超出的 token 被标记为"溢出"（overflow），直接跳过该层或通过残差连接传递。

### 4.5 DeepSeekMoE 的改进

DeepSeekMoE（Dai et al., 2024）在 Switch Transformer 基础上做了两个关键改进：

**（1）共享专家（Shared Experts）**
设置 $N_s$ 个"共享专家"（如 2 个），**所有 token 都会经过它们**。这保证了通用语义表示始终被维护，避免路由决策失误时完全丢失基础能力。

**（2）细粒度专家拆分（Fine-grained Expert Segmentation）**
把传统 MoE 中 $N$ 个"大专家"拆成 $mN$ 个"小专家"，每次选 $mk$ 个。例如原来 16 个专家选 2 个，现在拆成 64 个专家选 8 个。这提高了专家组合的组合多样性。

DeepSeekMoE 的输出形式：
$$y=\underbrace{\sum_{s=1}^{N_s}E_s^{\text{shared}}(x)}_{\text{共享专家必过}}+\sum_{i\in\mathcal{T}_{\text{ routed}}}\tilde{g}(x)_i\,E_i^{\text{routed}}(x).$$

> **参考文献**：
> - Fedus et al., *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, 2021. [arXiv](https://arxiv.org/abs/2101.03961)
> - Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, 2024. [arXiv](https://arxiv.org/abs/2401.06066)

---

## 第五部分：四者的关系总结

| 模块 | 解决什么问题 | 核心数学操作 | 与他人的关系 |
|---|---|---|---|
| **Linear Attention** | 注意力 $O(N^2)\to O(N)$ | 核函数 + 结合律，产生矩阵状态 $W_t=\sum v_i\tilde{k}_i^\top$ | 共同骨架 |
| **DeltaNet** | 状态只增不减、无法纠错 | 对 $\frac12\|Wk-v\|^2$ 做 SGD，得到 delta rule | 在 Linear Attention 骨架上换更新核 |
| **Gated DeltaNet** | 没有时序遗忘机制 | 先对角衰减 $\operatorname{Diag}(\alpha_t)$，再 delta rule | 把 Mamba2 的门控和 DeltaNet 的纠错缝合 |
| **Sparse MoE** | FFN 参数量太大 | Top-k 路由 + 负载均衡损失 $L_{\text{aux}}$ | 与前三者**正交**，可叠加使用 |

**一句话**：Linear Attention 用结合律造出了"矩阵记忆体"；DeltaNet 用在线梯度下降教会了记忆体"定向改写"；Gated DeltaNet 又给记忆体装上"可学习的遗忘闸门"；而 Sparse MoE 完全不碰序列建模，只把 FFN 拆成按需调用的专家阵列——这四者可以**叠加**在同一架构里，分别负责"记得快"、"记得准"、"忘得巧"、"算得省"。

---

