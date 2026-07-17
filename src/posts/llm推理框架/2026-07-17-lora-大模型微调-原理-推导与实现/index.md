---
layout: post.njk
post_id: 2026-07-17-lora-大模型微调-原理-推导与实现
archive: llm推理框架
title: LoRA 大模型微调：原理、推导与实现
date: 2026-07-17
tags:
  - post
---
# LoRA 大模型微调：原理、推导与实现

## 一、问题、目标与符号

### 1.1 **LoRA** 解决的问题

对大语言模型进行全量微调时，需要更新每个权重矩阵，显存占用、优化器状态和模型存储成本都很高。**LoRA（Low-Rank Adaptation，低秩适配）** 冻结预训练权重，仅训练一个低秩增量，从而以较少参数实现任务适配。（来源：Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022）

### 1.2 符号与维度

| 符号 | 维度 | 含义 |
|---|---:|---|
| **$x$** | $\mathbb{R}^{d_{\text{in}}}$ | 输入向量 |
| **$W_0$** | $\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ | 冻结的预训练权重 |
| **$\Delta W$** | $\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ | 微调产生的权重增量 |
| **$A$** | $\mathbb{R}^{r\times d_{\text{in}}}$ | LoRA 下投影矩阵 |
| **$B$** | $\mathbb{R}^{d_{\text{out}}\times r}$ | LoRA 上投影矩阵 |
| **$r$** | 标量 | **秩（rank）**，通常远小于原始维度 |
| **$\alpha$** | 标量 | LoRA 缩放系数 |

### 1.3 数学范围

以下推导只使用矩阵乘法、矩阵秩、链式法则与梯度下降，均在本科计算机专业常见数学范围内。

## 二、基础表达：从全量微调到 LoRA

### 2.1 全量微调

线性层的原始输出为：

$$
h=W_0x
$$

全量微调会直接学习完整的增量矩阵 $\Delta W$：

$$
h=(W_0+\Delta W)x
$$

其中，$\Delta W$ 与 $W_0$ 同形状，需要训练参数数量为：

$$
d_{\text{out}}d_{\text{in}}
$$

### 2.2 LoRA 的低秩参数化

LoRA 不直接学习 $\Delta W$，而是令：

$$
\Delta W=\frac{\alpha}{r}BA
$$

因此输出变为：

$$
h=W_0x+\frac{\alpha}{r}BAx
$$

> **【推导】**
>
> $$
> (W_0+\Delta W)x
> =W_0x+\Delta Wx
> $$
>
> $$
> =W_0x+\frac{\alpha}{r}BAx
> $$
>
> 使用规则：矩阵乘法对加法满足分配律。

这表示 LoRA 先用 $A$ 把输入压缩到 $r$ 维，再用 $B$ 映射回原输出空间：

$$
x\in\mathbb{R}^{d_{\text{in}}}
\rightarrow Ax\in\mathbb{R}^{r}
\rightarrow BAx\in\mathbb{R}^{d_{\text{out}}}
$$

## 三、为什么它是“低秩”更新

### 3.1 秩的上界

由矩阵乘法的基本性质：

$$
\operatorname{rank}(BA)
\leq\min(\operatorname{rank}(B),\operatorname{rank}(A))
\leq r
$$

因此：

$$
\operatorname{rank}(\Delta W)\leq r
$$

即使 $\Delta W$ 的形状是 $d_{\text{out}}\times d_{\text{in}}$，它实际只允许在最多 $r$ 个独立方向上改变模型行为。

### 3.2 参数量对比

LoRA 的可训练参数为：

$$
\underbrace{r d_{\text{in}}}_{A}
+
\underbrace{d_{\text{out}}r}_{B}
=r(d_{\text{in}}+d_{\text{out}})
$$

假设注意力投影矩阵为 $4096\times4096$，且 $r=8$：

| 方法 | 可训练参数量 | 相对全量微调 |
|---|---:|---:|
| **全量微调** | $4096^2=16,777,216$ | $100\%$ |
| **LoRA，$r=8$** | $8(4096+4096)=65,536$ | 约 **0.39%** |
| **LoRA，$r=16$** | $16(4096+4096)=131,072$ | 约 **0.78%** |

因此，LoRA 显著降低了需要保存的梯度、优化器状态和任务适配权重规模。（来源：Hu et al., ICLR 2022）

## 四、训练时究竟更新什么

### 4.1 冻结与可训练参数

训练时：

$$
W_0\ \text{固定}
$$

$$
A,B\ \text{参与梯度下降}
$$

令损失函数为 $\mathcal{L}(h)$，则：

$$
h=W_0x+sBAx,\qquad s=\frac{\alpha}{r}
$$

对 $B$ 的单个元素 $B_{ij}$，有：

$$
h_i=\sum_k (W_0)_{ik}x_k+s\sum_j B_{ij}(Ax)_j
$$

因此：

$$
\frac{\partial h_i}{\partial B_{ij}}=s(Ax)_j
$$

根据链式法则：

$$
\frac{\partial\mathcal{L}}{\partial B_{ij}}
=\frac{\partial\mathcal{L}}{\partial h_i}\cdot s(Ax)_j
$$

这说明，梯度会更新 $B$ 与 $A$，但不会更新冻结的 $W_0$。

### 4.2 为什么通常令 $B=0$ 初始化

常见初始化方式是随机初始化 $A$，并令：

$$
B=0
$$

则初始时：

$$
\Delta W=\frac{\alpha}{r}BA=0
$$

$$
h=W_0x
$$

因此刚开始训练时，模型行为与原始预训练模型完全一致，随后再逐步学习任务所需的增量。（来源：Hu et al., ICLR 2022）

## 五、缩放系数 $\alpha/r$ 的作用

如果不缩放，随着秩 $r$ 增大，$BA$ 的数值规模可能随中间求和项数量变化。LoRA 使用：

$$
\Delta W=\frac{\alpha}{r}BA
$$

其目的在于让不同秩设置下的更新幅度更易比较、更易稳定训练。

需要注意：$\alpha/r$ 是训练稳定性与超参数可迁移性的工程设计，并不是由“低秩分解”本身必然推导出的唯一系数。（来源：Hu et al., ICLR 2022）

## 六、与注意力层的对应关系

Transformer 自注意力中常见的投影为：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

若仅对查询投影使用 LoRA，则：

$$
Q=X\left(W_{Q0}+\frac{\alpha}{r}B_QA_Q\right)
$$

实践中，LoRA 常被施加到注意力模块中的 **Query（Q）**、**Value（V）** 投影，也可作用于 Key、输出投影或 MLP 线性层；具体目标模块取决于模型结构、训练预算与任务。（来源：Hu et al., ICLR 2022；Hugging Face PEFT 文档）

## 七、PyTorch 最小实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()

        # 冻结的预训练权重 W0
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=False,
        )

        # ΔW = (alpha / rank) · B · A
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, x):
        base = F.linear(x, self.weight)             # x W0^T
        lora = F.linear(F.linear(x, self.A), self.B)  # x A^T B^T
        return base + self.scaling * lora


# 初始时 B = 0，因此 LoRA 层等价于原始线性层
layer = LoRALinear(in_features=4, out_features=3, rank=2, alpha=4)
x = torch.randn(5, 4)

base_output = F.linear(x, layer.weight)
lora_output = layer(x)

assert torch.allclose(base_output, lora_output)
```

### 7.1 代码与公式映射

| 代码变量 | 数学符号 | 作用 |
|---|---|---|
| `weight` | **$W_0$** | 冻结的预训练权重 |
| `A` | **$A$** | 将输入投影至低维空间 |
| `B` | **$B$** | 将低维特征投影回输出空间 |
| `scaling` | **$\alpha/r$** | 控制 LoRA 更新幅度 |
| `base` | **$W_0x$** | 原模型输出 |
| `lora` | **$BAx$** | 低秩适配增量 |

## 八、合并权重与推理

训练结束后，可将 LoRA 增量直接合并：

$$
W_{\text{merged}}=W_0+\frac{\alpha}{r}BA
$$

推理时：

$$
h=W_{\text{merged}}x
$$

这与未合并时的计算完全等价：

$$
W_{\text{merged}}x=W_0x+\frac{\alpha}{r}BAx
$$

合并后无需额外执行两次低秩矩阵乘法，因此可以避免 LoRA 分支带来的额外推理开销。（来源：Hu et al., ICLR 2022）

## 九、结论与适用条件

1. **LoRA** 用 $\Delta W=\frac{\alpha}{r}BA$ 代替完整权重更新，且其增量秩不超过 **$r$**。
2. 当 $r\ll d_{\text{in}},d_{\text{out}}$ 时，训练参数量从 $d_{\text{out}}d_{\text{in}}$ 降至 $r(d_{\text{in}}+d_{\text{out}})$。
3. 它适合多任务适配、显存受限微调、保存多个轻量任务适配器等场景。
4. LoRA 是对“所需更新近似低秩”的建模假设；若任务需要高秩的大幅参数改变，过小的 $r$ 可能限制效果。

## 参考资料

1. Hu, E. J. et al. **LoRA: Low-Rank Adaptation of Large Language Models**. *ICLR*, 2022.
2. Hugging Face. **PEFT Documentation: LoRA**.
