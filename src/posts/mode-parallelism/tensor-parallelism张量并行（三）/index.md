---
layout: post.njk
archive: mode-parallelism
title: Tensor Parallelism张量并行（三）
date: 2026-05-13
tags:
  - post
---
在分布式大模型训练（如 GPT-3, Llama 3, DeepSeek）中，张量并行 (Tensor Parallelism, TP) 是处理超大规模参数的核心技术。而 Megatron-LM 的 TP 源码设计充满了系统工程的智慧。

今天我们将深入 Megatron-LM 剖析其最基础的组件——**ColumnParallelLinear**（列并行线性层）。我们将重点拆解两个看似简单的两个关键函数：

- `copy_to_tensor_model_parallel_region` 
- `gather_from_tensor_model_parallel_region`

---

## 一、什么是 Column Parallelism（列并行）？

在标准 PyTorch 中，一个线性层（Linear Layer）的计算逻辑是 $Y = XW$。其中：

- $X$: 输入矩阵，形状 $[Batch, Sequence, Hidden_{in}]$
- $W$: 权重矩阵，形状 $[Hidden_{in}, Hidden_{out}]$
- $Y$: 输出矩阵，形状 $[Batch, Sequence, Hidden_{out}]$

当 $W$ 太大无法放入单张 GPU 时，**列并行的做法是将 $W$ 沿着“列”的方向切分。**

假设我们有 2 张 GPU（$N = 2$）：

- 我们将 $W$ 切分为 $[W_1, W_2]$。
- **GPU 0 维护 $W_1$，计算 $Y_1 = XW_1$。**
- **GPU 1 维护 $W_2$，计算 $Y_2 = XW_2$。**
- 最终输出 $Y$ 就是 $[Y_1, Y_2]$ 的拼接。

为了实现这个数学逻辑，代码必须解决两个问题：**输入 $X$ 怎么分发？输出 $Y$ 怎么聚合？这就是 ColumnParallelLinear 的核心职责。**

以下是图片内容转换为 Markdown 格式，公式使用 `$` 包裹，且不额外渲染（保持原始 Markdown 文本）：

## 二、源码核心逻辑

在 Megatron-LM 的 layers.py 中，ColumnParallelLinear 的前向传播代码大致如下（简化版）：

```python
class ColumnParallelLinear(torch.nn.Module):
    def forward(self, input_):
        # 1. 处理输入数据的流向
        input_parallel = copy_to_tensor_model_parallel_region(input_)

        # 2. 核心计算：本地矩阵乘法
        # 此时 input_parallel 是完整的，weight 是切分后的
        output_parallel = F.linear(input_parallel, self.weight, self.bias)

        # 3. 处理输出数据的聚合
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel  # 保持切分状态（用于衔接 RowParallelLinear）
        
        return output
```

这里出现了两个至关重要的函数，它们分别控制了 Forward（前向）和 Backward（后向）的数据流动。它们是一对 **互逆的操作**。

### 1. 入口：copy_to_tensor_model_parallel_region

这个函数位于线性层的 **输入端**。

- **前提状态**：输入 $X$ 是 **Replicated（复制的）**。即每张 GPU 上都有一个完全一样的 $X$ 副本（例如来自上一层的 LayerNorm 输出）。
- **Forward 行为：Identity（直通）**
  - 因为每张卡都需要完整的 $X$ 来和自己的 $W_i$ 做乘法，所以数据不需要动，直接透传。
- **Backward 行为：All-Reduce（Sum）**
  - **原理**：在反向传播时，GPU 0 会算出 $X$ 的梯度 $\nabla X_0$（基于 $W_1$），GPU 1 会算出 $X$ 的梯度 $\nabla X_1$（基于 $W_2$）。
  - 因为 $X$ 是共享的，它对 $Y_1$ 和 $Y_2$ 都有贡献，根据链式法则，总梯度
    $$
    \nabla X = \nabla X_0 + \nabla X_1
    $$
  - 所以，必须在此时触发 All-Reduce 通信，将所有卡的梯度加起来。

## 2. 出口：gather_from_tensor_model_parallel_region

这个函数位于线性层的输出端。

- **前提状态**：本地计算出的 $Y_i$ 是 **Sharded（分片的）**。GPU 0 只有结果的左半部分，GPU 1 只有右半部分。
- **Forward 行为**：**All-Gather（全收集）。**
  - **原理**：为了让下一层（或用户）拿到完整的输出 $Y$，需要把各个 GPU 上的碎片 $[Y_1, Y_2, \dots]$ 拼起来。
  - 通信后，每张 GPU 都拥有了完整的 $Y$。

- **Backward 行为**：**Split（切分）。**
  - **原理**：反向传播传回来的梯度 $\nabla Y$ 是完整的。但 GPU 0 只需要维护 $W_1$ 对应的梯度，也就是 $\nabla Y$ 的左半部分。
  - 因此，这里不需要通信，只需要把传回来的梯度“切一刀”，保留属于自己的那部分即可。

---

# 三、全流程通信量与参数分析

让我们通过一个具体的例子来量化这个过程。

## 场景设定

- **Hidden Size (H):** 4096  
- **Sequence Length (S):** 2048  
- **Batch Size (B):** 1  
- **精度:** BF16 (2 Bytes)  
- **TP Size (N):** 2 (2张 GPU)  

---

## 1. 通信次数

在一个完整的训练步（Forward + Backward）中，**共发生 2 次通信。**

- Forward: 1 次 All-Gather（由 gather_from 触发）  
- Backward: 1 次 All-Reduce（由 copy_to 触发）  

---

## 2. 通信量计算

请注意，**通信传输的是激活值（Activations），而不是权重。** 权重的梯度是在本地计算并同步的（那是 DP 的事，或者是优化器步骤的事，这里只谈 TP 的反向传播）。

### A. Forward 阶段 (All-Gather)

- 每张卡产出的局部结果大小：$B \times S \times (H/N)$  
- 我们需要把这些拼成 $B \times S \times H$。  
- **总通信量近似于（当 GPU 数量 N 足够多时）：** $B \times S \times H \times 2$ Bytes  
- **数值代入：** $1 \times 2048 \times 4096 \times 2 \approx 16 \, \text{MB}$  

### B. Backward 阶段 (All-Reduce)

- 每张卡算出的对输入 $X$ 的梯度大小：$B \times S \times H$  
- 我们需要把这些梯度加起来。  
- **总通信量近似于（当 GPU 数量 N 足够多时）：** $B \times S \times H \times 2$ Bytes  
- **数值代入：** 同样约为 **16 MB**  

---

**总结：** 虽然权重矩阵可能很大（例如 $4096 \times 12288$），但 TP 的通信瓶颈主要取决于 Sequence Length 和 Batch Size。

# 四、流程实战演练

为了彻底讲清楚，我们用极小的数字手动跑一遍流程。

## 设定：

- TP Group: GPU0, GPU1
- 输入 $X = [1, 2]$ (形状 $1 \times 2$)
- 权重 $W$ (形状 $2 \times 4$)，值为全 1 矩阵。
  - $W$ 切分：GPU0 持有前 2 列，GPU1 持有后 2 列。
  $$
  W_{local} (\text{GPU0}) = 
  \begin{bmatrix}
  1 & 1 \\
  1 & 1
  \end{bmatrix}, \quad W_{local} (\text{GPU1}) = 
  \begin{bmatrix}
  1 & 1 \\
  1 & 1
  \end{bmatrix}
  $$

---

# 阶段 1: 前向传播 (Forward)

## 1. 初始状态：

- GPU0 有 $X = [1, 2]$
- GPU1 有 $X = [1, 2]$

---

## 2. copy_to 函数：

- 输入 $[1, 2]$ -> **Identity** -> 输出 $[1, 2]$。无通信。

---

## 3. Local Linear ($X \times W_{local}$)：

- GPU0: 
  $$
  [1, 2] \times 
  \begin{bmatrix}
  1 & 1 \\
  1 & 1
  \end{bmatrix} =
  [3, 3]
  $$
  (这是 $Y_{part1}$)

- GPU1: 
  $$
  [1, 2] \times 
  \begin{bmatrix}
  1 & 1 \\
  1 & 1
  \end{bmatrix} =
  [3, 3]
  $$
  (这是 $Y_{part2}$)

---

## 4. gather_from 函数：

- 触发 **All-Gather**。
- GPU0 拿到 GPU1 的 $[3, 3]$，拼接得到 $Y = [3, 3, 3, 3]$。
- GPU1 拿到 GPU0 的 $[3, 3]$，拼接得到 $Y = [3, 3, 3, 3]$。
- **最终输出**：两张卡都输出了完整的 $[3, 3, 3, 3]$。

---

# 阶段 2: 反向传播 (Backward)

假设 Loss 对输出 $Y$ 的梯度 $\nabla Y = [0.1, 0.1, 0.2, 0.2]$。

## 1. 初始状态：

- GPU0 和 GPU1 都接收到了完整的 $\nabla Y = [0.1, 0.1, 0.2, 0.2]$。

## 2. gather_from 的 Backward (Split):

- GPU0 切前一半 -> $\nabla Y_{local} = [0.1, 0.1]$
- GPU1 切后一半 -> $\nabla Y_{local} = [0.2, 0.2]$
- 注意：这里没有通信。

---

## 3. Local Backward (计算对输入 $X$ 的梯度)：

- 公式：$\nabla X_{local} = \nabla Y_{local} \times W^T_{local}$
- GPU0: $[0.1, 0.1] \times 全1矩阵 = [0.2, 0.2]$
- GPU1: $[0.2, 0.2] \times 全1矩阵 = [0.4, 0.4]$
- 此时，两张卡对同一个输入 $X$ 算出了不同的梯度贡献。

---

## 4. copy_to 的 Backward (All-Reduce)：

- 触发 **All-Reduce (Sum)**。
- 计算 $[0.2, 0.2] + [0.4, 0.4] = [0.6, 0.6]$。
- 最终结果：GPU0 和 GPU1 都得到了输入 $X$ 真正的梯度 $\nabla X = [0.6, 0.6]$，可以继续传给上一层。

---

# 五、总结

理解 ColumnParallelLinear 的关键在于记住上面的两个通信组件。

## 技术洞察：

1. **为什么需要 RowParallelLinear？** 如果所有的层都用 Column Parallel，那么每一层结束后都要 All-Gather，下一层开始前又是 Identity。这样每层都有通信。Megatron 的优化是将 ColumnParallel 和 RowParallel 串联，中间不进行 All-Gather，让数据保持切分状态流转，从而将 2 次通信减少为 1 次 All-Reduce（在 Row 层末端）。这块不理解可以参考我之前写的这篇文章：[tensor-parallelism张量并行（三）](/src/posts/mode-parallelism/tensor-parallelism张量并行（三）)。

2. **网络带宽很重要：** 对于 4096 隐藏层、2K 长度的输入，每一步大约 32MB 的数据交换。在训练几千亿参数模型时，Hidden Size 可能达到 12K+，通信量成倍增加，这就是为什么我们需要 NVLink 和 InfiniBand 的原因。
