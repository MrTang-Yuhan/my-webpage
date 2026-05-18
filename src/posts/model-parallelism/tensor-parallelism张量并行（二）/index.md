---
layout: post.njk
archive: mode-parallelism
title: Tensor Parallelism张量并行（二）
date: 2026-05-13
tags:
  - post
---
这是一篇基于 Megatron-LM Tensor Parallelism (TP) 核心逻辑的源码解析文章。

代码路径：Megatron-LM/megatron/legacy/model/transformer.py

## 前言

在大模型训练中，Tensor Parallelism (张量并行，TP) 是最核心的并行策略之一。它不仅仅是简单的分片训练，而是将一个巨大的矩阵乘法操作切分，分配到不同的 GPU 上同时计算，最后再通过通信合并结果。

本文将基于 ParallelMLP 类的源码，深入剖析 Megatron 是如何通过 **Column Parallel (列并行)** 和 **Row Parallel (行并行)** 的组合，实现高效的模型切分与通信的。



## 1. 架构总览：TP 的三明治结构

在 Transformer 的 MLP 层（Feed Forward Network）中，计算逻辑通常是：

$Y = \text{Activation}(X \cdot A) \cdot B$

其中：

- $X$: 输入 (Batch, Seq, Hidden)
- $A$: 第一个全连接层权重 ($h \to 4h$)
- $B$: 第二个全连接层权重 ($4h \to h$)

Megatron 的天才之处在于它没有在每一步都进行通信，而是设计了一个 **Column -> Row** 的“三明治”结构，将通信延迟到最后一跳。

代码中的 `ParallelMLP` 完美体现了这一逻辑：

1. `dense_h_to_4h`: 使用 **ColumnParallelLinear**
2. `activation_func`: 在切分的数据上独立进行非线性变换。
3. `dense_4h_to_h`: 使用 **RowParallelLinear**

## 2. 第一层：ColumnParallelLinear (切分与映射)

这是 MLP 的第一刀。我们需要将 hidden state 投影到更大的维度（通常是 $4h$）。

### 源码分析

```python
# Project to 4h.
self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
    config.hidden_size,
    ffn_hidden_size,
    # ... 省略部分参数
    gather_output=False,  # <--- 重点 1
    skip_bias_add=True,   # <--- 重点 2
    is_expert=is_expert,
)
```


### 核心逻辑解析

- **Mapping (映射逻辑)**: 假设权重矩阵 $A$ 的形状是 $[h, 4h]$。如果有 2 张 GPU (TP size = 2)，Megatron 会按列切分组矩阵：
  - GPU 0 只有 $A_{col1}$ (形状 $[h, 2h]$)
  - GPU 1 只有 $A_{col2}$ (形状 $[h, 2h]$)
  - 输入 $X$ 是完整的（或者说是被复制到两张卡上的）。
- **Forward 通信**: 注意参数 `gather_output=False`。通常矩阵乘法后我们希望得到完整的输出。但在这里，GPU 0 计算 $Y_1 = X \cdot A_{col1}$，GPU 1 计算 $Y_2 = X \cdot A_{col2}$。
- **Megatron 选择不进行 All-Gather 通信**。此时，每张卡上只持有一部分输出特征。这为下一层省去了巨大的通信开销。
- **Bias 处理**: `skip_bias_add=True` 是为了性能优化（Fusion），将 Bias 的加法推迟到 Activation 函数中一起做（见代码中的 `bias_gelu_impl`），这也是 Megatron 的经典优化手段。

## 3. 中间层：Activation (本地计算)

```python
if self.bias_gelu_fusion:
    intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
else:
    # ... 常规激活
    intermediate_parallel = self.activation_func(intermediate_parallel)
```

**解析**: 由于上一层的输出 `intermediate_parallel` 是按特征维度切分的（Partitioned per GPU），而 GeLU/SiLU 等激活函数是 **Element-wise (逐元素)** 的。这意味着：
$GeLU([Y_1, Y_2]) = [GeLU(Y_1), GeLU(Y_2)]$

**结论**: 这一步完全在本地 GPU 进行，不需要任何通信。

## 4. 第二层：RowParallelLinear (合并与还原)

这是 MLP 的收尾，将维度从 $4h$ 变回 $h$。

### 源码分析

```python
# Project back to h.
self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
    config.ffn_hidden_size,
    config.hidden_size,
    # ...
    input_is_parallel=True, # <--- 重点 3
    # ...
)
```

### 核心逻辑解析

- **Mapping (映射逻辑)**: 权重矩阵 $B$ 的形状是 $[4h, h]$。为了配合上一层的输出，这次我们按行切分矩阵：
  - GPU 0 持有 $B_{row1}$ (形状 $[2h, h]$)，对应上一层 GPU 0 输出的部分特征。
  - GPU 1 持有 $B_{row2}$ (形状 $[2h, h]$)，对应上一层 GPU 1 输出的部分特征。

- **Forward 通信 (All-Reduce)**: 这里的计算逻辑是：
  $$
  Output = Y_1 \cdot B_{row1} + Y_2 \cdot B_{row2}
  $$

即 $Z_1 = Y_1 \cdot B_{row1}$，$Z_2 = Y_2 \cdot B_{row2}$。这两个 $Z$ 的形状都是 $[Batch, Seq, h]$，但它们都只是最终结果的一部分加数。为了得到最终的 Output，**必须在所有 GPU 间进行一次 All-Reduce (Sum) 操作**。

  参数 `input_is_parallel=True` 告诉这一层：你的输入已经是切分过了，不需要再手动切分输入。


## 5. 总结

1. **延迟通信**：整个 MLP 块内部只有一次同步通信（在 RowParallel 的末尾），极大地提高了训练效率。

2. **显存优化**：权重矩阵被物理切分，单卡显存占用随 TP 数量线性降低。

3. **代码细节**：`gather_output=False` 和 `input_is_parallel=True` 是连接两层的关键信号。

