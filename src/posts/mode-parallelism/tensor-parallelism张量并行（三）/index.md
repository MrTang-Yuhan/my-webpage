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

- `copy_to_tensor_model_parallel_region` 和
- `gather_from_tensor_model_parallel_region`。

---

## 一、什么是 Column Parallelism（列并行）？

在标准 PyTorch 中，一个线性层（Linear Layer）的计算逻辑是 $Y = XW$。其中：

- $X$: 输入矩阵，形状 $[Batch, Sequence, Hidden\_in]$
- $W$: 权重矩阵，形状 $[Hidden\_in, Hidden\_out]$
- $Y$: 输出矩阵，形状 $[Batch, Sequence, Hidden\_out]$

当 $W$ 太大无法放入单张 GPU 时，**列并行的做法是将 $W$ 沿着“列”的方向切分。**

假设我们有 2 张 GPU（$N = 2$）：

- 我们将 $W$ 切分为 $[W_1, W_2]$。
- **GPU 0 维护 $W_1$，计算 $Y_1 = XW_1$。**
- **GPU 1 维护 $W_2$，计算 $Y_2 = XW_2$。**
- 最终输出 $Y$ 就是 $[Y_1, Y_2]$ 的拼接。

为了实现这个数学逻辑，代码必须解决两个问题：**输入 $X$ 怎么分发？输出 $Y$ 怎么聚合？这就是 ColumnParallelLinear 的核心职责。**
