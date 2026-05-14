---
layout: post.njk
archive: interconnect
post_id: 互联网络
title: 互联网络（二）
date: 2026-05-14
description: 伴随通信算子和共轭算子
tags:
  - post
  - interconnection
---
## 什么是伴随通信算子

在神经网络的自动微分过程中，通信算子有一个特别重要的规律：

如果前向传播过程中使用通信算子 $A$，存在：
$$
y = Ax
$$

那么反向传播时：

$$
\nabla x = A^\top \nabla y
$$

其中 $\nabla y=\frac{\partial L}{\partial y}$, $\nabla x=\frac{\partial L}{\partial x}$

其中 $A^\top$ 就是 $A$ 的伴随通信算子。

常用通信算子及其伴随算子汇总如下[^1]：

[^1]: 这里都是指的"裸通信算子"，即只在意通信相关。

  比如在前向传播时，使用 all-reduce-sum 算子
  用矩阵乘法表示 $\mathbf{y} = A\mathbf{x}$，矩阵 $A$ 必须是一个 $p \times p$   的**全 1 矩阵**：

  $$
  A =
  \begin{bmatrix}
  1 & 1 & \cdots & 1 \\
  1 & 1 & \cdots & 1 \\
  \vdots & \vdots & \ddots & \vdots \\
  1 & 1 & \cdots & 1
  \end{bmatrix}_{p \times p}
  $$

  反向传播时，上游传来梯度向量：

  $$\nabla \mathbf{y} = [\nabla y_1, \nabla y_2, \dots, \nabla y_p]^\top$$

  我们需要计算 $\mathbf{x}$ 的梯度。根据链式法则：

  $$\nabla \mathbf{x} = A^\top \nabla \mathbf{y}$$

  关键在于**全 1 矩阵是对称的**：

  $$A^\top = A$$

  因为矩阵中每个元素都是 1，转置后仍然每个元素都是 1，矩阵不变。

  因此：

  $$\nabla \mathbf{x} = A \nabla \mathbf{y}$$

  也就是说，反向计算与正向计算的矩阵**完全相同**：

  $$\nabla x_i = \sum_{j=1}^p \nabla y_j, \quad \text{对所有 } i$$

  每台设备 $i$ 都收到所有上游梯度 $\nabla y_j$ 的总和。

| 前向通信算子 | 伴随/反向通信算子 | 说明与简单举例 |
|---|---|---|
| `AllReduce` | `AllReduce` | 所有 rank 的张量先求和/平均，再把结果发给所有 rank。反向仍是全局规约。例：数据并行中，各 GPU 计算本地梯度后用 `AllReduce` 得到全局平均梯度。 |
| `Broadcast` | `Reduce` | 前向是一到多，反向是多到一。例：rank 0 将参数 `W` 广播给所有 GPU；反向时各 GPU 上关于 `W` 的梯度需要 `Reduce` 回 rank 0。 |
| `Reduce` | `Broadcast` | 前向是多到一，反向是一到多。例：多个 GPU 的 loss 被 `Reduce` 到 rank 0；反向时 rank 0 上的梯度信号需要 `Broadcast` 回其他 rank。 |
| `Scatter` | `Gather` | 前向是把一个完整张量切分后分发到多个 rank；反向要把各分片梯度收集回来。例：rank 0 将输入 batch 切成多份 `Scatter` 给各 GPU；反向时各 GPU 的输入梯度用 `Gather` 收回。 |
| `Gather` | `Scatter` | 前向是多个 rank 的数据收集到一个 rank；反向要把梯度切开再分发回去。例：评估时各 GPU 的预测结果 `Gather` 到 rank 0；若参与反向，rank 0 上的梯度需 `Scatter` 回各 GPU。 |
| `AllGather` | `ReduceScatter` | 前向是各 rank 的分片被收集成完整张量，并且每个 rank 都拿到完整结果；反向时完整梯度需要规约后再按分片分发。例：FSDP 中前向前用 `AllGather` 收集完整参数，反向后用 `ReduceScatter` 得到各自的梯度分片。 |
| `ReduceScatter` | `AllGather` | 前向是先对所有 rank 的张量规约，再把结果切分给各 rank；反向需要把分片梯度重新收集成完整梯度。例：ZeRO/FSDP 中用 `ReduceScatter` 同步并切分梯度；反向对应路径需要 `AllGather` 汇集分片梯度。 |
| `AllToAll` | `AllToAll` | 前向是所有 rank 之间互相交换分片；反向通常也是一次相反方向或相反维度的 `AllToAll`。例：MoE 中先用 `AllToAll` 把 token 分发给不同专家，反向时再用 `AllToAll` 把梯度送回原 token 所在 rank。 |
| `Send` | `Recv` | 前向点对点发送数据，反向对应接收梯度。例：流水线并行中 stage 0 前向 `Send` 激活到 stage 1；反向时 stage 0 需要 `Recv` 来自 stage 1 的激活梯度。 |
| `Recv` | `Send` | 前向点对点接收数据，反向对应发送梯度。例：流水线并行中 stage 1 前向 `Recv` stage 0 的激活；反向时 stage 1 需要 `Send` 激活梯度回 stage 0。 |

## 伴随通信算子的作用

当前向传播使用通信算子 $A$ 时，反向传播就必须使用伴随通信算子 $A^\top$。

然而，这似乎有一个例外，如下图的 [tensor parallelism 架构](https://my-webpage-adu.pages.dev/posts/mode-parallelism/tensor-parallelism%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%B8%80%EF%BC%89/)：

![](img/tensor-parallelism.png)

根据论文中的说法，有：

- **$g$** 正向传播使用 all-reduce，反向传播使用 identity。
- **$f$** 正向传播使用 identity，反向传播使用 all-reduce。

然而，以 $f$ 举例，在前向传播过程中，它实际上做了个 copy/replicate 
