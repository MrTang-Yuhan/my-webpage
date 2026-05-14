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

常用通信算子及其伴随算子汇总如下：

| Forward 算子 | Backward 算子 | 说明 |
|---|---|---|
| identity | identity | 普通恒等映射 |
| all-gather | reduce-scatter | 最常见的一对 |
| reduce-scatter | all-gather | 上一行反过来 |
| all-reduce-sum | all-reduce-sum | 自伴随 |
| all-reduce-mean | all-reduce-mean | 自伴随，但带 \(1/t\) 缩放 |
| broadcast | reduce-to-source (只有root rank 得到 reduce结果) | 复制出去，梯度加回来 |
| reduce-to-source | broadcast | 求和到 root，梯度广播回去 |
| gather-to-root | scatter-from-root | 收集到 root，梯度切开发回 |
| scatter-from-root | gather-to-root | 切开发出，梯度收集回来 |
| concat | split | 拼接的反向是切分 |
| split | concat | 切分的反向是拼接 |
| replicate / copy | sum-reduce | 多副本梯度要相加 |
| sum-reduce | replicate / broadcast | 求和的每个输入都收到同一份梯度 |
| all-to-all | inverse all-to-all | 通常仍表现为 all-to-all |
| send $i\to j$ | send gradient $j\to i$ | 点对点反向通信 |
| recv $i\to j$ | send gradient $j\to i$ | 接收的反向是发送梯度回去 |


## 伴随通信算子的作用

当前向传播使用通信算子 $A$ 时，反向传播就必须使用伴随通信算子 $A^\top$。

然而，这似乎有一个例外，如下图的 [tensor parallelism 架构](https://my-webpage-adu.pages.dev/posts/mode-parallelism/tensor-parallelism%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%B8%80%EF%BC%89/)：

![](img/tensor-parallelism.png)

根据论文中的说法，有：

- $f$ 正向传播使用 all-reduce，反向传播使用 identity。
- $g$ 正向传播使用 identity，反向传播使用 all-reduce。

