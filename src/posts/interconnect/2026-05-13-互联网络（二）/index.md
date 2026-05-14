---
layout: post.njk
archive: interconnect
post_id: 互联网络
title: 互联网络（二）
date: 2026-05-14
description: ""
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

其中 $A^\top$ 就是 $A$ 的伴随通信算子

## 伴随通信算子的作用

当前向传播使用通信算子 $A$ 时，反向传播就必须使用伴随通信算子 $A^\top$。

如下图，使用的

![](img/tensor-parallelism.png)


