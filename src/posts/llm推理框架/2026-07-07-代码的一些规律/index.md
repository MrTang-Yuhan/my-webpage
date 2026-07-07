---
layout: post.njk
post_id: 2026-07-07-代码的一些规律
archive: llm推理框架
title: 代码的一些规律
date: 2026-07-07
description: ""
tags:
  - post
---
# 自动微分约定：梯度形状守恒

## 核心约定

对于**标量损失** $L \in \mathbb{R}$，参数 $x$ 的梯度形状**严格等于** $x$ 的形状：

$$\frac{\partial L}{\partial x} \in \mathbb{R}^{n_1 \times \cdots \times n_k} \quad\Longleftrightarrow\quad x \in \mathbb{R}^{n_1 \times \cdots \times n_k}$$

即：

$$\text{shape}\left(\frac{\partial L}{\partial x}\right) \equiv \text{shape}(x)$$

- 标量损失 $L$ 是一个**数值**，对张量 $x$ 的每个元素 $x_i$ 求偏导 $\frac{\partial L}{\partial x_i}$，结果自然与 $x$ 同形。
- 自动微分框架（PyTorch、JAX 等）的 `x.grad` 严格遵循此形状守恒，以便**原地更新**：$x \leftarrow x - \eta \cdot \text{grad}$。
---

## 注释和调试注意

- 注释标注变量的形状。


