---
layout: post.njk
archive: mode-parallelism
title: transformer模型的GPU显存使用分析（三）：反向传播到底需要哪些中间结果
date: 2026-05-12
description: 本文旨在梳理训练阶段前向传播中必须保留的中间结果（如激活值、掩码、统计量等），以正确支持反向传播
tags:
  - post
  - GPU memory usage
  - transformer
  - model parallelism
---
# 基本原则

**反向传播时，某个梯度公式如果要用到前向里的某个值，这个值就要暂存。**

假设前向传播公式：

$$y = f(x)$$

发

## 汇总


## 推导
