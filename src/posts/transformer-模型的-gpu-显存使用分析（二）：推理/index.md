---
layout: post.njk
archive: mode-parallelism
title: transformer 模型的 GPU 显存使用分析（二）：推理
date: 2026-05-11
description: transformer 模型的 GPU 显存使用分析：推理阶段分析
tags:
  - post
  - GPU memory usage
  - transformer
  - model parallelism
---
# 整体架构

这幅图展示了 Decode-only Transformer 的总体架构图：

![decode-only](img/overview.png)

# 维度分析

* `B`: batch size
* `T`: 当前输入序列长度
* `L`: Transformer 层数
* `d_model`: 隐藏层维度
* `n_head`： 注意力头数
* `d_head`: 每个注意力头的维度。d_head = d_model / n_head

# Transformer-Block 结构

输入 `x`:

-  `[B, T, d_model]`

输出 `out`: 

- `[B, T, d_model]`

# Attention 中的维度 

输入 `x`: 

- `[B, T, d_model]`

经过线性层，得到`Q`, `K`, `V`:

- `Q` = x @ Wq`
- `K` = x @ Wk`
- `V` = x @ Wv`

如果是标准多头注意力 MHA，则

- `Q, K, V`: [B, T, n_head, d_head]

一般会转置成：

- `Q, K, V`: `[B, n_head, T, d_head]`


