---
layout: post.njk
post_id: 2026-08-27-解析-flashattention-5-triton-和-flashattention
archive: llm推理框架
title: 解析 FlashAttention（5）：FlashAttention 和常用 triton 算子
date: 2026-08-27
tags:
  - post
---
# 1. triton 代码

[triton_docs_tutorials-main.zip](attach/triton_docs_tutorials-main.zip)

附件中的代码包含以下算子的 **triton 实现**，以及 `.ipynb` 文件内的可视化交互图：

- **04_vector_addition**：向量加法算子。
- **05_fused_softmax**：融合 softmax 算子。
- **06_matmul**：矩阵乘算子。
- **07_dropout**：dropout 算子。
- **08_layernorm**：layernorm 算子。
- **09_flash_attention**：FlashAttention 算子。triton 代码实现还没怎么看懂。

