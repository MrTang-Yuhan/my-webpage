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

符号规则
