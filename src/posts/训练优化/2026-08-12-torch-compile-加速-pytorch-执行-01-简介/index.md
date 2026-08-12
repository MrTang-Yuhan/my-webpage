---
layout: post.njk
post_id: 2026-08-12-torch-compile-加速-pytorch-执行-01-简介
archive: 训练优化
title: " torch.compile 加速 Pytorch 执行 (01)：简介"
date: 2026-08-12
tags:
  - post
---
# torch
torch.compile 的底层是一个多层编译流水线，核心思想是：把 Python 代码动态地抓取成计算图（Graph），然后对这个图进行深度优化，最后生成高效的机器码。
