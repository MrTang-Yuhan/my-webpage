---
layout: post.njk
post_id: 2026-06-30-gpu-内存子系统分析-四-l2-布局
archive: gpu逆向工程
title: GPU 内存子系统分析（四）：L2 布局
date: 2026-06-30
tags:
  - post
---
# L2 Slice 到底是多少？

在 [Blackwell GPU Architecture](https://www.emergentmind.com/topics/blackwell-gpu-architecture) 和 [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/html/2507.10789v2) 中都提到 GB203，也就是 RTX 5080 GPU 的芯片型号下，L2 缓存是一个 monilithic cache。
