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


# Non-Uniform L2 Cache Latency Across the Streaming Multiprocessors of an NVIDIA L40

> **作者**: Faruk Alpay*, Baris Basaran  
> **机构**: Department of Computer Engineering, Bahcesehir University, Istanbul, Turkey  
> **邮箱**: {faruk.alpay, baris.basaran}@bahcesehir.edu.tr  
> **arXiv**: [2606.22588](https://arxiv.org/abs/2606.22588) [cs.AR]  
> **发表日期**: 2026-06-21  
> **论文页数**: 16 pages, 10 figures, 5 tables  
> **代码/数据**: 论文提供了完整的 arXiv ancillary artifact（含 CUDA 源码、原始 CSV 数据、训练好的 placement oracle、复现脚本）  
> **CUDA 核心源码**: [`src/l2_topology.cu`](https://arxiv.org/src/2606.22588v1/anc/src/l2_topology.cu)（arXiv ancillary files）

---

