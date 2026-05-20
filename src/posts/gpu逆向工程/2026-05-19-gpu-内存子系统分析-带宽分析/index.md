---
layout: post.njk
archive: gpu逆向工程
title: GPU 内存子系统分析--带宽分析
date: 2026-05-20
tags:
  - post
---
# 带宽和吞吐量含义区分

由于 [Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis](https://arxiv.org/abs/2501.12084) 和 [Dissecting the NVIDIA Blackwell Architecture with Microbenchmarks](https://arxiv.org/abs/2507.10789) 两篇论文中对带宽和吞吐量的描述较为混乱，本文统一以 NVIDIA 白皮书中的定义为准。具体总结如下：

首先需要明确 GPU 中 **带宽（bandwidth）** 和 **吞吐量（throughput）** 这两个概念的区别：

- **带宽**：指单位时间内某个数据通路所能传输的数据量。  
  在 NVIDIA 白皮书中，带宽通常用于描述 HBM 显存带宽、NVLink / NVLink Switch 带宽等，常见单位为 GB/s、TB/s。

- **吞吐量**：指单位时间内 GPU 能够完成的计算操作、指令或任务的数量。  
  在 NVIDIA 白皮书中，吞吐量常用于描述 P16 / BF16 / FP8 Tensor Core 吞吐量等，最常见单位为 TFLOP/s、PFLOP/s。对于整数运算，则经常使用 TOPS（tera operations per second）。

