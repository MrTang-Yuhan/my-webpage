---
layout: post.njk
archive: GPU逆向工程
title: GPU 内存子系统分析--延迟分析
date: 2026-05-15
description: 分析GPU 的 L1/Shared Memory，L2，DRAM 的访问延迟
tags:
  - post
---
# 延迟的定义

延迟可以分为两种：

## True Latency （真实依赖延迟）

构造一串前后依赖的指令，让后一条指令必须等前一条指令的结果出来才能执行。

True Latency 反映在无法并行处理情况下的延迟。

  例如：
```assembly
FFMA R0, R0, R1, R2
FFMA R0, R0, R1, R2
FFMA R0, R0, R1, R2
FFMA R0, R0, R1, R2
```
每条指令都读写 R0，所以第 2 条必须等第 1 条的 R0 结果可用，第 3 条必须等第 2 条，以此类推。

这种情况下，硬件不能通过流水线重叠来隐藏延迟。

## Completion Latency (完成延迟 / 平均完成间隔)

构造一组互相独立的指令，它们之间没有数据依赖，所以硬件可以让它们并行、流水线化、重叠执行。

汇编层面类似：

```assembly
FFMA R0,  R0,  R20, R21
FFMA R1,  R1,  R20, R21
FFMA R2,  R2,  R20, R21
FFMA R3,  R3,  R20, R21
FFMA R4,  R4,  R20, R21
FFMA R5,  R5,  R20, R21
```

这些指令分别操作不同寄存器 R0, R1, R2, ...，所以彼此不需要等待。

假设单条 FFMA 的 true latency 是 10 cycles，但执行单元是流水线化的，可以每个 cycle 接收一条新指令。此时虽然每条指令从发射到结果可用仍然是 10 cycles，但因为流水线重叠了，所以从整体看，平均每条指令的完成间隔可能接近 1 cycle/instruction。

# 内存子系统的 True Latency

## L1，L2 和 DRAM 

我使用 pointer-chasing 方法测量 L1，L2 和 DRAM 的 true latency。

使用的平台是 NVIDIA 5080 GPU，Compute Capability 为 sm_120。

测量前记得:

- 锁定 gpu 和 Memory 的频率
  
  - 通过 `nvidia-smi lgc gpu_clocks` 锁定 gpu 频率
  - 通过 `nvidia-smi lmc mem_clocks` 锁定内存频率

- 避免使用 shared memory


1[](#L1)







