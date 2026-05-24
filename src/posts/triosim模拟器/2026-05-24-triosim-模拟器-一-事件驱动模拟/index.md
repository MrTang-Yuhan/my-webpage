---
layout: post.njk
post_id: 2026-05-24-triosim-模拟器-一-事件驱动模拟
archive: triosim模拟器
title: " TrioSim 模拟器 （一）：事件驱动模拟"
date: 2026-05-24
tags:
  - post
---
# 事件驱动模拟器（Event-Driven Simulator）

## 事件驱动模拟器简介

事件驱动模拟器是只有当某个事件发生时，模拟器才处理它，并把模拟时间跳到下一个事件的发生时间。一般它的组成是：

1. 维护一个**按时间排序的事件队列**，
2. 每次取出**最早发生的事件**执行，
3. 执行事件时可能产生新的未来事件，
4. 然后继续处理下一个事件。

在事件驱动模拟器中，“事件”被定义为“在某个模拟时间点要发生的事情”。例如：

- 时间 10ns：CPU 发出一个内存请求
- 时间 15ns：请求到达 L2 Cache
- 时间 30ns：DRAM 返回数据
- 时间 35ns：CPU 收到响应

这里每一项都是事件。

## 事件驱动模拟器的优点

- 由于事件驱动模拟器仅在事件发生时推进模拟过程，而无需对每个周期进行精确建模，因此其**仿真效率通常显著高于周期精确模拟器**。

- 事件驱动模拟器通常采用事件或消息作为模块间的通信机制，这种设计有效**降低了模块之间的耦合度**，并提升了模拟器的可维护性与可扩展性。

- 事件驱动模拟器更适合用于模拟异步系统，能够**自然地描述事件触发、消息传递、请求响应等行为**。

# TrioSim 模拟器

Triosim 是一个基于事件驱动的深度学习训练/推理模拟器，用于评估不同并行策略（数据并行、张量并行、流水线并行）的性能表现。它自带以下 6 个 case：

- Case 0: 纯推理回放，不做梯度同步（无 AllReduce）
- Case 1: 推理 + Ring AllReduce
- Case 2: 分布式数据并行（Data Parallel）
- Case 3: 张量并行（Tensor Parallel）
- Case 4: 流水线并行（Pipeline Parallel）
- Case 5: HOP（High Bandwidth Optical Path） AllReduce 通信测试

其中，
- Case 0 仅有 forward 阶段；
- Case 1 ~ Case 4 有 forward + backwrad 阶段；
- Case 5 重点测试 AllReduce 通信。

TrioSim 模拟器是基于底层事件驱动引擎 Akita 构建的。考虑到直接介绍 Akita 引擎较为抽象难懂，我将在介绍 TrioSim 模拟器的建模过程中，对用到的 Akita 引擎知识进行逐步解释。

