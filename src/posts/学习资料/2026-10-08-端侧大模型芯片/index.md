---
layout: post.njk
post_id: 2026-10-08-端侧大模型芯片
archive: 学习资料
title: 端侧大模型芯片
date: 2026-10-08
updated: 2026-10-08
tags:
  - post
---
# 1. Netron
- [Netron 网页](https://netron.app/) 
- **神经网络模型可视化工具**。
- 支持 ONNX、TensorFlow Lite、PyTorch 等格式。
- 拖入模型即可查看网络结构、算子、张量形状。

# 2. Coral NPU 裸机编程 — 
- [Google Coral 文档](https://developers.google.com/coral/guides/intro-platform)
- [github 项目](https://github.com/google-coral/coralnpu)
- Coral NPU 是 **32 位 RISC-V 微控制器**，裸机运行，无操作系统。
- 需自备：RISC-V 交叉编译器、链接脚本、启动代码。
- 启动需配置：
  - `RESET_CONTROL`：释放复位、使能时钟
  - `PC_START`：ITCM 中程序起始地址
- 引导程序任务：加载 ITCM/DTCM，设置 `PC_START`。
- 前置环境：Bazel 6.2.1、Python 3.9–3.12。
- 可用模拟器：MPACT（行为级，快）、Verilator（周期精确，准）。
