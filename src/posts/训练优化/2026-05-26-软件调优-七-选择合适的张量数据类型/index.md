---
layout: post.njk
post_id: 2026-05-26-软件调优-七-选择合适的张量数据类型
archive: 训练优化
title: " 软件调优（七）：选择合适的张量数据类型"
date: 2026-05-26
tags:
  - post
---
最初，神经网络训练使用的张量类型都是 fp32，但速度慢、占用显存大。此后，NVIDIA 提出了[混合精度训练](https://developer.nvidia.com/blog/video-mixed-precision-techniques-tensor-cores-deep-learning/)，研究人员结合使用 fp16 和 fp32 两种浮点精度格式，这一创新极大地提升了模型训练速度。
