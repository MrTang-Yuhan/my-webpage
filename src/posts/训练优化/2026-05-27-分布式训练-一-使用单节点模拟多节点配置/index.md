---
layout: post.njk
post_id: 2026-05-27-分布式训练-一-使用单节点模拟多节点配置
archive: 训练优化
title: 分布式训练（一）：使用单节点模拟多节点配置
date: 2026-05-27
tags:
  - post
---

如果一个节点配备了多张 GPU，可以将该节点内的每张 GPU 模拟成一个独立的节点，从而实现“单节点模拟多节点”的配置。

需要注意的是，这种模拟方式下，各 GPU 之间通过本地的 NVLink 或 PCIe 进行互联，而不是像真实多节点环境那样使用 InfiniBand 或以太网。因此，该方法是否适用，取决于具体测试对网络互联特性的依赖程度。

**更详细的配置方法与实现原理，可参考 [Machine Learning Engineering by Stas Bekman](https://github.com/stas00/ml-engineering) 一书的 **Emulate a multi-node setup using just a single node** 一节（第 271 页）。**

> 该方案依赖 SSH 进行通信，因此在容器环境中配置会较为繁琐，不建议在容器内进行。并且要求运行环境具备多 GPU 支持。
