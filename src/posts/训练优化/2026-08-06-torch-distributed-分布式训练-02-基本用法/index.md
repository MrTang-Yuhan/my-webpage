---
layout: post.njk
post_id: 2026-08-06-torch-distributed-分布式训练-02-基本用法
archive: 训练优化
title: torch.distributed 分布式训练（02）：基本用法
date: 2026-08-06
tags:
  - post
---
- torchrun启动参数

- 日志的使用

- dist分布式整体流程
- dist.dist.new_group() 子通信组可以直接组内通信
- dist.init_process_group() 默认组包含所有进程

- apply的用法，以及里面的ctx。apply 是自定义 torch.autograd.Function 的标准调用方式

- tp,pp,dp三种的结果图
