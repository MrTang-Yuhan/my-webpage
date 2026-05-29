---
layout: post.njk
post_id: 2026-05-29-deepseek-v2-一-deepseekmoe
archive: 大模型架构
title: DeepSeek-V2（一）：DeepSeekMoE
date: 2026-05-29
description: ""
tags:
  - post
---
**MoE（混合专家）的核心目的就是显著减少训练和推理时的计算代价，同时保持甚至提升模型能力。**

---

# 从 Dense 大模型到普通 MoE 大模型

MoE 主要是对 FeedFoward Layer 进行改造。

![](img/attention_arch.png)
![](img/dense_feedfoward.png)


 
