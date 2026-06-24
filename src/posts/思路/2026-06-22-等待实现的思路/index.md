---
layout: post.njk
post_id: 2026-06-22-等待实现的思路
archive: 思路
title: 等待实现的思路
date: 2026-06-22
description: ""
tags:
  - post
---
# 论文

## Agent Memory 的缓存实践可能性？

在阅读资料 [Agent Memory 这三年——AI 圈最热的问题，为什么三年都没解决](https://zhuanlan.zhihu.com/p/2043305140757778513) 时，产生了一个疑问：**Agent Memory 的遗忘有这么难吗？**

**能否借鉴缓存的策略，比如 LRU？或者使用 scratch 小模型来进行类似缓存的 Agent Memory 遗忘？**

## NPC Agent 的定义是否仍存在缺陷？

论文 [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) 使用认知科学的方法构建 Agent NPC。但是我有一个疑问，如果是认知科学的方法，每个 Agent NPC 是否应该具备以下三要素：
1. **心理活动**
2. **生理活动**
3. **一切社会关系的总和**

这样是否更加符合"人"的概念的定义？


# 实践
