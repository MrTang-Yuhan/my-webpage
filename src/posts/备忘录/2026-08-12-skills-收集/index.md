---
layout: post.njk
post_id: 2026-08-12-skills-收集
archive: 备忘录
title: 受欢迎的 skills 收集
date: 2026-08-12
tags:
  - post
---
# 代码相关 skills

**安装时，优先将官网链接发送给对应的 harness，由其直接完成安装；若无法安装，再参照下方步骤操作。**

---

## [spec-kit](https://github.com/github/spec-kit/tree/main)

Spec Kit 是一个开源工具套件，为 **AI 编码助手** 提供结构化流程、可复用模板和有据可查的成果。它具备三个功能：

- **构建构建功能或应用。**
- **排查并修复异常行为。**
- **判断一个想法是否值得投入**。

### 1. 安装方法

[Speckit才是codex的最佳编程搭配，看大瑜实操！](https://zhuanlan.zhihu.com/p/1964737305861398909) 这份比官网讲的清晰很多，并且有使用指南。

### 2. 使用指南

[Spec Kit 中文文档](https://github.com/github/spec-kit/blob/main/README.zh-CN.md#%E8%A7%84%E8%8C%83%E9%A9%B1%E5%8A%A8%E5%BC%80%E5%8F%91) 有详细说明三个功能各自的实战指南和命令参考。

---

## [Archify](https://github.com/tt-a1i/archify)

**优先将官网链接发送给对应的 harness 安装。**

Archify skill 可以在对话里，把代码仓库或系统描述变成**漂亮、可靠、可交互的系统地图**。

### 1. 安装方法

[Archify 快速上手](https://tt-a1i.github.io/archify/start.html?agent=codex&type=architecture&source=direct&input=repository)

### 2. 使用指南

[Archify 中文文档](https://github.com/tt-a1i/archify/blob/main/README_ZH.md) 。

## [code-review-graph](https://github.com/tirth8205/code-review-graph/blob/staging/README.zh-CN.md)

你有没有遇到过这种情况：在 Harness 里问一句「这个认证流程是怎么跑的？」、「我改这个类会影响哪里？」或者「帮我 review 一下最近的改动」，它就开始一轮又一轮地搜索、读取、拼上下文。

时间一长，你会发现一个很现实的问题：AI 对代码库的理解，很难持续积累。

本文要讲的，就是如何用 code-review-graph 给本地代码仓库建立一张知识图谱，再通过 MCP 接入 Harness 。**这样 Harness 不再只会反复读文件，而是能像查地图一样查询代码结构、依赖关系、核心节点和影响范围**。

### 1. 安装方法和使用指南

相比官方教程，[开源 Claude Code 本地代码知识图谱：code-review-graph 完整上手攻略](https://zhuanlan.zhihu.com/p/2037933211586670828) 写的更清楚。

[code-review-graph 中文官网](https://github.com/tirth8205/code-review-graph/blob/staging/README.zh-CN.md) 有 **"使用说明"**、**"命令"** 页面。

---

## [Everything Claude Code (ECC)](https://github.com/affaan-m/ECC/blob/main/README.zh-CN.md)

ECC 不止是配置文件，而是一整套完整系统：技能体系、本能行为、记忆优化、持续学习、安全扫描，以及研究优先的开发模式。 包含可直接用于生产环境的**智能体、技能模块、钩子、规则、MCP 配置，以及兼容传统命令的适配层**——所有内容均经过 10 个多月高强度日常使用与真实产品开发迭代打磨而成。

这里介绍已用的**代码层**功能：

- [codebase-onboarding](https://github.com/affaan-m/ECC/blob/main/skills/codebase-onboarding/SKILL.md) 用于**分析不熟悉的代码库，并生成一份结构化的上手指南**，内容包括架构图、关键入口、代码规范约定以及一个初始的 CLAUDE.md 文件。适用于加入新项目，或首次在某个代码库中配置 Claude Code （**其他 harness 也能用**）的场景。

### 1. 安装方法

**优先将官网链接发送给对应的 harness 安装。**

### 2. 使用指南

- **[ECC 中文官网页](https://github.com/affaan-m/ECC/blob/main/README.zh-CN.md) 对仓库中的组件有详细说明**，可据此查阅各 agents、skills、commands 的作用与用法。
- skills 的使用指南请参看 [skills](https://github.com/affaan-m/ECC/tree/main/skills) 中每个具体 skill 的 `skill.md` 文件。

---

[academic-research-skills
](https://github.com/Imbad0202/academic-research-skills)

[nature-skills](https://github.com/Yuan1z0825/nature-skills)
