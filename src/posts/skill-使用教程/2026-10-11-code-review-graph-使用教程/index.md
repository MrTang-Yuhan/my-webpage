---
layout: post.njk
post_id: 2026-10-11-code-review-graph-使用教程
archive: skill 使用教程
title: code-review-graph 使用教程
date: 2026-10-11
updated: 2026-10-11
tags:
  - post
---
你有没有遇到过这种情况：在 Harness 里问一句「这个认证流程是怎么跑的？」「我改这个类会影响哪里？」或者「帮我 review 一下最近的改动」，它就开始一轮又一轮地搜索、读取、拼上下文。

时间一长，你会发现一个很现实的问题：**AI 对代码库的理解，很难持续积累**。

本文要讲的，就是如何用 [code-review-graph](https://github.com/tirth8205/code-review-graph/blob/staging/README.zh-CN.md) 给本地代码仓库建立一张知识图谱，再通过 MCP 接入 Harness。这样，Harness 不再只会反复读文件，而是能像查地图一样查询代码结构、依赖关系、核心节点和影响范围。

## 安装

把 code-review-graph 仓库直接发给 Harness，让它自行完成安装。

## 使用

### 1. 初始化 git

```bash
# code review graph 使用前必须初始化 git
git init
```

### 2. 构建图谱

在 Codex 等 Harness 中输入：

```
为这个项目构建 code review graph。
```

你也可以指定项目内的某些目录来构建。

### 3. 验证图谱已接入

在 Harness 中输入：

```
如果我修改认证模块，会影响哪些调用路径？
```

如果回复过程中调用了 `get_impact_radius_tool`、`query_graph_tool`、`semantic_search_nodes` 这类 MCP 工具，说明图谱已经正常接入。

也需要在命令行进一步确认：

```bash
# 查看图谱统计
code-review-graph status
```

输出的 `Nodes`、`Edges` 等信息不应该为 0。

### 4. 持续监听

在命令行启动监听模式，文件改动时图谱会自动更新：

```bash
code-review-graph watch
```
