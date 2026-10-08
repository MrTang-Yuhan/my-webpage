---
layout: post.njk
post_id: 2026-10-08-远程连接和-wsl-连接配置-2-mobaxterm-免密连接-跳板机
archive: 备忘录
title: 远程连接和 WSL 连接配置（2）：MobaXterm 免密连接+跳板机
date: 2026-10-08
updated: 2026-10-08
tags:
  - post
---
# MobaXterm 免密连接

**场景**：目标服务器 `202.197.4.156` 因特殊网络配置无法直接访问，必须通过跳板机 `47.101.171.14` 中转连接。MobaXterm 的免密连接配置如下：

![](img/remote-mobaxterm-1.png)
![](img/remote-mobaxterm-2.png)
