---
layout: post.njk
archive: 备忘录
title: VSCode SSH 免密远程连接配置
date: 2026-05-18
description: 在 Windows 系统下配置 SSH 免密登录远程服务器，并且支持多用户配置。
tags:
  - post
---
# VSCode 安装插件

![](img/remote-ssh-plugin.png)

# 如何免密连接

在本机（即 Windows 系统）中生成 SSH 公钥文件和私钥文件：

```bash

# 方法1
生成 ed25519 格式公钥和私钥文件
ssh-keygen

# 方法2



```

# config 文件

## 参考格式

```
Host 任意用户名
  HostName 主机的IP地址/域名
  User 主机上的用户名
  Port 端口号
  IdentityFile 私钥文件
```

举例：

```bash
Host yiyang
  HostName 10.7.124.11
  User yiyang
  Port 22
  IdentityFile C:\Users\i26298\.ssh\id_ed25519

Host yuhan
  HostName 10.7.124.11
  User i26298
  Port 22
  IdentityFile C:\Users\i26298\.ssh\id_ed25519
```
