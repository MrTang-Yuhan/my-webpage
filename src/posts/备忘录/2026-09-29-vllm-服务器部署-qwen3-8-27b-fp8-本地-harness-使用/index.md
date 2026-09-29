---
layout: post.njk
post_id: 2026-09-29-vllm-服务器部署-qwen3-8-27b-fp8-本地-harness-使用
archive: 备忘录
title: vLLM 服务器部署 Qwen3.8-27B-FP8，本地 Harness 使用
date: 2026-09-29
updated: 2026-09-29
tags:
  - post
---
# 服务器部署完成

需要提前在服务器上完成 `Qwen3.8-27B-FP8` 大模型的部署（本文使用 vLLM，部署过程略）。部署完成后，`CC-switch` 的配置参考如下：

![CC-switch 配置](img/cc-switch.png)

---

# 本机与服务器建立隧道

在本地终端执行以下命令，建立本机到服务器的 SSH 隧道：

```cmd
ssh -N -L 8000:127.0.0.1:8000 tyh@202.197.4.150
```

**命令解析：**

*   `-N`：表示不执行远程命令，仅做端口转发。
*   `-L 8000:127.0.0.1:8000`：将本地的 `8000` 端口转发到服务器的 `127.0.0.1:8000`。
*   `-p SSH端口`：你的 SSH 服务端口（默认 22，上面命令中省略即使用默认端口）。
*   `用户名@服务器IP`：你的 SSH 登录信息（如 `tyh@202.197.4.150`）。

> ⚠️ **注意：命令执行后请保持该窗口运行，不要关闭（可最小化到后台）**，否则隧道将断开。

---

# Cindy Harness 配置

在本地 `Cindy Harness` 中添加模型。由于 SSH 隧道已建立，程序通常会自动识别模型，直接选择即可。

![Cindy Harness 配置](img/cindy-1.png)

![Cindy Harness 配置](img/cindy-2.png)
