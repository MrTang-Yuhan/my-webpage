---
layout: post.njk
post_id: 2026-09-29-vllm-服务器部署-qwen3-8-27b-fp8-本地-harness-使用
archive: 备忘录
title: vLLM 服务器部署 Qwen3.8-27B-FP8，本地 Harness 使用
date: 2026-09-29
updated: 2026-10-08
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


**测试**：

另开本地终端执行以下命令，应该有 json 格式的输出：

```cmd
curl http://127.0.0.1:8000/v1/models
```

---

# Cindy Harness 配置

在本地 `Cindy Harness` 中添加模型。由于 SSH 隧道已建立，程序通常会自动识别模型，直接选择即可。

![Cindy Harness 配置](img/cindy-1.png)

![Cindy Harness 配置](img/cindy-2.png)

---

# 在 WSL 中使用

完成上述配置（即已执行 `ssh -N -L 8000:127.0.0.1:8000 tyh@202.197.4.150` 建立转发）后，进入 WSL 执行：

```bash
curl http://127.0.0.1:8000/v1/models
```

此时会发现请求一直没有响应。

这通常是因为配置了代理环境变量（为了使用梯子）：

```bash
http_proxy=http://127.0.0.1:7897    # 以及 HTTP_PROXY / https_proxy / HTTPS_PROXY 等
```

`curl` 会优先遵循这些变量，把请求转发给 `127.0.0.1:7897` 上的本地代理。但 WSL2 有独立的网络命名空间，WSL 内部的 `127.0.0.1` 并不是 Windows 的 `127.0.0.1`——梯子跑在 Windows 侧，在 WSL 里根本连不上，于是请求挂起、无响应。

解决方法：

```bash
# 方式 1：单次绕过代理
curl --noproxy '*' http://127.0.0.1:8000/v1/models

# 方式 2：改用 localhost（若 no_proxy 中包含 localhost，会精确命中并绕过代理）
curl http://localhost:8000/v1/models
```


