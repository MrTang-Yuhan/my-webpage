---
layout: post.njk
post_id: 2026-07-26-vllm安装
archive: 备忘录
title: vLLM安装
date: 2026-07-26
tags:
  - post
---
# vLLM 从源代码构建安装

> **环境**：
>
> - **CUDA**: 12.8
> - **GPU**: NVIDIA 4070 Super
>
> **安装参考**：[vLLM 中文手册](https://docs.vllm.com.cn/en/latest/contributing/#job-board)

## 开发 vLLM 的 Python 和 CUDA/C++ 代码

如果目标是开发 **vLLM 的 Python 和 CUDA/C++ 代码**，请按以下步骤操作。

### 1. 安装 uv

安装命令：
```bash
pip install uv
```

特别注意，**安装 uv 一定要使用 pip 而不是 uv 官网的一键安装脚本**，因为脚本会导致后续命令 `uv venv --python 3.12 --seed --managed-python` 失败，因为使用了 `--seed`。

- `--seed`: 在创建虚拟环境时，预装 pip、setuptools、wheel 这三个基础包。
- `--managed-python`: 让 uv 自动下载并管理 Python 解释器，而不是使用系统已安装的 Python。

### 2. 使用 uv 创建 Python 虚拟环境

安装 uv 后，您可以使用以下命令创建新的 Python 环境。

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate
```

### 3. 安装 Pytorch

由于 CUDA 版本为 12.8，所以要安装对应版本的 Pytorch。

```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

### 4. 克隆 vLLM 仓库

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

