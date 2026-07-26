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

### 5. 从 vLLM 仓库安装必要依赖

从 requirements/build/cuda.txt 安装必要的构建依赖，跳过 torch，因为它已经安装.

```bash
grep -v '^torch==' requirements/build/cuda.txt | uv pip install -r -
```

### 6. 安装 vLLM

最后使用以下命令安装 vLLM：

```bash
uv pip install -e . --no-build-isolation
```

### 7. 增量编译 vLLM

在开发位于 csrc/ 目录下的 vLLM C++/CUDA 核函数（kernels）时，每次更改都使用 uv pip install -e . 重新编译整个项目会非常耗时。使用 CMake 的增量编译工作流允许在初始设置后仅重新编译必要的组件，从而实现更快的迭代。具体请参考：[vLLM 增量编译工作流](https://docs.vllm.com.cn/en/latest/contributing/#developing)。


