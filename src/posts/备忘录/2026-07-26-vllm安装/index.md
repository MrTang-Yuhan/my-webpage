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

> **宿主机环境**：
>
> - **CUDA Version: 13.1**
> - **GPU**: NVIDIA 4070 Super
>
> **Docker镜像环境**：
> - **nvcc 版本：12.8**
> 
>
> **安装参考**：[vLLM 中文手册](https://docs.vllm.com.cn/en/latest/contributing/#job-board)
> 
> **环境依赖链条**：
>
> - **第 1 层：GPU 硬件**
>
> 每块 GPU 有固定的架构（如 Pascal、Ampere、Blackwell）和计算能力（sm_xx）。它决定了驱动"能不能识别你这块卡"——太老的驱动不认识新卡，太新的驱动会主动砍掉对旧卡的支持。
>
> - **第 2 层：NVIDIA 驱动（宿主机唯一必需）**
> 
> 宿主机上**只需要装驱动 (如 535.54 / 570.86 / 580.12)**，不需要装 CUDA Toolkit。当你用 `docker run --gpus all` 启动容器时，`nvidia-container-toolkit` 会自动把驱动的用户态库（如 `libcuda.so`）和设备文件挂载进容器，让容器里的程序能直接调用 GPU。
> 
> - **第 3 层：驱动支持的最高 CUDA 版本**
> 
> 在宿主机执行 `nvidia-smi`，右上角显示的 **"CUDA Version"** 就是这块驱动能支持的最高 CUDA 版本。驱动向后兼容，所以容器里跑更低版本的 CUDA 也没问题。
>
> - **第 4 层：容器内的 CUDA Toolkit（可选）**
> 
> 只有当你需要在容器里**编译 CUDA 程序**（比如 `pip install` 某些需要 `nvcc` 的包）时才需要装。它只是个编译工具链，单纯跑 PyTorch 训练/推理时，容器里完全可以没有 CUDA Toolkit。其版本同样不能超过驱动支持的上限。
> 
> - **第 5 层：PyTorch / TensorFlow 等框架**
>
> 官方发布的 wheel 包（如 `torch==2.7.0+cu128`）**已经自带了 CUDA 运行时库**，不依赖容器里有没有装 CUDA Toolkit。唯一的硬性约束是：**框架自带的 CUDA 版本 ≤ 驱动支持的最高 CUDA 版本**。


## 开发 vLLM 的 Python 和 CUDA/C++ 代码

如果目标是开发 **vLLM 的 Python 和 CUDA/C++ 代码**，请按以下步骤操作。

### 1. 安装 uv

> 安装参考： [uv 文档](https://uv.doczh.com/getting-started/installation/#docker)

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

由于 CUDA 版本为 12.8，所以要安装对应版本的 Pytorch，这里验证 torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 测试通过。

```bash
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

注意下面的没有指定版本的做法，会**默认安装最新的 torch, torchvision 和 torchaudio，导致后续依赖安装出错**：

```bash
uv pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
```

### 4. 克隆 vLLM 仓库

注意**切换到 vLLM 的 v0.12.0 版本**，支持 Pytorch 2.9.0+cu128 和 CUDA 12.8。

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.12.0
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

**注意：** `pip install -e .`（editable 模式）的本质是在 Python 的 site-packages 目录下创建一个指向你源码目录的链接（.pth 文件或 .egg-link）。Python 解释器导入 vllm 包时，会直接读取你源码目录下的 .py 文件，而不是复制到 site-packages 里的副本。所以，**使用 `uv pip install -e . --no-build-isolation` 命令安装时，改 python 代码也无需再重新安装。**

### 7. 测试修改 vLLM 的 Python 代码无需重新编译安装

验证方法：

```bash
# 1. 找到 vllm 包的实际位置
python -c "import vllm; print(vllm.__file__)"
# 输出应该是类似 /workspace/vllm/vllm/__init__.py
# 而不是 /root/.cache/uv/... 或 site-packages 下的路径

# 2. 修改任意 .py 文件，加一行 print
echo 'print("THIS_IS_MY_EDIT")' >> vllm/__init__.py

# 3. 直接 python 导入，立刻看到输出
python -c "import vllm"
# 你会看到 THIS_IS_MY_EDIT，证明读取的是源码目录的最新内容
```


### 8. 修改 vLLM 的 CUDA/C++ 代码，但是使用增量编译

在开发位于 csrc/ 目录下的 vLLM C++/CUDA 核函数（kernels）时，每次更改都使用 `uv pip install -e .` 重新编译整个项目会非常耗时。使用 CMake 的增量编译工作流允许在初始设置后仅重新编译必要的组件，从而实现更快的迭代。在 vllm 根目录下进行如下操作：

```bash
# 1. 先完整编译一次（不用预编译 wheel）
uv pip install -e . --no-build-isolation

# 2. 之后所有 C++ 修改都用 CMake 增量编译
## 生成 CMake 配置文件
python tools/generate_cmake_presets.py
## 初始化 CMake 构建环境首次构建并安装
cmake --preset release
## 首次构建并安装
cmake --build --preset release --target install
```

**后续如果修改 CUDA/C++ 代码，只需要重新运行下面的命令进行增量编译：**

```bash
cmake --build --preset release --target install
```


> 参考：[vLLM 增量编译工作流](https://docs.vllm.com.cn/en/latest/contributing/#developing)。

### 9. vLLM 代码库自测

vLLM 使用 pytest 测试代码库。

安装测试依赖：

```bash
# CUDA 平台完整测试依赖
uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

# 最小测试依赖（通用）
uv pip install pytest pytest-asyncio
```

运行测试：

```bash
# 全量测试
pytest tests/

# 单个文件详细输出
pytest -s -v tests/test_logger.py
```


### 10. 代码检查

vLLM 使用 pre-commit 对代码库进行 linting 和格式化。如果您不熟悉 pre-commit，请参阅 [pre-commit 手册](https://pre-commit.git-scm.cn/#usage)。设置 pre-commit 就像这样简单：

```bash
uv pip install "pre-commit>=4.5.1"
pre-commit install
```
vLLM 的 pre-commit 钩子现在将在您每次 `git commit` 提交时自动运行。

当然也可以手动进行代码自查：

```
# 改代码前的自查（养成习惯）
pre-commit run -a
```



