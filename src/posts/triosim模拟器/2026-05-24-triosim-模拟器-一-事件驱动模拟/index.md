---
layout: post.njk
post_id: 2026-05-24-triosim-模拟器-一-事件驱动模拟
archive: triosim模拟器
title: " TrioSim 模拟器 （一）：事件驱动模拟"
date: 2026-05-24
tags:
  - post
---
# 事件驱动模拟器（Event-Driven Simulator）

## 事件驱动模拟器简介

事件驱动模拟器是只有当某个事件发生时，模拟器才处理它，并把模拟时间跳到下一个事件的发生时间。一般它的组成是：

1. 维护一个**按时间排序的事件队列**，
2. 每次取出**最早发生的事件**执行，
3. 执行事件时可能产生新的未来事件，
4. 然后继续处理下一个事件。

在事件驱动模拟器中，“事件”被定义为“在某个模拟时间点要发生的事情”。例如：

- 时间 10ns：CPU 发出一个内存请求
- 时间 15ns：请求到达 L2 Cache
- 时间 30ns：DRAM 返回数据
- 时间 35ns：CPU 收到响应

这里每一项都是事件。

## 事件驱动模拟器的优点

- 由于事件驱动模拟器仅在事件发生时推进模拟过程，而无需对每个周期进行精确建模，因此其**仿真效率通常显著高于周期精确模拟器**。

- 事件驱动模拟器通常采用事件或消息作为模块间的通信机制，这种设计有效**降低了模块之间的耦合度**，并提升了模拟器的可维护性与可扩展性。

- 事件驱动模拟器更适合用于模拟异步系统，能够**自然地描述事件触发、消息传递、请求响应等行为**。

# TrioSim 模拟器

Triosim 是一个基于事件驱动的深度学习训练/推理模拟器，用于评估不同并行策略（数据并行、张量并行、流水线并行）的性能表现[^1]。它自带以下 6 个 case：

[^1]: 尽管 TrioSim 模拟器用于建模基于 GPU 的训练与推理过程，但它并不对 GPU 微架构进行模拟。相反，TrioSim 将每个 GPU 抽象为一个设备节点，从系统层面描述 GPU 间的执行与交互过程，而不涉及 GPU 内部计算单元、缓存层次、调度器等架构细节；其模拟主要依赖 GPU 相关的 trace 输入。

- Case 0: 纯推理回放，不做梯度同步（无 AllReduce）
- Case 1: 推理 + Ring AllReduce
- Case 2: 分布式数据并行（Data Parallel）
- Case 3: 张量并行（Tensor Parallel）
- Case 4: 流水线并行（Pipeline Parallel）
- Case 5: HOP（High Bandwidth Optical Path） AllReduce 通信测试

其中，
- Case 0：是纯推理回放（通常只看 forward）
- Case 1~4：是训练相关路径（包含同步/并行训练语义），有 forward + backwrad
- Case 5：对，重点是 HOP AllReduce 通信测试，不是完整训练流程

TrioSim 模拟器是基于底层事件驱动引擎 Akita 构建的。考虑到直接介绍 Akita 引擎较为抽象难懂，我将在介绍 TrioSim 模拟器的建模过程中，对用到的 Akita 引擎知识进行逐步解释。

# C++ 版本模拟器的使用

## 1. Tracer

TrioSim 提供了处理好的示例 trace，可直接使用。这些 trace 位于 `./sample_trace/` 目录下。

如果想快速开始测试：可以跳过 trace 收集步骤（第 1 节），直接前往第 2 节开始模拟。

### 1.1 Trace 收集

#### 使用环境

- Python: 3.10.12
- CUDA: 12.1
- torch: 2.1.0+cu121
- torchvision: 0.16.0+cu121
- torchaudio: 2.1.0+cu121

#### 数据集

代码使用 `ILSVRC2012_img_val` 数据集。为了便于快速上手，在 `./tracer` 目录下提供了一个包含 256 张图像的子集。

#### 使用方法

下面是一个在 batch size 为 16 时收集 trace 的示例命令：

```bash
cd tracer
python datacollect.py 16
```

为了从 PyTorch 模型中收集 trace，我们使用 PyTorch Profiler 来获取各层或各算子的时间信息，并使用 Execution Graph Observer 工具来收集详细的输入、输出以及其他张量或数据信息。

batch size 通过命令行参数设置。你也可以直接在代码中自定义迭代次数（`num_iters`）以及要进行 trace 的模型（`listmodel`）。

这将生成两类文件：

- `tracer/data/graph/graph_xx.json`：包含每个算子的时间信息
- `tracer/data/profiler/profiler_xx.json`：包含详细的张量信息

### 1.2 Trace 数据处理

运行以下命令可将收集到的 trace 转换为 TrioSim 格式：

```bash
# 这个脚本运行预计消耗 1 个小时左右的时间
cd tracer
python dataprocess.py
```

代码文件中，变量 `TARGET_OP_PREFIXES` 允许用户定义哪些层会被纳入张量并行。默认情况下，它包括 `convolution`、`linear` 和 `embedding` 层。


处理后的 trace 文件 `tensor.csv` 和 `trace.csv` 将保存在：

```bash
./tracer/data/middledata/trace/XXmodel
```

## 2. TrioSim

### 2.1 配置

模拟器可通过以下命令行参数进行配置：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `-trace-dir` | string | `"../sample_trace/trace2-h100-bs128/vgg13/"` | 包含 trace 文件的目录 |
| `-batch-size` | int | 128 | 原始 trace 的 batch size |
| `-batch-size-sim` | int | -1 | 模拟时使用的 batch size（默认与 `batch-size` 相同） |
| `-bandwidth` | float | 696 | GPU 到远程内存的带宽（GBps） |
| `-ptp-bandwidth` | float | 65 | GPU 到 GPU 的带宽（GBps） |
| `-GPUnumber` | int | 8 | GPU 数量 |
| `-micro-batch-size` | int | -1 | 用于流水线并行的 micro batch size |
| `-case` | int | 0 | 模拟模式：0=训练，1=标准数据并行，2=分布式数据并行，3=张量并行，4=流水线并行 |
| `-capacity` | int | 40 | 每个设备的内存容量（`1 << capacity`） |
| `-numCols` | int | -1 | 光网络 mesh 中的列数 |
| `-numRows` | int | 1 | 光网络 mesh 中的行数 |
| `-interconnects` | int | 0 | 互连类型：0=电互连，1=光互连 |

### 2.3 运行模拟器

#### 基本用法

1. 进入 `triosim-cpp` 目录：

```bash
cd triosim-cpp
```

2. 创建 `build` 目录和通过 `cmake` 构建 makefile 

```bash
mkdir -p build && cd build
# 启用调试模式
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS_DEBUG="-O0 -g3"
```

3. 编译生成可执行文件

```bash
make -j
```

4. 运行程序

```bash
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu \
./triosim/triosim \
--trace-dir=/home/tang/triosim-cpp/sample_trace/trace2-h100-bs128/vgg13/  \
--batch-size=128 --batch-size-sim=-1 --GPUnumber=16 --case=5 --interconnects=0 --numRows=4 --numCols=4
```
