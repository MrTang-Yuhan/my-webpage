---
layout: post.njk
post_id: 2026-05-22-软件调优-一-numa-绑定
archive: 训练优化
title: 软件调优（一）：NUMA 绑定
date: 2026-05-22
description: NUMA 亲和性和 NUMA 绑定
tags:
  - post
---
# 什么是 NUMA 亲和性

## 为什么要绑定 NUMA

如果没有绑定 NUMA，GPU 进程可以使用任意的 CPU 核心和任意 NUMA 节点的内存。
绑定 NUMA 的核心作用：让 GPU 进程使用与它物理上最近的 CPU 和内存，减少数据传输延迟，提升训练吞吐量。




# 环境配置

## 安装工具
```bash
# 安装控制 NUMA 内存和 CPU 亲和性的命令行工具
apt install numactl


# 安装 NVIDIA Management Library (NVML) 的 Python 库
pip install nvidia-ml-py

```

## 显示系统 NUMA 硬件拓扑信息

```bash
# 显示系统 NUMA 硬件拓扑信息
numactl --hardware
```
打印的信息含义如下：
| 字段                   | 含义                  |
| -------------------- | ------------------- |
| `available: N nodes` | NUMA 节点总数           |
| `node X cpus`        | 该节点包含的 CPU 核心编号     |
| `node X size`        | 该节点管理的物理内存总量        |
| `node X free`        | 该节点当前空闲内存           |
| `node distances`     | 跨节点内存访问延迟相对值，数值越大越慢 |

- **如果当前的 NUMA 节点为 1，则不需要 NUMA 绑定。只有多节点才需要进行 NUMA 绑定。**


# 实施 NUMA 绑定

# 脚本 + 启动器配置使用

`numa-set.sh` 脚本
```bash
#!/usr/bin/bash

# 这个辅助工具执行 NUMA 节点绑定，可以与 torchrun 及其他启动器配合使用
# 由 https://github.com/yifuwang 贡献

# 1. 首先赋予执行权限：
#
# chmod a+x ./numa-set.sh
#
# 2. 启动 torchrun 并测试是否正确分配了核心
#
# torchrun --nproc_per_node=8 --no-python ./numa-set.sh \
# python -c 'import os; cs=os.sched_getaffinity(0); print(f"{len(cs)} visible cpu cores: {cs}")'
#
# 所以如果你的原始 torchrun 启动命令是：
#
# torchrun --nproc_per_node=8 --nnodes 2 ... train.py
#
# 现在它变成：
#
# torchrun --nproc_per_node=8 --nnodes 2 ... --no-python ./numa-set.sh python train.py
#
# 命令中的 ... 是指省略了一些参数设置
# 为什么需要 --no-python？ 因为 torchrun 默认行为是自动在命令前加上 python，把后面的参数当作 Python 脚本的参数
# 但此处我们执行的是 shell 脚本

# 查询设备 LOCAL_RANK 的 PCIe 总线 ID
BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id -i $LOCAL_RANK --format=csv,noheader)        # LOCAL_RANK 是 torchrun 分配的进程编号（0,1,2,3...）
BUS_ID=${BUS_ID,,}                                                                      # 把变量内容全部小写

# 查找设备 LOCAL_RANK 所在的 NUMA 节点
NODE=$(cat /sys/bus/pci/devices/${BUS_ID:4}/numa_node)                                  # ${BUS_ID:4}: 从索引 4 开始截取

# 用 numactl 包裹后面的命令，在启动前先设置进程的 NUMA 绑定，然后再执行该命令
echo "Starting local rank $RANK on NUMA node $NODE"
numactl --cpunodebind=$NODE --membind=$NODE "$@"                                        # "$@"：用户传入的实际命令（如 python train.py）及所有参数
```

- 按照 `numa-set.sh` 脚本的注释说明，使用脚本 + torchrun 等启动器执行 NUMA 节点绑定。
- 在运行脚本中的步骤 2 “启动 torchrun 并测试是否正确分配了核心” 时，如果出现了报错，那么使用命令 `numactl --hardware` 检查当前是否为单节点，或者当前环境没有配置成功 NUMA。


