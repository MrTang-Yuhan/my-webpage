---
layout: post.njk
archive: GPU逆向工程
title: GPU 内存子系统分析--延迟分析
date: 2026-05-15
description: 分析GPU 的 L1/Shared Memory，L2，DRAM 的访问延迟
tags:
  - post
---
# 延迟的定义

延迟可以分为两种：

## True Latency （真实依赖延迟）

构造一串前后依赖的指令，让后一条指令必须等前一条指令的结果出来才能执行。

True Latency 反映在无法并行处理情况下的延迟。

  例如：
```assembly
FFMA R0, R0, R1, R2
FFMA R0, R0, R1, R2
FFMA R0, R0, R1, R2
FFMA R0, R0, R1, R2
```
每条指令都读写 R0，所以第 2 条必须等第 1 条的 R0 结果可用，第 3 条必须等第 2 条，以此类推。

这种情况下，硬件不能通过流水线重叠来隐藏延迟。

## Completion Latency (完成延迟 / 平均完成间隔)

构造一组互相独立的指令，它们之间没有数据依赖，所以硬件可以让它们并行、流水线化、重叠执行。

汇编层面类似：

```assembly
FFMA R0,  R0,  R20, R21
FFMA R1,  R1,  R20, R21
FFMA R2,  R2,  R20, R21
FFMA R3,  R3,  R20, R21
FFMA R4,  R4,  R20, R21
FFMA R5,  R5,  R20, R21
```

这些指令分别操作不同寄存器 R0, R1, R2, ...，所以彼此不需要等待。

假设单条 FFMA 的 true latency 是 10 cycles，但执行单元是流水线化的，可以每个 cycle 接收一条新指令。此时虽然每条指令从发射到结果可用仍然是 10 cycles，但因为流水线重叠了，所以从整体看，平均每条指令的完成间隔可能接近 1 cycle/instruction。

# 内存子系统的 True Latency

## L1，L2 和 DRAM 

我使用 pointer-chasing 方法测量 L1、L2 和 DRAM 的真实延迟（true latency）。

测试平台为 NVIDIA 5080 GPU，Compute Capability 为 sm_120。

测量前请注意以下事项：

1. **锁定 GPU 和显存频率**
   - 使用 `nvidia-smi -lgc <gpu_clocks>` 锁定 GPU 频率
   - 使用 `nvidia-smi -lmc <mem_clocks>` 锁定显存频率
   - 请使用脚本或者命令 `nvidia-smi dmon` 查看当前 GPU 频率和显存频率[^1]
   - [锁定 GPU 和显存频率的脚本](#lock_gpu_mem_clock)

[^1]: 使用 `nvidia-smi` 命令配置的频率不一定是实际运行时的频率。当 CUDA 应用启动后，已配置的频率仍可能动态变化。因此，每次运行 CUDA 程序时，都应在程序执行期间再次检查当前设备频率以进行确认。

2. **避免使用共享内存（Shared Memory）**
   - 在代码中确保不通过 `__shared__` 分配共享内存
   - 通过 CUDA 编程接口，尽可能将统一的 L1 / 共享内存空间分配给 L1 缓存：

   ```cuda
   // 尽量把统一空间划分给 L1 cache
   cudaFuncSetAttribute(
       pchase_latency_kernel,
       cudaFuncAttributePreferredSharedMemoryCarveout,
       cudaSharedmemCarveoutMaxL1
   );
   ```

3. ** PTX 运算符决定是否绕过 L1 访问**
   在 CUDA 代码中内嵌 PTX 指令，可以指定访存操作是否绕过 L1 缓存。
   - `.ca`：允许在各级缓存（L1 和 L2）中进行缓存[^2]
   - `.cg`：仅在 L2 及更低层级缓存中分配，绕过 L1[^3]

[^2]: 各级缓存中的数据，可能会被再次访问。<br>
默认的加载指令缓存操作为 ld.ca，它会在所有缓存层级（L1 和 L2）中分配缓存行，并采用常规的逐出策略。全局数据在 L2 层级上是一致的，但对于全局数据而言，多个 L1 缓存之间并不保持一致。如果一个线程通过其 L1 缓存向全局内存写入数据，而另一个线程通过另一个 L1 缓存使用 ld.ca 指令读取该地址的数据，那么第二个线程可能会读到过时的 L1 缓存数据，而不是第一个线程写入的数据。因此，驱动程序必须在相互依赖的并行线程网格之间，对全局数据的 L1 缓存行进行作废操作。这样，第一个网格程序写入的数据，就能够被第二个网格程序通过默认的 ld.ca 加载指令（该指令会将数据缓存在 L1 中）正确地获取到。

[^3]: 全局层级的缓存（仅在 L2 及更低层级缓存，不包含 L1）。<br>
可以使用 ld.cg 指令，使加载操作仅在全局层级进行缓存，即绕过 L1 缓存，仅将数据缓存在 L2 缓存中。





































# 相关资料

[CUDA 官方编程手册](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)

[CUDA 官方 PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)


# 脚本


<a id="lock_gpu_mem_clock"></a>
## 锁定 GPU 和显存频率的脚本

`gpu_clock_lock.sh`
```shell
#!/usr/bin/env bash
set -euo pipefail

GPU_ID=0
SM_CLOCK=""
MEM_CLOCK=""
ACTION=""

usage() {
    cat <<EOF
Usage:
  $0 status [--gpu GPU_ID]
  $0 lock   --sm SM_MHZ [--mem MEM_MHZ] [--gpu GPU_ID]
  $0 unlock [--gpu GPU_ID]

Examples:
  $0 status
  sudo $0 lock --sm 2500
  sudo $0 lock --sm 2500 --mem 14000
  sudo $0 unlock

Notes:
  --sm  : lock SM/core clock in MHz, e.g. 2500
  --mem : lock memory clock in MHz, e.g. 14000
  --gpu : GPU index, default 0
EOF
}

need_nvidia_smi() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not found"
        exit 1
    fi
}

need_root_for_modify() {
    if [[ "${ACTION}" == "lock" || "${ACTION}" == "unlock" ]]; then
        if [[ "${EUID}" -ne 0 ]]; then
            echo "ERROR: lock/unlock requires root. Please run with sudo."
            exit 1
        fi
    fi
}

show_status() {
    echo "==== GPU status ===="
    nvidia-smi -i "${GPU_ID}" \
        --query-gpu=index,name,pstate,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.current.graphics,clocks.current.memory,clocks.max.sm,clocks.max.graphics,clocks.max.memory \
        --format=csv

    echo
    echo "==== nvidia-smi full summary ===="
    nvidia-smi -i "${GPU_ID}"
}

lock_clocks() {
    echo "==== Enable persistence mode ===="
    nvidia-smi -i "${GPU_ID}" -pm 1 || true
    echo

    if [[ -n "${SM_CLOCK}" ]]; then
        echo "==== Lock SM/core clock to ${SM_CLOCK} MHz ===="
        nvidia-smi -i "${GPU_ID}" -lgc "${SM_CLOCK},${SM_CLOCK}"
        echo
    else
        echo "ERROR: --sm SM_MHZ is required for lock"
        exit 1
    fi

    if [[ -n "${MEM_CLOCK}" ]]; then
        echo "==== Try to lock memory clock to ${MEM_CLOCK} MHz ===="
        if nvidia-smi -i "${GPU_ID}" -lmc "${MEM_CLOCK},${MEM_CLOCK}"; then
            echo "Memory clock locked to ${MEM_CLOCK} MHz"
        else
            echo "WARNING: Failed to lock memory clock."
            echo "This is common on some GeForce GPUs/drivers."
        fi
        echo
    fi

    show_status
}

unlock_clocks() {
    echo "==== Reset SM/core clock lock ===="
    nvidia-smi -i "${GPU_ID}" -rgc || true
    echo

    echo "==== Reset memory clock lock ===="
    nvidia-smi -i "${GPU_ID}" -rmc || true
    echo

    echo "==== Keep persistence mode enabled ===="
    nvidia-smi -i "${GPU_ID}" -pm 1 || true
    echo

    show_status
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

ACTION="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --sm)
            SM_CLOCK="$2"
            shift 2
            ;;
        --mem)
            MEM_CLOCK="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            exit 1
            ;;
    esac
done

need_nvidia_smi
need_root_for_modify

case "${ACTION}" in
    status)
        show_status
        ;;
    lock)
        lock_clocks
        ;;
    unlock)
        unlock_clocks
        ;;
    *)
        echo "Unknown action: ${ACTION}"
        usage
        exit 1
        ;;
esac
```




