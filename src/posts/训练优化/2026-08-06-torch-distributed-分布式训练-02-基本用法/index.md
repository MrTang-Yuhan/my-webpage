---
layout: post.njk
post_id: 2026-08-06-torch-distributed-分布式训练-02-基本用法
archive: 训练优化
title: torch.distributed 分布式训练（02）：基本用法
date: 2026-08-06
tags:
  - post
---

# 1. `torchrun` 启动参数

## 单节点训练

如果机器上有 8 张 GPU，并且希望每张 GPU 使用一个进程，可以执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --standalone --nproc_per_node=8 train.py
```

也可以省略 `CUDA_VISIBLE_DEVICES`，直接使用所有可见 GPU：

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

参数说明：

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`：限制当前任务可见的物理 GPU。程序内部会将这些 GPU 重新编号为逻辑设备 `cuda:0` 到 `cuda:7`。
- `--standalone`：使用本机自动创建 rendezvous（进程集合点），适用于单节点训练。
- `--nproc_per_node=8`：当前节点启动 8 个 Python 进程，通常对应 8 张 GPU，每个进程负责一张 GPU。

如果想要**单卡多进程来检测通信逻辑是否正确**，则可以：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=8 train.py
```

虽然可以启动 8 个进程，但只有一张 GPU 对这些进程可见。注意此时要 `torch.cuda.set_device(0)`。


如果代码按照 `LOCAL_RANK` 为每个进程分配不同 GPU，例如：

```python
torch.cuda.set_device(local_rank)
```

那么 `local_rank` 大于 0 的进程通常会因为找不到对应 GPU 而失败。因此，GPU 训练时通常应满足：

```text
nproc_per_node <= 当前进程可见的 GPU 数量
```

---

## 多节点训练

假设使用 2 个节点，每个节点有 4 张 GPU：

- 主节点 IP：`192.168.1.10`
- 每个节点启动 4 个进程
- 节点编号从 0 开始

### 节点 0

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=192.168.1.10 \
    --master_port=29500 \
    train.py
```

### 节点 1

```bash
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=192.168.1.10 \
    --master_port=29500 \
    train.py
```

参数说明：

- `--nnodes=2`：参与训练的节点总数为 2。
- `--nproc_per_node=4`：每个节点启动 4 个 Python 进程。
- `--node_rank`：当前节点编号，取值范围为 `0` 到 `nnodes - 1`。
- `--master_addr`：主节点的可访问 IP 地址。
- `--master_port`：主节点用于 rendezvous 的端口。该端口必须在节点间可访问，并且不能被其他任务占用。

在多节点训练中，通常每个进程对应一张 GPU，因此总进程数为：

```text
world_size = nnodes × nproc_per_node = 2 × 4 = 8
```

主节点 IP 可以通过以下命令查看：

```bash
hostname -I
```

启动训练后，可以在其他节点测试端口连通性：

```bash
nc -zv 192.168.1.10 29500
```

需要注意，只有当主节点上的 `torchrun` 已经开始监听该端口时，测试才会成功。

> `torchrun` 支持弹性训练。节点故障或节点数量变化时，torchrun 可以根据配置重新组织进程，但训练代码通常需要处理 checkpoint 恢复、进程重启以及数据划分变化等问题。

---

# 2. 分布式训练中的日志

在分布式训练中，多个进程同时进入 IDE 调试器，可能因为进程间通信和同步等待而造成阻塞。因此，实际调试时通常使用日志或带 rank 信息的输出。

下面介绍三种常用方式：

1. 使用 `logging` 同时输出到文件和控制台；
2. 只让全局 rank 0 打印；
3. 让所有进程打印，并附带 rank、local rank 和进程号。

## 2.1 为每个 rank 配置文件和控制台日志

```python

import logging


def setup_logging(filemode: str = "a") -> None:
  r"""为当前分布式 rank 配置文件和控制台日志。

  必须在 ``dist.init_process_group()`` 成功后调用，因其依赖全局 rank。

  Args:
    filemode: 日志文件打开模式。``"a"`` 追加到已有日志，``"w"`` 在每次
      启动时覆盖对应 rank 的旧日志。

  Raises:
    ValueError: ``filemode`` 不是 ``"a"`` 或 ``"w"``。
  """
  if filemode not in ("a", "w"):
    raise ValueError("filemode must be either 'a' (append) or 'w' (overwrite).")

  rank = dist.get_rank()

  # 每个 rank 使用独立文件，避免多进程同时写同一日志文件。
  logging.basicConfig(
      filename=f"rank_{rank}.log",
      filemode=filemode,
      level=logging.INFO,
      format="%(asctime)s - %(levelname)s - %(message)s",
  )

  # 同时输出到控制台，便于观察各 rank 的实时状态。
  # 注释下面这段代码就只会打印到日志文件
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s - Rank %(rank)s - %(message)s")
  console.setFormatter(formatter)
  logging.getLogger("").addHandler(console)

  # 为所有日志记录注入当前 rank，供控制台格式化字符串使用。
  old_factory = logging.getLogRecordFactory()

  def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    record.rank = rank
    return record

  logging.setLogRecordFactory(record_factory)
```

使用方法：

```python
dist.init_process_group()
setup_logging()

logging.info("training started")
```

每个全局 rank 使用独立的日志文件，例如：

```text
rank_0.log
rank_1.log
rank_2.log
```

如果多个任务在同一个目录中运行，建议将日志放入带有任务 ID 的独立目录中，避免不同任务相互覆盖。

## 2.2 只由全局 rank 0 打印

```python
def rank_zero_log(message: str) -> None:
  r"""仅由全局 rank 0 打印消息。"""
  if dist.get_rank() == 0:
    print(message, flush=True)
```

这种方式适合打印只需要显示一次的信息，例如数据集大小、模型结构或最终评估结果。

## 2.3 所有进程打印，并附带进程信息

```python
def rank_log(message: str) -> None:
    """由所有进程打印带有 rank 信息的消息。"""
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"[rank={rank}, local_rank={local_rank}, pid={os.getpid()}] {message}",
          flush=True)
```

其中：

- `dist.get_rank()`：进程的全局 rank；
- `LOCAL_RANK`：进程在当前节点内的编号，通常用于选择 GPU；
- `WORLD_SIZE`：所有节点上的进程总数；
- `pid`：当前操作系统进程号。

`dist.get_rank()` 只有在 `dist.init_process_group()` 成功后才能调用；如果需要在初始化之前打印，应读取 `RANK` 环境变量或使用默认值。

---

# 3. `dist` 分布式操作流程

分布式通信通常包括三个阶段：

1. 初始化默认进程组；
2. 执行集体通信操作；
3. 任务结束后销毁进程组。

```python
import torch.distributed as dist

# 1. 初始化默认进程组
dist.init_process_group(backend="nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()

# 2. 默认进程组上的集体通信
dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
dist.broadcast(tensor, src=0)
dist.all_gather(tensor_list, tensor)
dist.barrier()
```

所有参与通信的进程必须以相同顺序调用对应的 collective，否则可能导致阻塞或死锁。

## 创建子通信组

可以使用 `dist.new_group()` 将部分进程划分到独立的通信组中。例如，将 4 个进程划分为两个子组：

```python
group_01 = dist.new_group(ranks=[0, 1])
group_23 = dist.new_group(ranks=[2, 3])

if rank in (0, 1):
    dist.all_reduce(tensor, group=group_01)
elif rank in (2, 3):
    dist.all_reduce(tensor, group=group_23)
```

注意：

- 所有进程都必须调用 `dist.new_group()`；
- 各进程创建通信组的顺序必须一致；
- 只有通信组成员可以在该组上执行集体通信；
- 使用 NCCL 时，通信张量必须位于对应的 CUDA 设备上。

```python
# 3. 销毁进程组
dist.destroy_process_group()
```

如需显式释放子通信组，也可以调用：

```python
dist.destroy_process_group(group_01)
dist.destroy_process_group(group_23)
dist.destroy_process_group()
```







---

# 待完善

- torchrun启动参数

- 日志的使用

- dist分布式整体流程
- dist.dist.new_group() 子通信组可以直接组内通信
- dist.init_process_group() 默认组包含所有进程

- apply的用法，以及里面的ctx。apply 是自定义 torch.autograd.Function 的标准调用方式

- tp,pp,dp三种的结果图

- dp中functional.mse_loss(...)，其默认行为是 reduction='mean'，导致梯度必须要缩放


设总样本集为 $\mathcal{B} = \{ (x_i, y_i) \}_{i=1}^{N}$，模型参数为 $\mathbf{W}$，样本损失为 $\ell_i(\mathbf{W}) = \ell(f(x_i; \mathbf{W}), y_i)$。

代码中 `functional.mse_loss` 默认使用 `reduction='mean'`，因此**全局目标函数**为：

$$
L(\mathbf{W}) = \frac{1}{N} \sum_{i=1}^{N} \ell_i(\mathbf{W})
$$

其梯度为基准真值：

$$
\mathbf{g}^\star = \nabla L(\mathbf{W}) = \frac{1}{N} \sum_{i=1}^{N} \nabla \ell_i(\mathbf{W}) \tag{1}
$$

### 1. 单卡梯度累加（Gradient Accumulation）

将 $\mathcal{B}$ 拆分为 $M$ 个 micro-batch，每份大小 $B = N/M$。对第 $m$ 份 $\mathcal{B}_m$ 计算局部损失（mean reduction）：

$$
L_m(\mathbf{W}) = \frac{1}{|\mathcal{B}_m|} \sum_{i \in \mathcal{B}_m} \ell_i(\mathbf{W}) = \frac{M}{N} \sum_{i \in \mathcal{B}_m} \ell_i(\mathbf{W})
$$

反向传播得到局部梯度：

$$
\mathbf{g}_m = \nabla L_m(\mathbf{W}) = \frac{M}{N} \sum_{i \in \mathcal{B}_m} \nabla \ell_i(\mathbf{W})
$$

累加 $M$ 个 micro-batch 的梯度：

$$
\sum_{m=1}^{M} \mathbf{g}_m = \sum_{m=1}^{M} \frac{M}{N} \sum_{i \in \mathcal{B}_m} \nabla \ell_i(\mathbf{W}) = \frac{M}{N} \sum_{i=1}^{N} \nabla \ell_i(\mathbf{W}) = M \cdot \mathbf{g}^\star \tag{2}
$$

**结论**：累加后的梯度是基准真值的 $M$ 倍。必须执行 **缩放**：

$$
\mathbf{g}_{\text{acc}} = \frac{1}{M} \sum_{m=1}^{M} \mathbf{g}_m = \mathbf{g}^\star
$$

---

### 2. 多卡数据并行（Data Parallel）

将 $\mathcal{B}$ 拆分为 $D$ 份（$D =$ DP size），卡 $k$ 持有 $\mathcal{B}_k$，$|\mathcal{B}_k| = N/D$。卡 $k$ 的局部损失：

$$
L_k(\mathbf{W}) = \frac{1}{|\mathcal{B}_k|} \sum_{i \in \mathcal{B}_k} \ell_i(\mathbf{W}) = \frac{D}{N} \sum_{i \in \mathcal{B}_k} \ell_i(\mathbf{W})
$$

局部梯度：

$$
\mathbf{g}_k = \nabla L_k(\mathbf{W}) = \frac{D}{N} \sum_{i \in \mathcal{B}_k} \nabla \ell_i(\mathbf{W})
$$

执行 `all_reduce` 求和（SUM）：

$$
\mathbf{g}_{\text{sum}} = \sum_{k=1}^{D} \mathbf{g}_k = \sum_{k=1}^{D} \frac{D}{N} \sum_{i \in \mathcal{B}_k} \nabla \ell_i(\mathbf{W}) = \frac{D}{N} \sum_{i=1}^{N} \nabla \ell_i(\mathbf{W}) = D \cdot \mathbf{g}^\star \tag{3}
$$

**结论**：all-reduce SUM 后的梯度是基准真值的 $D$ 倍。必须执行 **平均**：

$$
\mathbf{g}_{\text{dp}} = \frac{1}{D} \sum_{k=1}^{D} \mathbf{g}_k = \mathbf{g}^\star
$$

