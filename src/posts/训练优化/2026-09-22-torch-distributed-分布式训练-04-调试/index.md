---
layout: post.njk
post_id: 2026-09-22-torch-distributed-分布式训练-04-调试
archive: 训练优化
title: torch.distributed 分布式训练（04）：调试
date: 2026-09-22
tags:
  - post
---
2 卡及以上不是不能使用断点，但普通 PyCharm 断点不能像单卡一样随意使用。

核心原因是：多个 rank 必须按相同顺序执行 barrier、all_reduce、all_gather 等集合通信。如果 rank 0 被断点暂停，而 rank 1 已经进入集合通信，rank 1 会等待 rank 0，最终表现为卡死或超时。

所以，推荐的调试顺序是：

**单卡 pycharm 断点** -> **2 卡使用日志输出** 

# 2 卡使用日志输出

```python
import os
import torch  # 仅在需要打印张量示例时使用
import torch.distributed as dist


def dlog(message: str) -> None:
    """打印带分布式进程标识的调试日志。

    每条日志自动加上 rank（进程组内编号）和 pid（操作系统进程号），
    便于在多进程 / 多 GPU 场景下定位日志来源；输出立即刷新（flush），
    避免多进程日志交错或进程崩溃时丢失内容。

    Args:
        message: 待打印的日志文本。打印变量/张量的推荐写法::

            x = 42
            dlog(f"x = {x}")                        # 普通变量：直接 f-string 拼接

            t = torch.randn(4, device="cuda")
            dlog(f"t = {t.cpu()}")                  # 整个张量：先 .cpu() 挪回内存再打印
            dlog(f"t[0] = {t[0].item():.4f}")       # 单个标量：.item()，触发一次 GPU→CPU 同步
            dlog(f"t.shape = {tuple(t.shape)}")     # 只看形状/类型：无 GPU 同步开销

            # 注意：.item() 会让 CPU 阻塞等待 GPU，训练/推理热路径中
            # 应攒批后再打印，避免每个 step 高频调用拖慢吞吐。

    Returns:
        None
    """
    # 分布式未初始化时（如单机调试）将 rank 视为 0，
    # 防止 dist.get_rank() 在未初始化状态下抛异常
    rank = dist.get_rank() if dist.is_initialized() else 0
    # flush=True：多进程并发写 stdout 时立即刷新缓冲区，
    # 防止日志滞留导致顺序错乱，崩溃时也不会丢已打印内容
    print(
        f"[rank={rank}][pid={os.getpid()}] {message}",
        flush=True,
    )
```
