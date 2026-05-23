---
layout: post.njk
post_id: 2026-05-23-triosim-模拟器-流水线并行
archive: triosim模拟器
title: " triosim 模拟器: 流水线并行"
date: 2026-05-23
tags:
  - post
---
整体思路：  
把一个 batch 切成多个 `micro_batch`，让不同 GPU（不同 stage）同时处理不同 `micro_batch`，实现时间重叠。

可用这三个维度理解：

- `GPU id`：表示 pipeline stage（哪张卡负责哪段层）
- `micro_batch_id`：同一 batch 里第几个小批次（代码里基本对应 `round_id`）
- `batch_id`：外层大 batch（你当前一次运行通常固定 1 个 batch）

---

举例：4 张 GPU，1 个 batch 被切成 4 个 micro-batch

- `batch_id = 0`
- `micro_batch_id = 0,1,2,3`
- `GPU0..GPU3` 各负责一段连续层

时间片示意（forward）：

1. `t0`  
- `GPU0` 处理 `(batch0, micro0)`

2. `t1`  
- `GPU0` 处理 `(batch0, micro1)`  
- `GPU1` 处理 `(batch0, micro0)`

3. `t2`  
- `GPU0` 处理 `(batch0, micro2)`  
- `GPU1` 处理 `(batch0, micro1)`  
- `GPU2` 处理 `(batch0, micro0)`

4. `t3`  
- `GPU0` 处理 `(batch0, micro3)`  
- `GPU1` 处理 `(batch0, micro2)`  
- `GPU2` 处理 `(batch0, micro1)`  
- `GPU3` 处理 `(batch0, micro0)`

这就是“流水线填满”。

---

代码里怎么落地：

1. `PipelineStart()`  
- 分层到各 GPU  
- `num_round_ = 0`，启动 `round 0`（即 `micro_batch_id=0`）

2. 每个 `round_id`（micro_batch）独立维护：
- `pipe_fetching_layer_index[round_id]`
- `pipe_computing_layer_index[round_id]`
- `pipe_action_[round_id]`（`doforward/dobackward`）

3. 当需要进入下一 micro-batch 时  
- 通过消息 `purpose=="nextRound"`  
- `NotifyRecv -> NextRoundPipelineStart()`  
- `num_round_++`，开启新的 `round_id`

4. GPU 间通过 `nextGPU` 消息传递该 micro-batch 的输出，驱动下一 stage。

