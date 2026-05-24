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


# HOP
在你这份代码里，HOP 规约可以理解为：

1. 选中要规约的层（`aten::_foreach_addcdiv_`）。
2. 每个 GPU 准备一个“完整更新张量”（不是 Ring 那种分片）。
3. 进入轮次循环：`send -> recv -> send -> ...`。
4. `send` 阶段：发给 out-neighbors，并把消息记录到对方 `update_queues_`。
5. `recv` 阶段：从自己的 `update_queues_` 看当前 `recv_step` 是否收够 `denum` 份；够了就归约。
6. 归约后推进到下一轮；最后回到 `nextlayer` 处理下一层。

注意：你这版 `ReduceTensor` 用的是 `AvgChunks(...)`，所以这里更像“对收到值做平均”的模型。

---

**4 GPU 例子（简化）**

假设 GPU0 的 in-neighbors 有 3 个（1,2,3）。

- `backup_workers=0`  
  `denum = 3 - 0 = 3`，GPU0 必须等 3 份都到齐才归约。  
  行为偏“全量同步”。

- `backup_workers=1`  
  `denum = 3 - 1 = 2`，GPU0 收到任意 2 份主路径更新就可先归约；第 3 份可视作备份/冗余。  
  行为偏“更快推进，允许冗余路径”。

---

**和 Ring 的直观区别**

- Ring：分片，`scatter + allgather`，严格 N-1 步传播 chunk。
- 这里 HOP：整包多邻居发送，按阈值收包后归约，靠 `update_queues_ + token_queues_ + backup_workers_` 控节奏与容错。
