---
layout: post.njk
post_id: 2026-08-06-torch-distributed-分布式训练-02-基本用法
archive: 训练优化
title: torch.distributed 分布式训练（02）：基本用法
date: 2026-08-06
tags:
  - post
---
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

