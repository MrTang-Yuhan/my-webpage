---
layout: post.njk
archive: mode-parallelism
title: transformer 模型的 GPU 显存使用分析（二）：推理
date: 2026-05-11
description: transformer 模型的 GPU 显存使用分析：推理阶段分析
tags:
  - post
  - GPU memory usage
  - transformer
  - model parallelism
---
# 整体架构

这幅图展示了 Decode-only Transformer 的总体架构图：

![decode-only](img/overview.png)

# 维度分析

* `B`: batch size
* `T`: 当前输入序列长度
* `L`: Transformer 层数
* `d_model`: 隐藏层维度
* `n_head`： 注意力头数
* `d_head`: 每个注意力头的维度。d_head = d_model / n_head

## Transformer-Block 中的维度

输入 `x`:

-  `[B, T, d_model]`

输出 `out`: 

- `[B, T, d_model]`

## Attention 中的维度 

输入 `x`: 

- `[B, T, d_model]`

经过线性层，得到`Q`, `K`, `V`:

- `Q` = `x @ Wq`
- `K` = `x @ Wk`
- `V` = `x @ Wv`

如果是标准多头注意力 MHA，则

- `Q, K, V`: `[B, T, n_head, d_head]`

一般会转置成：

- `Q, K, V`: `[B, n_head, T, d_head]`

### KV Cache 的维度

KV Cache 存的是每一层的历史 `K` 和 `V`。

对第 `l` 层:

- `k_cache[l], v_cache[l]`: `[B, n_head, T_cache, d_head]`

其中, 

- `T_cache` 是已经处理过的 token 数
- 每一层都有自己的 K/V cache
- Q 不缓存，因为每一步只需要当前 token 的 Q

总 KV Cache 结构可以理解为：

```text
KV Cache:
[
  layer 0: K, V
  layer 1: K, V
  ...
  layer L-1: K, V
]
```

### Attention 在 Prefill 和 Decode 阶段的区别

#### Prefill 阶段

假设 prompt 长度是 `T_prompt`。

输入：

- ```text
  x: [B, T_prompt, d_model]
  ```

每层生成：


- ```text
  K, V: [B, n_head, T_prompt, d_head]
  ```

并写入 cache：

- ```text
  K_cache, V_cache: [B, n_head, T_prompt, d_head]
  ```

Attention 计算：

- ```text
  scores = Q @ K^T
  ```

维度：

- ```text
  Q:      [B, n_head, T_prompt, d_head]
  K^T:    [B, n_head, d_head, T_prompt]
  
  scores: [B, n_head, T_prompt, T_prompt]
  ```

输出：

- ```text
  attn_out: [B, n_head, T_prompt, d_head]
  合并 heads 后: [B, T_prompt, d_model]
  ```

#### Decode 阶段

每次只输入一个新 token：

-  `x_new: [B, 1, d_model]`

当前 token 生成：

- `Q_new, K_new, V_new: [B, n_head, 1, d_head]`

把新的 K/V 追加到 cache：

- `K_cache, v_cache : [B, n_head, T_cache + 1, d_head]`

Attention 使用 Q_new 和完整的 K_cache/V_cache 做注意力，计算维度：

```text
Q_new:    [B, n_head, 1, d_head]
K_cache:  [B, n_head, T_cache + 1, d_head]

scores = Q_new @ K_cache^T
scores:   [B, n_head, 1, T_cache + 1]
```

再乘以 V：

```text
V_cache:  [B, n_head, T_cache + 1, d_head]

attn_out: [B, n_head, 1, d_head]
```

合并 heads：

```text
attn_out: [B, 1, d_model]
```

#### MLP 层维度

Attention 输出后进入 MLP。

输入：

- `[B, T, d_model]`

常见 FFN 维度：

```text
- up projection:   [B, T, d_model] → [B, T, d_ff]
- activation
- down projection: [B, T, d_ff] → [B, T, d_model]
```

通常：

- `d_ff ≈ 4 × d_model`



#### 最后输出 logits

最后一层输出：

- `hidden: [B, T, d_model]`

经过 LM Head：

- `logits = hidden @ W_vocab`

其中：

- `W_vocab: [d_model, vocab_size]`

输出：

- `logits: [B, T, vocab_size]`

Decode 阶段只关心最后一个位置：

- `logits: [B, 1, vocab_size]`

然后采样得到下一个 token。

# FLOPs 分析




