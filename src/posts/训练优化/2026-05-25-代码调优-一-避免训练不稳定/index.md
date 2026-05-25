---
layout: post.njk
post_id: 2026-05-25-代码调优-一-避免训练不稳定
archive: 训练优化
title: 代码调优（一）：避免训练不稳定
date: 2026-05-25
tags:
  - post
---
为了避免训练不稳定，需要着重注意以下要点。内容整理自 Stas Bekman 的 [Machine Learning Engineering by Stas Bekman](https://github.com/stas00/ml-engineering)。

# 权重初始化标准差（std）的选择

**权重初始化的标准差（std）不是固定的，必须根据隐藏层维度（hidden dimension）调整。** 选错了，模型在训练初期就会崩溃。

## 背景故事

BLOOM 团队在训练 **1040 亿参数** 的预实验模型时，遇到了严重的训练不稳定问题：

- 使用 Megatron-LM 框架的 **默认初始化 std = 0.02**
- 结果：**训练几千步后就崩溃**（loss 爆炸、梯度消失/爆炸）
- 排查后发现：**0.02 对这个规模的模型来说太大了**



## 两种初始化公式对比

| 来源 | 论文 | 公式 | 等价形式 |
|------|------|------|----------|
| **Transformers without Tears** | [arXiv:1910.05895](https://arxiv.org/abs/1910.05895) | `sqrt(2 / (NHIDDEN * 5))` | `sqrt(0.4000 / NHIDDEN)` |
| **530B 模型训练实践** | [arXiv:2201.11990](https://arxiv.org/abs/2201.11990) | `sqrt(1 / (NHIDDEN * 3))` | `sqrt(0.3333 / NHIDDEN)` |


**BLOOM 选择了第二个（530B 的公式）**，因为它给出的初始化值**更小、更保守**。其中, `NHIDDEN` 是隐藏层维度。

因此，当 `NHIDDEN = 14336` 时，计算结果为 $\sqrt{1/(14336 \times 3)} = 0.00482$，这就是实际采用的标准差的值。作者认为，这当然不是 BLOOM-176B 训练过程中没有出现稳定性问题的唯一原因，但它是关键因素之一。

## 结论

虽然这不是 BLOOM-176B 训练稳定的唯一原因，但作者认为它是**关键因素之一**。


这本质上是在控制**初始权重的幅度**：太大的初始化会让前向传播初期的激活值爆炸，太小则可能导致梯度消失。大模型的深度和宽度放大了这个问题。


# 避免数值溢出

在 FP16/BF16 混合精度训练中，`Q @ K^T` 可能先产生很大的中间结果；如果缩放因子 `1 / self.norm_factor`[^1] 是在矩阵乘法之后才乘上去，那么中间结果可能已经溢出成 `inf` 或产生不稳定值，后面的缩放已经救不回来。

[^1]: norm_factor 的取值可能根据**模型架构、attention 变体以及长度外推方法不同而有所改变**。在原始 Transformer 论文的 scaled dot-product attention 中，`norm_factor` 取值为 `sqrt(d_k)`。

所以，纠正做法是：不要等 `Q @ K^T` 算完后再整体乘缩放因子，而是把缩放因子提前拆到 `Q` 和 `K` 上，让它们先变小，再做矩阵乘法。即:

```
(Q / sqrt(norm_factor)) @ (K^T / sqrt(norm_factor))
= (Q @ K^T) / norm_factor
```

这样可以有效避免数值溢出。

