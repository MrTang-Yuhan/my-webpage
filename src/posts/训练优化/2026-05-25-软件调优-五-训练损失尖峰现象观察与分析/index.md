---
layout: post.njk
post_id: 2026-05-25-软件调优-五-训练损失尖峰现象观察与分析
archive: 训练优化
title: 软件调优（五）：训练损失尖峰现象观察与分析
date: 2026-05-25
tags:
  - post
---
此处展示一些好的、不好的以及异常的训练模式。内容整理自 Stas Bekman 的 [Machine Learning Engineering by Stas Bekman](https://github.com/stas00/ml-engineering)。

# 一个非常失败的训练案例

在启动 BLOOM-176B 训练之前，我们用 104B 模型做了多次实验。但始终没能解决 **训练很早就出现梯度发散（diverge）** 的问题。如下图所示：

![](img/fail_example.png)


我们做了很多尝试，也用了很多技术（详见之前的记录）。我们认为主要障碍有两个：
- 使用了 **FP16**（半精度浮点数）
- 数据中包含大量**垃圾/噪声数据**

对于 [BLOOM-176B](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml) 的训练，我们改为使用 **BF16**，使用了**更干净的数据**，并额外添加了一个 **embedding layer-norm**。这些改进带来了巨大的不同。BLOOM-176B 的训练损失曲线几乎完美，只出现了一次明显的尖峰（spike），而且在 200 步内就恢复了。如下图所示：

![](img/successful_example.png)

你可以查看 [TensorBoard](https://huggingface.co/bigscience/tr11-176B-logs/tensorboard) 放大细节并检查其他曲线。

## 该失败案列的整个训练日志

BigScience 团队原本想训练一个约 **104B 参数**的 Transformer 语言模型，但训练过程中反复出现类似问题：

- 训练刚开始一段时间 loss 正常下降；
- 到某个迭代步附近，尤其是学习率 warmup 接近结束或结束后，loss 突然飙升；
- 有时 loss 会短暂恢复；
- 但之后又继续恶化，最后变成 **NaN**；
- 多次回滚 checkpoint、改随机种子、改优化器参数、改模型结构、改学习率，都没能彻底解决。

所以这份文档记录了他们从 **Experiment 1 到 Experiment 10** 的尝试过程。原文档[bigscience/train/tr8-104B-wide/chronicles.md](https://github.com/bigscience-workshop/bigscience/raw/refs/heads/master/train/tr8-104B-wide/chronicles.md)。


## Experiment 1：第一次训练，loss 在 7000 步附近爆炸

配置：

```text
Nodes: 64
Seed: 42
Started from iteration 0
```

训练从头开始。到大约 iteration 7000 时，`lm loss` 从 6.4 突然跳到 14，后来又回到约 7，但训练质量已经受损，之后进入 NaN。`lm loss` 随 iteration 变化如下图：

![](img/tr8-104b-glitch-1.png)

他们认为继续训练没有帮助，于是采取措施：

1. 回滚到最后一个正常 checkpoint：`global_step6210`；
2. 改随机种子，从 `42` 改为 `43`；
3. 清理 TensorBoard 日志中回滚之后的部分；
4. 保存原始出问题的 TensorBoard 数据，方便以后复盘。

他们还提到，改 seed 会重新生成数据顺序。如果问题是某一批坏数据导致的，那么重新 shuffle 数据有可能解决。

但后续证明，问题并没有消失。

---

## Experiment 2：换 seed 后问题更早出现

配置：

```text
Nodes: 64
Seed: 43
Restarted from global_step6210
```


这次从 6210 步 checkpoint 继续训练，随机种子变成 43。

结果类似：

- loss 从 6.3 上升到 9，再到 10；
- 梯度 norm 也出现异常增大；
- 问题比第一次更早出现。

这里有一个重要讨论：Conglong Li 观察到这几次 glitch 都接近 **学习率 warmup 结束**[^1]。

[^1]: `LR warmup` 是学习率预热。训练初期不直接用最大学习率，而是从较小值逐步升高到目标学习率

他认为：

- warmup 结束附近梯度最大；
- Adam 优化器的梯度方差在学习率峰值附近也最不稳定；
- 这是训练最容易发散的时候。

团队还回顾了之前的 13B 模型训练，也曾在 warmup 结束附近出现过巨大 loss spike，但较小的模型后来恢复了，如下图所示。而 104B 这个模型没有恢复。

![](img/tr1-13b-glitch-1-2.png)

LR warmup 在大概第 25k 次迭代时世界树，而第一个大的 glitch 出现在大概第 29k 次迭代时。25k 和 29 k在数值上是足够接近的。

此外，团队对 [1.5B 参数的 GPT-2 模型进行的研究](https://arxiv.org/pdf/2108.06084.pdf)中，使用了 3k 步的学习率预热。在这里看到，梯度方差范数（左图）直到 8k 多步后才达到底部，而基线的梯度方差最大值在前 10k 多步期间一直很不稳定。

![](img/step-wise-adam-variance-1.png)


因此他们开始怀疑：

> 这可能不是单纯的数据问题，而是大模型训练稳定性问题。

---

## Experiment 3：尝试更稳定的 self-attention 计算

有人建议 attention 的计算方式可能导致 fp16 数值不稳定。

原先计算 attention score 时，可能是：

```text
QK^T 之后再乘缩放因子
```

如果 Q 和 K 的维度很大，矩阵乘法结果可能先变得非常大，再乘缩放因子已经来不及避免溢出。

建议改成：

```text
先缩放 Q 和 K，再做矩阵乘法
```

数学上两者等价：

```text
n * (A dot B) == (sqrt(n) * A) dot (sqrt(n) * B)
```

这样可以减少中间结果过大的风险。

他们回滚到 `iteration 6210`，保持 seed 43，也关闭了 codecarbon 的日志噪音。

但结果：

- Experiment 3 仍然以类似 Experiment 2 的方式失败。

说明 attention 中这个数值稳定性修改没有解决根本问题。

---

## Experiment 4：尝试 Adam beta2 = 0.95，并更早回滚

Iz Beltagy 提出了一组假设和建议。

他认为可能原因包括：

### 可能原因 1：坏数据

如果是坏数据导致，重新 shuffle 数据应该有帮助。但团队之前改 seed 后问题仍然存在，所以他不太相信是数据问题。

### 可能原因 2：fp16 问题

如果 fp16 不稳定，可以试试全 fp32。如果 fp32 正常，就说明问题在 fp16 数值稳定性上。

### 可能原因 3：Adam beta2 太高

他们使用的是：

```text
beta2 = 0.999
```

而 GPT-3 使用过：

```text
beta2 = 0.95
```

较低的 beta2 可能更稳定，但训练会慢一些。

### 可能原因 4：模型结构设计不好

这个模型被称为 “wide”，意思是宽度很大、层数相对较少。Iz 认为宽深比例可能不合理，导致模型学得不好，也更容易发散。

### 可能原因 5：回滚点太晚

他观察 loss-scale 曲线，认为模型可能早在 4700 步左右就已经开始朝发散方向发展。因此建议回滚到更早，比如 3000 步。

所以 Experiment 4 的改动是：

```text
回滚到 iteration 3000
adam-beta2 = 0.95
```

结果如图：

> 训练在 iteration 5000 附近又朝错误方向发展，实验停止。

![](img/tr8-104b-glitch-4.png)


## Experiment 5：从头开始，仍然 beta2 = 0.95

Experiment 5 基本和 Experiment 4 一样，但不是从 3000 步恢复，而是从头开始。

结果如下图：

> 仍然在 5000 步附近发散。

![](img/tr8-104b-glitch-5.png)

这说明仅仅改 Adam beta2 并不能解决问题。


## Experiment 6：发现模型配置有严重错误

这是文档中很重要的转折点。

他们发现之前所有实验都有一个严重配置错误：

```text
FFN_HIDDEN_SIZE 没有从 13B 配置更新到 104B 配置
```

原来它仍然是：

```text
FFN_HIDDEN_SIZE = 20480
```

但正确应该是：

```text
FFN_HIDDEN_SIZE = 65536
```

在 Transformer 中，FFN hidden size 通常是 hidden size 的 4 倍：

```text
FFN_HIDDEN_SIZE = 4 * HIDDEN_SIZE
```

由于这个值太小，之前训练的其实不是预期中的 104B 模型，而是一个结构非常不平衡的约 **58B 模型**。

他们总结教训：

> 以后不要在 slurm 脚本里手动设置 FFN_HIDDEN_SIZE，而是让 Megatron 自动设置为 4 * HIDDEN_SIZE，避免类似错误。

修正后，模型显存需求大幅增加，需要更多并行配置：

```text
TP_SIZE = 4
PP_SIZE = 32
```

解释：

- TP = Tensor Parallelism，张量并行；
- PP = Pipeline Parallelism，流水线并行；
- 模型越大，需要更多 GPU 切分。

但即使修复模型结构后，Experiment 6 的结果仍然类似：

> loss 仍然出现发散。

![](img/tr8-104b-glitch-6.png)


这说明 FFN 配置错误虽然严重，但不是唯一原因。



## Experiment 7：调整宽深比，做成更“正常”的模型

之前模型太“宽”，宽深比例大约是 512。团队认为这可能不合理。

于是尝试把模型改成更深、更窄：

```text
NLAYERS = 64
NHIDDEN = 11600
NHEADS = 80
```

也就是：

- 层数从 32 增加到 64；
- hidden size 从 16384 降到 11600；
- attention heads 改为 80；
- 宽深比例从 512 降到约 180。

他们认为这更接近 Megatron 论文中大模型的常见比例。

结果：

> Experiment 7 仍然失败。

![](img/tr8-104b-glitch-7.png)


## Experiment 8：学习率减半，warmup 略加长

配置修改：

```text
lr = 3e-5
lr-warmup-samples = 300_000
```

也就是把学习率从之前的 `6e-5` 降到 `3e-5`，并把 warmup 稍微加长。

结果：

> 失败模式和 Experiment 7 类似。

![](img/tr8-104b-glitch-8.png)

## Experiment 9：更长 warmup

配置：

```text
lr = 3e-5
lr-warmup-samples = 1_000_000
```

这次 warmup 更长，想让学习率上升得更慢，训练更稳定。

结果：

> 仍然类似失败。

![](img/tr8-104b-glitch-9.png)

## Experiment 10：学习率降到 1e-5

配置：

```text
lr = 1e-5
```

最初他们想从 Experiment 9 的 6900 步继续训练，但 Megatron 不允许 checkpoint 中的学习率配置和当前配置不一致，于是只能从头开始。

结果：

> 仍然失败。

![](img/tr8-104b-glitch-10.png)



不过这些学习率实验有一个规律：

| 实验 | 学习率 | warmup |
|---:|---:|---:|
| 7 | 6e-5 | 0.26M |
| 8 | 3e-5 | 0.3M |
| 9 | 3e-5 | 1M |
| 10 | 1e-5 | 1M |

他们总结说：

> 四个实验行为很相似，只是学习停止和发散发生得越来越晚。

也就是说，降低学习率和拉长 warmup 能推迟问题，但没有消除问题。

![](img/tr8-104b-glitch-7-10.png)

