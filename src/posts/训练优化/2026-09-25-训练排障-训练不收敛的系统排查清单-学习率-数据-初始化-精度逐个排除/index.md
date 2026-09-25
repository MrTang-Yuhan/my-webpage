---
layout: post.njk
post_id: 2026-09-25-训练排障-训练不收敛的系统排查清单-学习率-数据-初始化-精度逐个排除
archive: 训练优化
title: 【训练排障】训练不收敛的系统排查清单：学习率、数据、初始化、精度逐个排除
date: 2026-09-25
updated: 2026-09-25
tags:
  - post
---
# 训练不收敛：视频严格版操作笔记

> 来源视频：[训练排障：训练不收敛的系统排查清单：学习率、数据、初始化、精度四步走](https://www.bilibili.com/video/BV1LQ8467E4e/)
>



## 1. 视频中的三种不收敛表现

1. **Loss 完全不降**：一直在初始值附近晃动。
2. **Loss 持续震荡**：降一点、升一点，没有明确趋势。
3. **Loss 先降后升**：常见方向是过拟合或学习率太大。

正常训练允许 Loss 小幅波动，但一段时间内应有整体下降趋势。

![](img/loss-error-fig.png)

## 2. 视频规定的排查顺序

```text
学习率（最常见） → 数据（第二常见） → 初始化 → 精度
```

每次只修改一项，并记录修改内容和效果。

## 3. 第一步：学习率

### 视频中的方法

- 学习率太大：Loss 震荡或飙升。
- 学习率太小：Loss 几乎不动。
- 先比较：当前学习率、当前值的十分之一、当前值的十倍。
- 更科学的方法：**LR Range Test**，从极小学习率逐步增大，画出 Loss 曲线，寻找 Loss 下降最陡的区间。

### 代码：三个学习率对照

```python
from copy import deepcopy
import torch


def run_short(model, loader, criterion, lr, device="cuda", steps=100):
    """使用指定学习率进行短程训练，观察 Loss 是否下降。

    Args:
        model: 待测试的 PyTorch 模型。
        loader: 提供 inputs 和 labels 的训练数据加载器。
        criterion: 损失函数。
        lr: 本次实验使用的学习率。
        device: 训练设备，例如 cuda 或 cpu。
        steps: 最多执行的训练步数。

    Returns:
        包含学习率、状态和 Loss 记录的字典。
    """
    # 深拷贝模型，确保不同学习率实验互不影响。
    model = deepcopy(model).to(device)
    # 每次实验使用独立优化器，避免复用动量等历史状态。
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    iterator = iter(loader)
    model.train()  # 开启训练模式。

    for _ in range(steps):
        # 数据迭代器耗尽后重新开始，保证短实验能连续取到 batch。
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        # 将当前 batch 移动到模型所在设备。
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        # 清空上一轮梯度，避免梯度错误累积。
        optimizer.zero_grad(set_to_none=True)

        # 前向计算和损失。
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Loss 出现 NaN 或 Inf 时立即停止，避免继续污染参数。
        if not torch.isfinite(loss):
            return {"lr": lr, "status": "NaN/Inf", "losses": losses}

        # 反向传播并更新参数。
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return {
        "lr": lr,
        "status": "finite",
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
    }


# 这里的数值只是示例；实际应替换为当前项目的学习率。
base_lr = 1e-4

# 按视频要求比较当前值、十分之一和十倍。
for lr in [base_lr / 10, base_lr, base_lr * 10]:
    print(run_short(model, train_loader, criterion, lr))
```

### 结果应该怎么看

| 结果 | 视频中的判断 |
|---|---|
| Loss 震荡或飙升 | 学习率可能太大 |
| Loss 几乎不动 | 学习率可能太小 |
| 某个学习率区间下降明显 | 优先在该区间继续调节 |
| 三组都不下降 | 转向检查数据、初始化和精度 |
| 三组都出现 NaN/Inf | 优先检查数值稳定性和初始化 |

`1e-4` 和 `100` 只是运行示例，不是视频规定的固定正确值。

### 代码：LR Range Test

```python
import math


def lr_range_test(model, loader, criterion,
                  start_lr=1e-7, end_lr=1e-1,
                  steps=200, device="cuda"):
    """逐步增大学习率，记录 Loss 曲线以定位有效区间。

    Args:
        model: 待测试的 PyTorch 模型。
        loader: 训练数据加载器。
        criterion: 损失函数。
        start_lr: 起始学习率。
        end_lr: 终止学习率。
        steps: 最大测试步数。
        device: 训练设备。

    Returns:
        每一步的 step、lr 和 loss 记录。
    """
    # 使用指数增长，让学习率覆盖较宽范围。
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    growth = (end_lr / start_lr) ** (1 / max(steps - 1, 1))
    records = []
    iterator = iter(loader)
    model.train()

    for step in range(steps):
        # 数据迭代器耗尽后重新开始，保证短实验能连续取到 batch。
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        # 在每一步更新优化器中的学习率。
        lr = start_lr * growth ** step
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 将当前 batch 移动到模型所在设备。
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        # 清空上一轮梯度，避免梯度错误累积。
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), labels)
        records.append({"step": step, "lr": lr, "loss": loss.item()})

        # Loss 非有限时停止，避免继续扩大数值异常。
        if not math.isfinite(loss.item()):
            break
        loss.backward()
        optimizer.step()

    return records
```

![](img/lr-range-test.png)

## 4. 第二步：数据

### 视频要求检查的三件事

#### 4.1 DataLoader 是否 shuffle

如果不 shuffle，每个 epoch 的数据顺序固定，模型可能学到顺序而不是内容。

```python
from torch.utils.data import DataLoader

# 训练集启用 shuffle，避免每个 epoch 使用完全相同的数据顺序。
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
```

正确检查重点：训练集 DataLoader 的 `shuffle` 应为 `True`。

#### 4.2 输入和标签是否合理

```python
# 取出一个 batch，直接检查送入模型的真实数据。
batch = next(iter(train_loader))

print("inputs shape:", batch["inputs"].shape)
print("labels shape:", batch["labels"].shape)
print("inputs sample:", batch["inputs"][0])
print("labels sample:", batch["labels"][0])  # 检查标签是否与输入对应。
```

视频建议打印几个 batch，检查：

- 输入是否是预期内容。
- 标签是否与输入对应。
- 输入和标签的形状是否合理。
- 标签是否明显错误。

#### 4.3 标签分布是否偏斜

```python
from collections import Counter

# 统计标签出现次数，用于发现类别分布明显偏斜。
label_counts = Counter()
for batch in train_loader:
    label_counts.update(batch["labels"].view(-1).tolist())

print(label_counts)
```

如果某些类别样本太少，数据分布可能导致模型难以学习这些类别。视频没有给出一个固定的“正确比例”，判断标准是是否存在明显偏斜，或是否与预期数据分布不一致。

## 5. 第三步：初始化

### 视频中的原理

权重初始化方差不对，会让前向信号逐层放大或缩小，从而造成**梯度消失**或**梯度爆炸**。

视频提到：大模型通常使用截断正态分布初始化，标准差一般为 **0.02**。如果修改模型结构后忘记调整初始化，可能导致不收敛。

### 5.1 查看参数统计

```python
# 遍历可训练参数，检查初始化后的数值规模。
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue

    # 转成 float 统计，避免低精度类型影响观察结果。
    value = param.detach().float()
    print(
        name,
        "mean=", value.mean().item(),
        "std=", value.std().item(),
        "min=", value.min().item(),
        "max=", value.max().item(),
    )
```

### 5.2 查看每层激活值范数

```python
# 保存各层前向输出的范数，用于观察信号是否逐层放大或缩小。
activation_norms = {}


def make_hook(name):
    """创建记录指定模块激活值范数的 forward hook。"""
    def hook(module, inputs, output):
        del module, inputs  # hook 中不需要这两个参数。
        value = output[0] if isinstance(output, tuple) else output
        activation_norms[name] = value.detach().float().norm().item()
    return hook


# 在线性层和 LayerNorm 上注册 hook，覆盖视频重点关注的层。
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm)):
        module.register_forward_hook(make_hook(name))

# 运行一次前向传播，触发所有 hook。
_ = model(batch["inputs"].to("cuda"))

for name, value in activation_norms.items():
    print(name, "activation_norm=", value)
```

### 5.3 视频中的判断方式

| 观察结果 | 应检查的方向 |
|---|---|
| 激活范数逐层明显变大 | 初始化方差可能过大，信号逐层放大 |
| 激活范数逐层明显变小 | 初始化方差可能过小，信号逐层缩小 |
| 某一层突然异常 | 检查该层结构和初始化 |
| 激活出现 NaN/Inf | 检查初始化和数值稳定性 |
| 激活各层整体稳定 | 暂未发现视频所说的明显初始化异常 |

`0.02` 是视频提到的常见大模型初始化标准差，不是所有层的硬性标准。

## 6. 第四步：精度和数值稳定性

### 视频中的风险

- 混合精度中的 FP16 可能导致某些层精度不够。
- FP16 可能出现梯度下溢。
- LayerNorm 的 epsilon 太小，可能除以接近零的数。

### 6.1 FP32 对照实验

```python
# 使用 FP16 混合精度计算，作为待检查的训练路径。
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss_fp16 = criterion(model(inputs), labels)

# 关闭 autocast，使用 FP32 作为对照实验。
with torch.autocast(device_type="cuda", enabled=False):
    loss_fp32 = criterion(model(inputs.float()), labels)

print("FP16 loss:", loss_fp16.item())
print("FP32 loss:", loss_fp32.item())
print("FP16 finite:", torch.isfinite(loss_fp16).item())
print("FP32 finite:", torch.isfinite(loss_fp32).item())
```

### 6.2 结果应该怎么看

| FP16 | FP32 | 视频中的判断 |
|---|---|---|
| Loss 不能下降或出现 NaN | Loss 能下降 | 说明可能是精度问题 |
| Loss 和 FP32 都不能下降 | 都有问题 | 回到学习率、数据、初始化排查 |
| 两者都能下降 | 暂未发现明显精度问题 | 继续观察训练趋势 |

### 6.3 检查每层梯度范数

```python
loss.backward()

# 遍历可训练参数，检查初始化后的数值规模。
for name, param in model.named_parameters():
    if param.grad is None:
        continue

    # 使用 float 统计梯度，便于识别接近零或非有限值。
    value = param.grad.detach().float()
    print(
        name,
        "grad_norm=", value.norm().item(),
        "finite=", torch.isfinite(value).all().item(),
    )
```

视频中的判断：

- 梯度接近零：检查梯度下溢或梯度消失。
- 梯度出现 NaN/Inf：检查数值稳定性。
- FP32 中梯度正常、混合精度中异常：精度问题的可能性增加。

视频没有给出所有模型通用的固定梯度阈值，应比较同一 batch 下 FP16 和 FP32 的结果。

### 6.4 检查 LayerNorm epsilon

```python
# 检查 LayerNorm 的 epsilon，确认是否存在过小的配置。
for name, module in model.named_modules():
    if isinstance(module, torch.nn.LayerNorm):
        print(name, "epsilon=", module.eps)
```

如果 epsilon 太小，且训练出现 NaN、Inf 或 FP16/FP32 结果明显不同，应重点检查该层的数值稳定性。

## 7. 严格按视频执行的一次实验流程

### 第一步：确认现象

记录 Loss 是：

- 完全不降。
- 持续震荡。
- 先降后升。

### 第二步：只改学习率

依次测试：

```text
当前值
当前值 / 10
当前值 × 10
```

记录哪一个值能让 Loss 更稳定地下行。

### 第三步：只查数据

- 检查 `shuffle`。
- 打印几个 batch 的输入和标签。
- 检查标签是否正确。
- 统计标签分布。

### 第四步：只查初始化

- 打印每层权重统计。
- 打印每层激活值范数。
- 看信号是否逐层放大或缩小。

### 第五步：只查精度

- 关闭混合精度。
- 用 FP32 跑几个 step。
- 对比 Loss 和每层梯度范数。
- 检查 LayerNorm 的 epsilon。

## 8. 视频内容对应的记录表

| 实验 | 只修改的项目 | 设置 | Loss 现象 | 结论 |
|---|---|---|---|---|
| 1 | 学习率 | 当前值 |  |  |
| 2 | 学习率 | 当前值 ÷ 10 |  |  |
| 3 | 学习率 | 当前值 × 10 |  |  |
| 4 | 数据 | 检查 shuffle |  |  |
| 5 | 数据 | 打印 batch 和标签 |  |  |
| 6 | 数据 | 统计标签分布 |  |  |
| 7 | 初始化 | 打印激活值范数 |  |  |
| 8 | 精度 | FP32 对照 |  |  |

## 9. 视频核心结论

训练不收敛时，按照视频的顺序执行：

1. **先查学习率**，因为它最常见。
2. **再查数据**，确认 shuffle、输入、标签和分布。
3. **再查初始化**，观察激活值是否逐层放大或缩小。
4. **最后查精度**，用 FP32 对照混合精度，并检查梯度范数和 LayerNorm epsilon。
5. **一次只改一个变量**，并记录每次改动的效果。

> **来源**：哔哩哔哩视频页面 BV1LQ8467E4e。本文没有把视频未提到的固定阈值、额外训练技巧或扩展排查流程写成视频结论。

## 10. 视频画面

以下为该视频在哔哩哔哩页面提供的**视频封面图**，用于标识笔记对应的视频来源。

![视频封面：训练不收敛的系统排查清单](bilibili_training_nonconvergence_cover.png)

图片来源：哔哩哔哩视频 BV1LQ8467E4e 的公开页面封面图。


