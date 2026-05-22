---
layout: post.njk
post_id: 2026-05-22-软件调优-三-torch-compile
archive: 训练优化
title: 软件调优（三）：torch.compile
date: 2026-05-22
tags:
  - post
---
# 介绍 torch.compile

`torch.compile` 通过将 PyTorch 代码 JIT 编译成优化的内核来加速 PyTorch 代码的运行，同时只需要极少的代码修改。

- [torch.compile 编程模型](https://docs.pytorch.ac.cn/docs/stable/user_guide/torch_compiler/compile/programming_model.html#google_vignette) 详细介绍了 torch.compiler 原理。

- [torch.compile 使用教程](https://docs.pytorch.ac.cn/tutorials/intermediate/torch_compile_tutorial.html) 详细介绍了如何使用 torch.compiler。

# torch.compile 使用

## 使用示例

**无论是训练还是推理，只要条件允许，都应尽可能使用 `torch.compile` 来获得性能提升。**

### 训练

尽管 `torch.compile` 有[多种参数配置](https://docs.pytorch.ac.cn/docs/stable/generated/torch.compile.html)，但目前还没有找到特别好的资料来说明哪种配置在训练情况下最好。所以，直接使用默认配置就行：
```python
import torch

model = torch.compile(model, mode="default")
``` 

### 推理

追求极致的性能。

```python
import torch

# LLM 推理推荐：max-autotune 模式，极致性能
model = torch.compile(
    model,
    mode="max-autotune",           # 编译时间最长，推理性能最好
    # mode="reduce-overhead",      # 如果编译太慢，用这个平衡
    dynamic=False,                 # 推理时通常固定最大长度，用 padding
    fullgraph=True,                # 生产环境要求完整图，不允许 graph break
)
```

## 注意事项

### 首次编译开销

`torch.compile` 会增加编译开销，会产生首次编译延迟。所以应该**在训练/推理前做一次 warm-up**。示例：
```pytorch

compiled_model = torch.compile(model)

# 第一次前向传播会触发编译，产生时间开销
_ = compiled_model(x) 

for i in range(1000):
    y = compiled_model(x)  

```

### 动态形状会导致重编译

当模型输入发生形状改变时，可能重新编译 `torch.compile`，如下：

```
# 每轮 batch size 不同，触发多次编译
compiled_model = torch.compile(model)
for bs in [16, 32, 64, 16, 32]:
    x = torch.randn(bs, 128, device="cuda")
    y = compiled_model(x)  # 每次 bs 变化都可能重编译！
```
所以：
- 训练时**尽量使用固定 batch**。
- 推理时如果序列长度变化大，考虑使用参数`torch.compile(model, dynamic=True)`。编译器尝试生成一个能容忍一定范围形状变化的通用内核，从而避免因为形状改变引起的重新编译。但是更加广泛的做法是固定最大序列长度，不足则 pad，然后实际有效长度通过 attention 掩码控制。

### 图断裂（Graph Breaks）

torch.compile 遇到数据依赖的控制流或不支持的 Python 操作时，会"断图"，回退到 eager mode：

```
def forward(self, x):
    x = self.linear(x)
    
    # Graph Break
    if x.sum() > 0:  # 编译器无法静态确定走哪个分支
        x = self.branch_a(x)
    else:
        x = self.branch_b(x)
    
    return x
```
