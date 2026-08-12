---
layout: post.njk
post_id: 2026-08-12-torch-compile-加速-pytorch-执行-01-简介
archive: 训练优化
title: " torch.compile 加速 Pytorch (01)：简介"
date: 2026-08-12
tags:
  - post
---


> 来源：[Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)  
> 作者：William Wen  
> 适用版本：PyTorch >= 2.0

---

## 1. torch.compile 简介

`torch.compile` 是加速 PyTorch 代码的新方式。它通过 **JIT 编译** 将 PyTorch 代码编译为优化的内核（optimized kernels），同时只需极少的代码改动。

> `torch.compile` 的加速来源于：减少 Python 开销、减少 GPU 读写、算子融合。

其工作原理是**追踪（tracing）** Python 代码，寻找其中的 PyTorch 算子。对于难以追踪的代码，`torch.compile` 会产生 **graph break（图断点）**，这意味着丢失了优化机会，退回 eager 模式执行，但**不会**导致错误或者静默的不正确结果。

**所有深度学习训练/推理的代码，如果没有特殊理由，在正式部署时最好都使用 torch.compile**。

---

## 2. 基本用法

### 2.1 作为装饰器用于任意函数

`torch.compile` 是一个装饰器，可以接受**任意的 Python 函数**。

```python
import torch

@torch.compile
def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

# 也可以作为函数调用
opt_foo2 = torch.compile(foo)
```

`torch.compile` 会**递归应用**，因此顶层编译函数内部嵌套的函数调用也会被编译。

```python
@torch.compile
def outer(x, y):
    def inner(x):
        return torch.sin(x)
    a = inner(x)
    b = torch.cos(y)
    return a + b
```

### 2.2 用于 torch.nn.Module

也可以直接优化 `torch.nn.Module` 实例，有两种等价方式：

1. 调用模块的 `.compile()` 方法
2. 直接用 `torch.compile(module)`

这两种方式都等价于对模块的 `__call__` 方法（间接调用 `forward`）进行编译。

```python
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(3, 3)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))

mod = MyModule().cuda()
opt_mod = torch.compile(mod)
# 等价于：mod.compile()
```

> **本质**：用于 `torch.nn.Module` 类时，本质上是加速其 `__call__` 方法（最终调用 `forward`）。

---

## 3. 初次编译会增加时间

`torch.compile` 在**首次执行**时，相比 eager 模式会花费**显著更长的时间**，因为它需要额外的编译开销。

但 `torch.compile` 会尽可能**复用已编译的代码**。当多次运行优化后的模型时，就能看到相比 eager 模式的显著加速。

---

## 4. torch.compile 加速演示

以下代码演示 `torch.compile` 的加速效果。使用 CUDA Event 进行精确计时。

```python
def foo3(x):
    y = x + 1
    z = torch.nn.functional.relu(y)
    u = z * 2
    return u

opt_foo3 = torch.compile(foo3)

# 返回 fn() 的运行结果和耗时（秒）
def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

inp = torch.randn(4096, 4096).cuda()

# 第一次运行：编译版本更慢（包含编译开销）
print("compile:", timed(lambda: opt_foo3(inp))[1])
print("eager:", timed(lambda: foo3(inp))[1])
```

**首次运行输出示例**：
```
compile: 0.539781494140625
eager: 0.0267775993347168
```

可以看到首次编译非常慢。接下来多次运行取中位数：

```python
import numpy as np

eager_times = []
for i in range(10):
    _, eager_time = timed(lambda: foo3(inp))
    eager_times.append(eager_time)

compile_times = []
for i in range(10):
    _, compile_time = timed(lambda: opt_foo3(inp))
    compile_times.append(compile_time)

eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
print(f"eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
```

**多次运行后输出示例**：
```
eager median: 0.0008657919764518738
compile median: 0.000347135990858078
speedup: 2.4941002928326195x
```



---

## 5. fullgraph=True 与 torch.cond

### 5.1 Graph Break（图断点）

Graph break 是 `torch.compile` 中最基础的概念之一。当遇到不支持的 Python 代码（如数据依赖的控制流）时，编译器会中断编译，用 Python 运行这段不支持代码（即 eager 模式），然后**恢复编译**。

Graph break 导致优化机会丢失，但比静默错误或硬崩溃更好。

示例：数据依赖的 `if` 语句导致 graph break

```python
def bar(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:          # 数据依赖控制流 → graph break
        b = b * -1
    return x * b

opt_bar = torch.compile(bar)
opt_bar(torch.randn(10), torch.randn(10))
```

此时 `torch.compile` 会生成多个子图，在 `if` 处断开，由 Python 解释器处理分支判断。

### 5.2 fullgraph=True

如果希望**禁止任何 graph break**，强制整个函数被完整编译为一个图，从而获取极致的性能优化，则可以使用 `fullgraph=True`。一旦遇到 graph break，会直接抛出错误。

```python
torch._dynamo.reset()

opt_bar_fullgraph = torch.compile(bar, fullgraph=True)
opt_bar_fullgraph(torch.randn(10), torch.randn(10))
```

**报错信息示例**：
```
torch._dynamo.exc.Unsupported: Data-dependent branching
Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). 
Dynamo does not support tracing dynamic control flow.
Hint: Use `torch.cond` to express dynamic control flow.
```

### 5.3 使用 torch.cond 解决

对于数据依赖的动态控制流，可以使用 `torch.cond` 来表达，从而避免 graph break。

```python
from functorch.experimental.control_flow import cond

@torch.compile(fullgraph=True)
def bar_fixed(a, b):
    x = a / (torch.abs(a) + 1)

    def true_branch(y):
        return y * -1

    def false_branch(y):
        # 注意：torch.cond 不允许输出别名（aliased outputs），即输出必须是一个全新的张量。
        return y.clone()

    b = cond(b.sum() < 0, true_branch, false_branch, (b,))
    return x * b

inp1,inp2 = torch.ones(1), torch.ones(1)

bar_fixed(inp1, inp2)
bar_fixed(inp1, -inp2)
```

使用 `torch.cond` 后，整个函数可以被完整编译，不再产生 graph break。

> torch.cond 对两个分支函数的要求：
> 1. 输入输出签名一致（**参数个数、返回值个数、shape/dtype 必须相同**）；
> 2. 不允许返回别名张量（**不能返回输入本身或 view，必须创建新张量，如 .clone() 或运算结果**）

---
