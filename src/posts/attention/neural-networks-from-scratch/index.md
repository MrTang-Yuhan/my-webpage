---
layout: post.njk
title: "从零开始构建神经网络"
date: 2026-05-06
description: "通过不依赖任何库从零实现神经网络，深入理解其背后的数学原理和直觉。"
tags:
  - post
  - machine learning
  - neural networks
  - python
---

PyTorch 和 TensorFlow 等机器学习库让神经网络对数百万开发者变得触手可及。但从零开始构建网络有着巨大的价值——你将以使用高级 API 从未有过的方式理解内部原理。<sup class="footnote-ref"><a href="#fn1">[1]</a></sup>

## 感知机：单个神经元

让我们从简单的开始。感知机接收输入，将每个输入乘以权重，求和，然后通过激活函数传递结果：

```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = 0

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.step_function(z)

    def step_function(self, z):
        return 1 if z > 0 else 0
```

<aside id="fn1" class="footnote">
  <p>反向传播算法由 Rumelhart、Hinton 和 Williams 在其1986年的开创性论文中推广，它能高效计算梯度。</p>
</aside>

![神经网络可视化](https://picsum.photos/seed/neural-net/800/400)

## 前向传播

多层网络将感知机链接在一起。每一层的输出成为下一层的输入：

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            )
            self.biases.append(np.zeros(layer_sizes[i+1]))

    def forward(self, X):
        self.activations = [X]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.activations[-1], w) + b
            a = self.sigmoid(z)
            self.activations.append(a)
        return self.activations[-1]
```

<aside id="fn2" class="footnote">
  <p>将权重初始化为小的随机值，可以防止困扰早期神经网络的梯度消失/爆炸问题。</p>
</aside>

## 损失函数

我们需要一种方法来衡量预测的错误程度。交叉熵损失是分类的标准方法：

```python
def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15  # 防止 log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))
```

<aside id="fn3" class="footnote">
  <p>交叉熵损失有一个很好的特性：它对 softmax 输出的梯度就是预测误差。</p>
</aside>

## 反向传播：学习算法

这就是神奇之处。反向传播通过应用链式法则，计算每个权重对误差的贡献程度：

```python
def backward(self, X, y, y_pred):
    m = y.shape[0]
    delta = y_pred - y

    for i in reversed(range(len(self.weights))):
        dw = np.dot(self.activations[i].T, delta) / m
        db = np.sum(delta, axis=0) / m

        if i > 0:
            delta = np.dot(delta, self.weights[i].T)
            delta *= self.sigmoid_derivative(self.activations[i])

        self.weights[i] -= self.learning_rate * dw
        self.biases[i] -= self.learning_rate * db
```

<aside id="fn4" class="footnote">
  <p>链式法则让我们能够高效地逐层计算梯度，从输出反向到输入——这就是"反向"传播的含义。</p>
</aside>

## 训练循环

将所有内容整合到一个完整的训练循环中：

```python
nn = NeuralNetwork([784, 128, 64, 10])
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    y_pred = nn.forward(X_train)
    loss = cross_entropy_loss(y_pred, y_train)
    nn.backward(X_train, y_train, y_pred)

    if epoch % 10 == 0:
        accuracy = np.mean(np.argmax(y_pred, axis=1) ==
                          np.argmax(y_train, axis=1))
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={accuracy:.4f}")
```

<aside id="fn5" class="footnote">
  <p>像 Adam 这样的现代优化器，在基本梯度下降的基础上增加了自适应学习率和动量。</p>
</aside>

## 为何要从零开始？

你可能会问：库都免费提供了，为何还要自己写？以下是收获：

1. **调试直觉** — 当模型出现问题时，你会确切知道去哪里找原因
2. **定制能力** — 新型架构需要理解基础知识
3. **性能** — 生产系统通常需要定制实现
4. **清晰度** — 面试官喜欢能解释反向传播的候选人

## 结论

从零开始构建神经网络让机器学习不再神秘。看似神奇的智能变成了优雅的数学——线性代数、微积分，以及一个简单得令人惊讶的优化循环。

下次你在喜欢的框架中调用 `model.fit()` 时，会对底层发生的事情有更深的理解。
