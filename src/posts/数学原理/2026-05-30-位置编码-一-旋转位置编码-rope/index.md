---
layout: post.njk
post_id: 2026-05-30-位置编码-一-旋转位置编码-rope
archive: 数学原理
title: 位置编码（一）：旋转位置编码 RoPE
date: 2026-05-30
tags:
  - post
---
# RoPE 旋转位置编码：从旋转矩阵到 LLaMA 源码实现

---

# 1. 为什么 Transformer 需要位置编码

Transformer 的 Self-Attention 机制本质上是对一组 token 做两两相关性计算。

给定一个长度为 $N$ 的输入序列：

$$
\mathbb{S}\_{N}=\\{w_i\\}_{i=1}^{N}
$$

其中 $w_i$ 表示第 $i$ 个 token。 

经过 embedding 层后，得到对应的词向量序列：

$$
\mathbb{E}\_N = \\{ \boldsymbol{x}\_i \\}_{i=1}^{N}
$$

其中 $\boldsymbol{x}_i \in \mathbb{R}^{d}$ 表示第 $i$ 个 token 的 $d$ 维词向量。

在 Self-Attention 中，每个 token 的 embedding 会被映射成 query、key、value：

$$
\boldsymbol{q}_m = f_q(\boldsymbol{x}_m, m)
$$

$$
\boldsymbol{k}_n = f_k(\boldsymbol{x}_n, n)
$$

$$
\boldsymbol{v}_n = f_v(\boldsymbol{x}_n, n)
$$

其中：

- $\boldsymbol{q}_m$：位置 $m$ 的 query；
- $\boldsymbol{k}_n$：位置 $n$ 的 key；
- $\boldsymbol{v}_n$：位置 $n$ 的 value；
- $m,n$ 表示 token 的位置。

Self-Attention 的核心计算是：

$$
a_{m,n} =
\frac{
\exp\left(
\frac{\boldsymbol{q}\_m^T \boldsymbol{k}\_n}{\sqrt{d}}
\right)
}{
\sum_{j=1}^{N}
\exp\left(
\frac{\boldsymbol{q}\_m^T \boldsymbol{k}\_j}{\sqrt{d}}
\right)
}
$$


$$
\boldsymbol{o}\_m =
\sum_{n=1}^{N} a_{m,n}\boldsymbol{v}_n
$$

也就是说，attention score 主要由：

$$
\boldsymbol{q}_m^T \boldsymbol{k}_n
$$

决定。

但是，如果没有位置编码，那么 attention 只知道 token 的内容相似度，不知道 token 的顺序。

例如：

```text
我 爱 你
你 爱 我
```

这两个句子的 token 集合类似，但语义不同。原因就在于 token 的顺序不同。

所以 Transformer 必须引入位置编码。

---

# 2. 绝对位置编码与相对位置编码

## 2.1 绝对位置编码

经典 Transformer 使用的是绝对位置编码。

常见做法是在 token embedding 上加一个位置向量：

$$
\boldsymbol{x}_i' = \boldsymbol{x}_i + \boldsymbol{p}_i
$$

然后再计算：

$$
\boldsymbol{q}_i = W_q(\boldsymbol{x}_i + \boldsymbol{p}_i)
$$

$$
\boldsymbol{k}_i = W_k(\boldsymbol{x}_i + \boldsymbol{p}_i)
$$

$$
\boldsymbol{v}_i = W_v(\boldsymbol{x}_i + \boldsymbol{p}_i)
$$

经典 Sinusoidal 位置编码定义为：

$$
\boldsymbol{p}_{i,2t} =
\sin
\left(
\frac{i}{10000^{2t/d}}
\right)
$$

$$
\boldsymbol{p}_{i,2t+1} =
\cos
\left(
\frac{i}{10000^{2t/d}}
\right)
$$

其中：

- $i$ 是 token 的位置；
- $d$ 是 embedding 维度;
- $t$：维度索引的分段参数，取值范围为 $ t = 0, 1, 2, \dots, d/2 - 1 $。


这种方法直接告诉模型“当前 token 在第几个位置”，所以称为绝对位置编码。

---

## 2.2 相对位置编码

在语言模型中，相对位置往往比绝对位置更重要。

例如：

```text
位置 5 和位置 6
位置 100 和位置 101
```

它们的绝对位置不同，但相对距离都是：$ 1 $

对于 attention 来说，query 和 key 之间的相对距离：

$$
m-n
$$

通常比它们各自的绝对位置 $m$、$n$ 更重要。

因此，相对位置编码希望 attention score 显式或隐式依赖：

$$
m-n
$$

RoPE 的目标就是：

> 让 query 和 key 在计算点积时，自然包含相对位置信息 $$m-n$$。

---

# 3. RoPE 的核心思想

RoPE，全称 Rotary Position Embedding，即旋转位置编码。

它的核心思想是：

> 不把位置向量加到 embedding 上，而是根据 token 的位置，对 query 和 key 做旋转变换。

也就是说，对于位置 $m$ 的 query：

$$
\boldsymbol{q}_m
\rightarrow
R_m \boldsymbol{q}_m
$$

对于位置 $n$ 的 key：

$$
\boldsymbol{k}_n
\rightarrow
R_n \boldsymbol{k}_n
$$

然后再计算 attention score：

$$
(R_m\boldsymbol{q}_m)^T(R_n\boldsymbol{k}_n)
$$

经过旋转矩阵的性质变换后，这个点积会变成只与相对位置 $n-m$ 或 $m-n$ 有关的形式。

这就是 RoPE 最重要的性质。

---

# 4. 旋转矩阵基础

## 4.1 二维旋转矩阵

在二维平面中，一个向量绕原点旋转角度 $\theta$，可以用旋转矩阵表示。

常见的列向量旋转矩阵为：

$$
R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

如果有二维列向量：

$$
\boldsymbol{x} =
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}
$$

那么旋转后为：

$$
R(\theta)\boldsymbol{x}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x_0 \\
x_1
\end{bmatrix}
$$

展开得到：

$$
R(\theta)\boldsymbol{x}
=
\begin{bmatrix}
x_0\cos\theta - x_1\sin\theta \\
x_0\sin\theta + x_1\cos\theta
\end{bmatrix}
$$

---

## 4.2 旋转矩阵的重要性质

旋转矩阵有两个非常关键的性质。

### 性质一：连续旋转可以相加

$$
R(\theta_1)R(\theta_2)
=
R(\theta_1+\theta_2)
$$

也就是说，先旋转 $$\theta_1$$，再旋转 $$\theta_2$$，等价于一次性旋转：

$$
\theta_1+\theta_2
$$

---

### 性质二：转置等于反向旋转

由于旋转矩阵是正交矩阵：

$$
R(\theta)^T R(\theta) = I
$$

所以：

$$
R(\theta)^T = R(\theta)^{-1}
$$

而旋转角度 $$-\theta$$ 正好是 $$\theta$$ 的逆旋转，因此：

$$
R(\theta)^T = R(-\theta)
$$

这两个性质是 RoPE 能够引入相对位置的数学基础。

---

# 5. 二维 RoPE 推导

为了理解 RoPE，可以先从二维情况开始。

假设 query 和 key 都是二维向量：

$$
\boldsymbol{q}_m =
\begin{bmatrix}
q_m^{(1)} \\
q_m^{(2)}
\end{bmatrix}
$$

$$
\boldsymbol{k}_n =
\begin{bmatrix}
k_n^{(1)} \\
k_n^{(2)}
\end{bmatrix}
$$

RoPE 对位置 $$m$$ 的 query 做旋转：

$$
f_q(\boldsymbol{x}_m,m)
=
R(m\theta)\boldsymbol{q}_m
$$

对位置 $$n$$ 的 key 做旋转：

$$
f_k(\boldsymbol{x}_n,n)
=
R(n\theta)\boldsymbol{k}_n
$$

其中：

$$
\theta
$$

是旋转频率。

二维旋转后的 query 为：

$$
R(m\theta)\boldsymbol{q}_m
=
\begin{bmatrix}
\cos m\theta & -\sin m\theta \\
\sin m\theta & \cos m\theta
\end{bmatrix}
\begin{bmatrix}
q_m^{(1)} \\
q_m^{(2)}
\end{bmatrix}
$$

展开：

$$
=
\begin{bmatrix}
q_m^{(1)}\cos m\theta - q_m^{(2)}\sin m\theta \\
q_m^{(1)}\sin m\theta + q_m^{(2)}\cos m\theta
\end{bmatrix}
$$

key 同理：

$$
R(n\theta)\boldsymbol{k}_n
=
\begin{bmatrix}
k_n^{(1)}\cos n\theta - k_n^{(2)}\sin n\theta \\
k_n^{(1)}\sin n\theta + k_n^{(2)}\cos n\theta
\end{bmatrix}
$$

这就是“旋转位置编码”名字的来源：

> 位置编码不是加法，而是对 query 和 key 进行旋转。

---

# 6. 从二维 RoPE 扩展到多维 RoPE

真实模型中的 query 和 key 通常是高维向量。

假设 head dimension 为 $$d$$，并且 $$d$$ 是偶数：

$$
\boldsymbol{x}
=
[x_0,x_1,x_2,x_3,\cdots,x_{d-2},x_{d-1}]^T
$$

RoPE 的做法是：

> 每两个维度为一组，在每个二维子空间中做旋转。

即：

```text
(x0, x1) 使用频率 θ0
(x2, x3) 使用频率 θ1
(x4, x5) 使用频率 θ2
...
(xd-2, xd-1) 使用频率 θd/2-1
```

---

## 6.1 多维旋转矩阵

多维 RoPE 的旋转矩阵是一个块对角矩阵：

$$
R_{\Theta,m}^{d}
=
\begin{bmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{\frac d2-1} & -\sin m\theta_{\frac d2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{\frac d2-1} & \cos m\theta_{\frac d2-1}
\end{bmatrix}
$$

其中频率集合为：

$$
\Theta
=
\left\{
\theta_i = 10000^{-2i/d},
i = 0,1,\cdots,\frac d2 - 1
\right\}
$$

也可以写成：

$$
\theta_i
=
\frac{1}{10000^{2i/d}}
$$

---

## 6.2 为什么不同维度使用不同频率？

RoPE 沿用了 Sinusoidal 位置编码中的频率设计：

$$
\theta_i
=
\frac{1}{10000^{2i/d}}
$$

不同维度对应不同频率：

- 低维频率较高，适合捕捉短距离位置变化；
- 高维频率较低，适合捕捉长距离位置变化。

这样可以让模型同时感知短程关系和长程关系。

---

# 7. RoPE 为什么天然包含相对位置信息

这是 RoPE 最核心的部分。

对于位置 $$m$$ 的 query：

$$
\tilde{\boldsymbol{q}}_m
=
R_m \boldsymbol{q}_m
$$

对于位置 $$n$$ 的 key：

$$
\tilde{\boldsymbol{k}}_n
=
R_n \boldsymbol{k}_n
$$

其中：

$$
R_m = R(m\theta)
$$

$$
R_n = R(n\theta)
$$

attention score 为：

$$
\tilde{\boldsymbol{q}}_m^T \tilde{\boldsymbol{k}}_n
=
(R_m\boldsymbol{q}_m)^T(R_n\boldsymbol{k}_n)
$$

展开：

$$
=
\boldsymbol{q}_m^T R_m^T R_n \boldsymbol{k}_n
$$

由于：

$$
R_m^T = R(-m\theta)
$$

所以：

$$
R_m^T R_n
=
R(-m\theta)R(n\theta)
$$

根据旋转矩阵相加性质：

$$
R(-m\theta)R(n\theta)
=
R((n-m)\theta)
$$

因此：

$$
\tilde{\boldsymbol{q}}_m^T \tilde{\boldsymbol{k}}_n
=
\boldsymbol{q}_m^T R((n-m)\theta)\boldsymbol{k}_n
$$

这说明 RoPE 之后的 attention score 只依赖相对位置：

$$
n-m
$$

而不是单独依赖绝对位置 $$m$$ 和 $$n$$。

所以 RoPE 虽然使用绝对位置生成旋转角度，但最终在 attention score 中体现为相对位置编码。

---

# 8. RoPE 的高效计算形式

如果直接构造完整的旋转矩阵：

$$
R_{\Theta,m}^{d}
$$

再和 query、key 相乘，计算量和内存开销都很大。

但 RoPE 的旋转矩阵是稀疏块对角矩阵，因此可以高效实现。

对于向量：

$$
\boldsymbol{x}
=
[x_0,x_1,x_2,x_3,\cdots,x_{d-2},x_{d-1}]^T
$$

RoPE 旋转结果为：

$$
R_{\Theta,m}^{d}\boldsymbol{x}
=
\begin{bmatrix}
x_0\cos m\theta_0 - x_1\sin m\theta_0 \\
x_0\sin m\theta_0 + x_1\cos m\theta_0 \\
x_2\cos m\theta_1 - x_3\sin m\theta_1 \\
x_2\sin m\theta_1 + x_3\cos m\theta_1 \\
\vdots \\
x_{d-2}\cos m\theta_{\frac d2-1} - x_{d-1}\sin m\theta_{\frac d2-1} \\
x_{d-2}\sin m\theta_{\frac d2-1} + x_{d-1}\cos m\theta_{\frac d2-1}
\end{bmatrix}
$$

可以写成逐元素运算：

$$
R_{\Theta,m}^{d}\boldsymbol{x}
=
\boldsymbol{x}\odot \boldsymbol{\cos}_m
+
\operatorname{rotate}(\boldsymbol{x})\odot \boldsymbol{\sin}_m
$$

其中：

$$
\boldsymbol{\cos}_m
=
[
\cos m\theta_0,
\cos m\theta_0,
\cos m\theta_1,
\cos m\theta_1,
\cdots,
\cos m\theta_{\frac d2-1},
\cos m\theta_{\frac d2-1}
]
$$

$$
\boldsymbol{\sin}_m
=
[
\sin m\theta_0,
\sin m\theta_0,
\sin m\theta_1,
\sin m\theta_1,
\cdots,
\sin m\theta_{\frac d2-1},
\sin m\theta_{\frac d2-1}
]
$$

并且：

$$
\operatorname{rotate}(\boldsymbol{x})
=
[-x_1,x_0,-x_3,x_2,\cdots,-x_{d-1},x_{d-2}]
$$

这就是 RoPE 的高效实现形式。

---

# 9. RoPE 的远程衰减特性

RoPE 不仅能表达相对位置，还具有一定的远程衰减特性。

将 query 和 key 两两分组后，可以用复数形式表示 RoPE 后的内积：

$$
\left(
R_{\Theta,m}^{d}\boldsymbol{q}
\right)^T
\left(
R_{\Theta,n}^{d}\boldsymbol{k}
\right)
=
\operatorname{Re}
\left[
\sum_{i=0}^{d/2-1}
\boldsymbol{q}_{[2i:2i+1]}
\boldsymbol{k}_{[2i:2i+1]}^{*}
e^{\mathrm{i}(m-n)\theta_i}
\right]
$$

其中：

- $$\operatorname{Re}$$ 表示取复数实部；
- $$*$$ 表示复共轭；
- $$\theta_i = 10000^{-2i/d}$$。

当相对距离：

$$
|m-n|
$$

变大时，不同频率的旋转项：

$$
e^{\mathrm{i}(m-n)\theta_i}
$$

会产生不同相位的振荡，相互之间可能抵消，使得整体内积存在衰减趋势。

这意味着：

> 距离越远的 token，其 attention score 往往会受到一定程度的自然削弱。

这对语言模型是合理的，因为大多数情况下，相邻或较近 token 之间的关系更强。

当然，这种衰减不是硬性的距离截断，而是一种由频率设计带来的软衰减。

---

# 10. Hugging Face LLaMA 中 RoPE 代码分析

下面进入代码分析部分。

特别说明：本节先给出完整代码，然后再逐个分析。

这里展示的是 Hugging Face LLaMA 中 RoPE 相关逻辑的典型实现形式，主要包括：

1. `LlamaRotaryEmbedding`
2. `rotate_half`
3. `apply_rotary_pos_emb`

---

## 10.1 完整代码

```python
import torch
from torch import nn


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        if seq_len is not None:
            # 在新版 transformers 中，seq_len 参数已经逐渐废弃
            pass

        # x: [batch_size, num_attention_heads, seq_len, head_dim]
        # position_ids: [batch_size, seq_len]

        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
        )

        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) else "cpu"

        # 强制使用 float32 计算，避免 bf16 在长上下文下精度不足
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float()
                @ position_ids_expanded.float()
            ).transpose(1, 2)

            emb = torch.cat((freqs, freqs), dim=-1)

            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    Rotates half the hidden dims of the input.

    输入:
        x: [..., head_dim]

    输出:
        [-x2, x1]

    如果 x = [x1, x2]，其中 x1 和 x2 分别是前半部分和后半部分，
    那么 rotate_half(x) = [-x2, x1]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,
    unsqueeze_dim=1,
):
    """
    Applies Rotary Position Embedding to the query and key tensors.

    参数:
        q: [batch_size, num_heads, seq_len, head_dim]
        k: [batch_size, num_key_value_heads, seq_len, head_dim]
        cos: [batch_size, seq_len, head_dim]
        sin: [batch_size, seq_len, head_dim]
        position_ids: 兼容旧版本接口，当前实现中通常不再使用
        unsqueeze_dim: 用于扩展 cos 和 sin 的维度，使其可以和 q、k 广播

    返回:
        q_embed: 旋转位置编码后的 query
        k_embed: 旋转位置编码后的 key
    """

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

---

## 10.2 `LlamaRotaryEmbedding.__init__` 分析

先看初始化部分：

```python
class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        device=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
```

这里的几个参数含义是：

| 参数 | 含义 |
|---|---|
| `dim` | 每个 attention head 的维度，也就是 `head_dim` |
| `max_position_embeddings` | 最大位置长度 |
| `base` | RoPE 中频率计算的底数，通常为 `10000.0` |
| `device` | 张量所在设备，如 CPU 或 GPU |

RoPE 的频率定义为：

$$
\theta_i
=
\frac{1}{10000^{2i/d}}
$$

在代码中，`base` 就对应公式里的：

$$
10000
$$

---

## 10.3 `inv_freq` 的计算

代码：

```python
inv_freq = 1.0 / (
    self.base
    ** (
        torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
        / self.dim
    )
)
```

先看：

```python
torch.arange(0, self.dim, 2)
```

如果：

```python
self.dim = 8
```

那么得到：

```python
[0, 2, 4, 6]
```

除以 `self.dim`：

```python
[0/8, 2/8, 4/8, 6/8]
```

对应数学中的：

$$
\frac{2i}{d}
$$

因此：

```python
self.base ** (torch.arange(0, self.dim, 2) / self.dim)
```

对应：

$$
10000^{2i/d}
$$

最后取倒数：

```python
1.0 / (...)
```

得到：

$$
\theta_i
=
\frac{1}{10000^{2i/d}}
$$

也就是：

$$
\theta_i = 10000^{-2i/d}
$$

所以 `inv_freq` 保存的是 RoPE 每一组二维旋转对应的频率。

---

## 10.4 为什么叫 `inv_freq`

`inv_freq` 的意思是 inverse frequency，即“频率倒数”或“逆频率”。

但在 RoPE 代码中，它实际表示的是：

$$
\theta_i
=
\frac{1}{\text{base}^{2i/d}}
$$

后续会用 position id 乘以它：

$$
m\theta_i
$$

从而得到位置 $$m$$ 在第 $$i$$ 个二维子空间中的旋转角度。

---

## 10.5 `register_buffer` 的作用

代码：

```python
self.register_buffer("inv_freq", inv_freq, persistent=False)
```

这行代码表示：

- `inv_freq` 是模型的一部分；
- 但它不是可训练参数；
- 它会随着模型移动到 GPU 或 CPU；
- `persistent=False` 表示它不一定会被保存到 `state_dict` 中。

为什么 `inv_freq` 不是可训练参数？

因为 RoPE 默认固定使用：

$$
\theta_i = 10000^{-2i/d}
$$

虽然论文中也尝试过让 $$\theta_i$$ 可训练，但实验发现它通常不会发生明显变化，所以实践中一般固定。

---

## 10.6 `forward` 输入分析

代码：

```python
def forward(self, x, position_ids, seq_len=None):
```

主要输入：

```python
x
```

通常是 query 或 key 对应的张量，用来提供 dtype 和 device。

它的形状一般是：

```python
[batch_size, num_attention_heads, seq_len, head_dim]
```

即：

$$
[B,H,L,D]
$$

其中：

| 符号 | 含义 |
|---|---|
| $$B$$ | batch size |
| $$H$$ | attention head 数 |
| $$L$$ | sequence length |
| $$D$$ | head dimension |

另一个输入：

```python
position_ids
```

形状通常是：

```python
[batch_size, seq_len]
```

例如：

```python
position_ids = [
    [0, 1, 2, 3, 4]
]
```

表示序列中每个 token 的位置。

---

## 10.7 扩展 `inv_freq`

代码：

```python
inv_freq_expanded = (
    self.inv_freq[None, :, None]
    .float()
    .expand(position_ids.shape[0], -1, 1)
)
```

假设：

```python
head_dim = 8
```

那么：

```python
self.inv_freq.shape = [4]
```

即：

```python
[theta_0, theta_1, theta_2, theta_3]
```

经过：

```python
self.inv_freq[None, :, None]
```

形状变成：

```python
[1, head_dim // 2, 1]
```

再通过：

```python
.expand(position_ids.shape[0], -1, 1)
```

扩展成：

```python
[batch_size, head_dim // 2, 1]
```

也就是：

$$
[B,D/2,1]
$$

---

## 10.8 扩展 `position_ids`

代码：

```python
position_ids_expanded = position_ids[:, None, :].float()
```

原始：

```python
position_ids.shape = [batch_size, seq_len]
```

扩展后：

```python
position_ids_expanded.shape = [batch_size, 1, seq_len]
```

即：

$$
[B,1,L]
$$

---

## 10.9 计算旋转角度 `freqs`

代码：

```python
freqs = (
    inv_freq_expanded.float()
    @ position_ids_expanded.float()
).transpose(1, 2)
```

这里做的是矩阵乘法：

```python
[B, D/2, 1] @ [B, 1, L]
```

得到：

```python
[B, D/2, L]
```

然后：

```python
.transpose(1, 2)
```

变成：

```python
[B, L, D/2]
```

数学上，这一步计算的是：

$$
\text{freqs}_{m,i}
=
m\theta_i
$$

也就是：

$$
m \cdot \frac{1}{10000^{2i/d}}
$$

其中：

- $$m$$ 是 token 的位置；
- $$i$$ 是二维旋转组编号。

---

## 10.10 为什么使用 float32 计算

代码：

```python
with torch.autocast(device_type=device_type, enabled=False):
```

这表示在这一段中关闭自动混合精度，强制使用 float32 计算。

原因是：

> 在长上下文场景中，位置 id 可能很大，如果使用 bf16 或 fp16，计算 $$m\theta_i$$ 时可能出现精度问题。

所以 Hugging Face 的实现中，会特意让 RoPE 的角度计算保持在 float32。

---

## 10.11 拼接 `freqs`

代码：

```python
emb = torch.cat((freqs, freqs), dim=-1)
```

如果：

```python
freqs = [mθ0, mθ1, mθ2, mθ3]
```

那么拼接后：

```python
emb = [mθ0, mθ1, mθ2, mθ3, mθ0, mθ1, mθ2, mθ3]
```

形状从：

```python
[B, L, D/2]
```

变成：

```python
[B, L, D]
```

这样做是为了和 query、key 的最后一维 `head_dim` 对齐。

---

## 10.12 计算 `cos` 和 `sin`

代码：

```python
cos = emb.cos()
sin = emb.sin()
```

得到：

$$
\cos(m\theta_i)
$$

和：

$$
\sin(m\theta_i)
$$

最后返回：

```python
return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
```

将 `cos` 和 `sin` 转回和输入 `x` 一样的数据类型，比如 fp16 或 bf16。

---

## 10.13 `rotate_half` 函数分析

代码：

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    return torch.cat((-x2, x1), dim=-1)
```

它做的事情是：

```python
[x1, x2] -> [-x2, x1]
```

假设：

```python
x = [a, b, c, d]
```

那么：

```python
x1 = [a, b]
x2 = [c, d]
```

返回：

```python
[-c, -d, a, b]
```

在 Hugging Face LLaMA 的实现中，维度配对方式是：

```text
第 0 维 和 第 D/2 维配对
第 1 维 和 第 D/2 + 1 维配对
第 2 维 和 第 D/2 + 2 维配对
...
```

而不是视频中直观展示的：

```text
第 0 维 和 第 1 维配对
第 2 维 和 第 3 维配对
...
```

这两种方式本质相同，都是在二维子空间中做旋转，只是维度排列方式不同。

---

## 10.14 `rotate_half` 对应的数学形式

假设向量被分成前后两半：

$$
\boldsymbol{x}
=
[\boldsymbol{x}_1,\boldsymbol{x}_2]
$$

其中：

$$
\boldsymbol{x}_1
=
[x_0,x_1,\cdots,x_{D/2-1}]
$$

$$
\boldsymbol{x}_2
=
[x_{D/2},x_{D/2+1},\cdots,x_{D-1}]
$$

则：

$$
\operatorname{rotate\_half}(\boldsymbol{x})
=
[-\boldsymbol{x}_2,\boldsymbol{x}_1]
$$

对应二维旋转中的正弦项：

$$
[-y,x]
$$

因为二维旋转可以写成：

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
x\cos\theta - y\sin\theta \\
x\sin\theta + y\cos\theta
\end{bmatrix}
$$

也就是：

$$
[x',y']
=
[x,y]\cos\theta
+
[-y,x]\sin\theta
$$

---

## 10.15 `apply_rotary_pos_emb` 函数分析

完整代码：

```python
def apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids=None,
    unsqueeze_dim=1,
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed
```

这个函数是真正把 RoPE 应用到 query 和 key 上的地方。

---

## 10.16 为什么要 `unsqueeze`

原始 `cos` 和 `sin` 的形状是：

```python
[batch_size, seq_len, head_dim]
```

即：

$$
[B,L,D]
$$

而 query 的形状是：

```python
[batch_size, num_heads, seq_len, head_dim]
```

即：

$$
[B,H,L,D]
$$

为了让 `cos`、`sin` 可以和 query 广播相乘，需要增加一个维度：

```python
cos = cos.unsqueeze(1)
sin = sin.unsqueeze(1)
```

变成：

```python
[batch_size, 1, seq_len, head_dim]
```

即：

$$
[B,1,L,D]
$$

这样就可以广播到：

$$
[B,H,L,D]
$$

---

## 10.17 应用 RoPE 到 query

代码：

```python
q_embed = (q * cos) + (rotate_half(q) * sin)
```

数学形式：

$$
\tilde{\boldsymbol{q}}
=
\boldsymbol{q}\odot \cos
+
\operatorname{rotate\_half}(\boldsymbol{q})\odot \sin
$$

这正是 RoPE 的高效计算公式。

---

## 10.18 应用 RoPE 到 key

代码：

```python
k_embed = (k * cos) + (rotate_half(k) * sin)
```

数学形式：

$$
\tilde{\boldsymbol{k}}
=
\boldsymbol{k}\odot \cos
+
\operatorname{rotate\_half}(\boldsymbol{k})\odot \sin
$$

注意：

> RoPE 通常只作用在 query 和 key 上，不作用在 value 上。

原因是 attention score 由：

$$
QK^T
$$

决定。

位置关系应该影响 token 之间的相关性，而相关性正是通过 query 和 key 的点积体现的。

---

## 10.19 一个小例子理解 Hugging Face 的实现

假设：

$$
\boldsymbol{x}
=
[x_0,x_1,x_2,x_3]
$$

Hugging Face LLaMA 中会将其看作：

$$
\boldsymbol{x}_1 = [x_0,x_1]
$$

$$
\boldsymbol{x}_2 = [x_2,x_3]
$$

于是：

$$
\operatorname{rotate\_half}(\boldsymbol{x})
=
[-x_2,-x_3,x_0,x_1]
$$

假设：

$$
\cos =
[\cos\theta_0,\cos\theta_1,\cos\theta_0,\cos\theta_1]
$$

$$
\sin =
[\sin\theta_0,\sin\theta_1,\sin\theta_0,\sin\theta_1]
$$

那么：

$$
\boldsymbol{x}_{rope}
=
\boldsymbol{x}\odot\cos
+
\operatorname{rotate\_half}(\boldsymbol{x})\odot\sin
$$

展开为：

$$
[
x_0\cos\theta_0 - x_2\sin\theta_0,
x_1\cos\theta_1 - x_3\sin\theta_1,
x_2\cos\theta_0 + x_0\sin\theta_0,
x_3\cos\theta_1 + x_1\sin\theta_1
]
$$

可以看到，它是在如下二维平面中旋转：

```text
(x0, x2)
(x1, x3)
```

而不是：

```text
(x0, x1)
(x2, x3)
```

但数学本质一样。

---

## 10.20 与复数实现的关系

博客园文章中还提到一种 Meta LLaMA 早期实现中常见的复数写法：

```python
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2).float() / dim)
    )

    t = torch.arange(seq_len, device=freqs.device)

    freqs = torch.outer(t, freqs).float()

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

这个实现把每两个实数维度看成一个复数：

$$
x_0 + ix_1
$$

然后乘以：

$$
e^{im\theta}
=
\cos m\theta + i\sin m\theta
$$

复数乘法：

$$
(x_0 + ix_1)(\cos m\theta + i\sin m\theta)
$$

展开：

$$
=
(x_0\cos m\theta - x_1\sin m\theta)
+
i(x_0\sin m\theta + x_1\cos m\theta)
$$

这正好就是二维旋转：

$$
\begin{bmatrix}
x_0' \\
x_1'
\end{bmatrix}
=
\begin{bmatrix}
x_0\cos m\theta - x_1\sin m\theta \\
x_0\sin m\theta + x_1\cos m\theta
\end{bmatrix}
$$

所以：

> RoPE 的复数实现和 `cos/sin + rotate_half` 实现，本质上是等价的。

---

# 11. RoPE 整体流程总结

在 LLaMA 这样的模型中，RoPE 的整体流程可以总结为：

```text
输入 token
   ↓
Embedding
   ↓
线性映射得到 Q、K、V
   ↓
根据 position_ids 计算每个位置的旋转角度
   ↓
生成 cos 和 sin
   ↓
对 Q 和 K 应用旋转：
q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
   ↓
计算 attention score：
q_embed @ k_embed.T
   ↓
Softmax
   ↓
加权求和 V
```

用公式表示：

$$
\boldsymbol{q}_m
=
W_q\boldsymbol{x}_m
$$

$$
\boldsymbol{k}_n
=
W_k\boldsymbol{x}_n
$$

RoPE 后：

$$
\tilde{\boldsymbol{q}}_m
=
R_{\Theta,m}^{d}\boldsymbol{q}_m
$$

$$
\tilde{\boldsymbol{k}}_n
=
R_{\Theta,n}^{d}\boldsymbol{k}_n
$$

attention score：

$$
\tilde{\boldsymbol{q}}_m^T
\tilde{\boldsymbol{k}}_n
=
\boldsymbol{q}_m^T
\left(
R_{\Theta,m}^{d}
\right)^T
R_{\Theta,n}^{d}
\boldsymbol{k}_n
$$

由于：

$$
\left(
R_{\Theta,m}^{d}
\right)^T
R_{\Theta,n}^{d}
=
R_{\Theta,n-m}^{d}
$$

所以：

$$
\tilde{\boldsymbol{q}}_m^T
\tilde{\boldsymbol{k}}_n
=
\boldsymbol{q}_m^T
R_{\Theta,n-m}^{d}
\boldsymbol{k}_n
$$

这说明最终 attention score 中包含的是相对位置信息：

$$
n-m
$$

---

# 12. 一句话总结

RoPE 的本质是：

> 将 query 和 key 按维度两两分组，在每个二维子空间中根据 token 位置旋转不同角度；虽然旋转角度来自绝对位置，但 query 和 key 点积后会自然转化为相对位置差，因此 RoPE 既能高效注入位置信息，又天然适合相对位置建模。

---

# 参考资料

[^1]: [你还不懂旋转位置编码吗？_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1F1421B7iv/?spm_id_from=333.1387.search.video_card.click&vd_source=e98b669ccbafff4b5aa59dd6303b722f)

[^2]: [十分钟读懂旋转编码（RoPE）](https://www.cnblogs.com/gongzb/p/19069768)
