---
layout: post.njk
post_id: 2026-10-27-大模型中的数学-004-从位置到旋转-一步步理解-rope
archive: 数学原理
title: 大模型中的数学（004）：从位置到旋转，一步步理解 RoPE
date: 2026-10-27
updated: 2026-10-27
tags:
  - post
---
# 从位置到旋转：一步步理解 RoPE

在介绍 Transformer 的时候，我们经常会说模型需要知道词语的先后顺序。但具体来说，位置应该怎样进入注意力计算？如果我们希望模型同时关注词语内容和它们之间的距离，又该如何设计？本文从这两个问题出发，结合具体数字、旋转矩阵和代码，逐步理解 **RoPE（Rotary Position Embedding，旋转位置编码）**。

本文的讲解顺序参考了苏剑林的[《博采众长的旋转式位置编码》](https://spaces.ac.cn/archives/8265)，数学定义亦可参见 [RoFormer 论文](https://arxiv.org/abs/2104.09864)。

## 一、从注意力中的位置说起

### 1.1 注意力分数是什么

假设序列中第 $m$ 个 token 的查询向量是 **$q_m$**，第 $n$ 个 token 的键向量是 **$k_n$**。先忽略缩放和掩码，它们之间的注意力分数由点积给出：

$$
s_{m,n}=q_m^{\mathsf T}k_n.
$$

点积体现两个向量之间的关系，但这个公式本身没有直接写出位置 $m$ 和 $n$。为了让计算利用顺序，我们需要把位置信息引入 Q 和 K。

### 1.2 我们希望得到什么

假设先把内容向量固定为 $q$ 和 $k$，我们希望位置对点积的影响与**相对位置** $n-m$ 有关。比如，位置对 $(0,1)$ 和 $(1,2)$ 的相对位置都是 $1$，如果使用相同的 $q,k$，两组位置带来的结果就应相同。

这里的“固定内容”是关键。真实序列中不同 token 的 Q、K 通常不同，所以即使相对位置相同，最终分数也不一定相同。

RoPE 的思路是：**分别按照 Q 和 K 的绝对位置旋转向量，再利用点积让相对位置自然出现**。

## 二、先认识二维旋转

### 2.1 一个向量如何旋转

将二维向量 $(x,y)$ 逆时针旋转角度 $\alpha$，可以写成：

$$
\begin{pmatrix}x'\\y'\end{pmatrix}
=\underbrace{\begin{pmatrix}\cos\alpha&-\sin\alpha\\\sin\alpha&\cos\alpha\end{pmatrix}}_{R(\alpha)}
\begin{pmatrix}x\\y\end{pmatrix}.
$$

展开矩阵乘法，就得到代码需要使用的两行公式：

$$
x'=x\cos\alpha-y\sin\alpha,\qquad
y'=x\sin\alpha+y\cos\alpha.
$$

为什么它表示旋转？可以看两个坐标轴的方向：$(1,0)$ 经过变换变成 $(\cos\alpha,\sin\alpha)$，恰好是转过角度 $\alpha$ 后的单位向量；$(0,1)$ 变成 $(-\sin\alpha,\cos\alpha)$。任意向量都能拆成这两个方向的组合，因此它们共同旋转，就得到整个向量的旋转。

### 2.2 代入一个数字

假设原向量为 $(1,0)$，旋转 **1 弧度**。由于 $\cos(1)\approx0.5403$、$\sin(1)\approx0.8415$，于是

$$
\begin{aligned}
x'&=1\times0.5403-0\times0.8415=0.5403,\\
y'&=1\times0.8415+0\times0.5403=0.8415.
\end{aligned}
$$

所以 $(1,0)$ 旋转后约为 **$(0.5403,0.8415)$**。弧度是角度单位，$1$ 弧度约等于 $57.3^\circ$；程序中的正弦和余弦使用弧度。

## 三、旋转如何产生相对位置

### 3.1 先看一个性质

对于同一个频率下的二维旋转，记位置 $m$ 对应的旋转为 $R_m$。如果位置每增加 1 就旋转 $\theta$ 弧度，那么位置 $m$ 旋转的角度就是 $m\theta$。

旋转矩阵有两个性质：转置相当于反向旋转，连续旋转则可以把角度相加。因此：

$$
R_m^{\mathsf T}R_n=R_{n-m}.
$$

先按位置 $m$ 反向旋转，再按位置 $n$ 正向旋转，效果就是按相对位置 $n-m$ 旋转。

### 3.2 放到 Q、K 的点积中

对固定内容向量 $q,k$ 分别施加位置旋转，得到 $R_mq$ 和 $R_nk$。旋转后的点积是：

$$
\begin{aligned}
(R_mq)^{\mathsf T}(R_nk)
&=q^{\mathsf T}R_m^{\mathsf T}R_nk\\
&=q^{\mathsf T}R_{n-m}k.
\end{aligned}
$$

这便是 RoPE 的关键：**旋转各自使用绝对位置，但在固定内容向量的条件下，点积中的位置作用只与相对位置有关**。这个结论也适用于后面介绍的多组二维旋转。（来源：[RoFormer 论文](https://arxiv.org/abs/2104.09864)）

### 3.3 数字验证

令原始 $q=k=(1,0)$，并令每增加一个位置旋转 1 弧度。旋转前的点积为 $1$。旋转后：

| Q 的位置 $m$ | K 的位置 $n$ | 相对位置 $n-m$ | 旋转后的点积（约值） |
|---:|---:|---:|---:|
| 0 | 1 | 1 | 0.5403 |
| 1 | 2 | 1 | 0.5403 |
| 0 | 2 | 2 | -0.4161 |

例如第一行，Q 不旋转，K 旋转 1 弧度：

$$
(1,0)\cdot(0.5403,0.8415)=0.5403.
$$

第二行的两者分别旋转 1 和 2 弧度，点积约为 $0.5403\times(-0.4161)+0.8415\times0.9093\approx0.5403$。前两行相同，是因为原始向量相同、相对位置也相同。

注意，**旋转前后点积通常会改变**：这里从 $1$ 变成了约 $0.5403$。RoPE 保证的是上述相对位置性质，而不是对不同角度旋转的 Q、K 保持原点积。

## 四、从二维推广到多维

### 4.1 两两分组与不同频率

实际的 Q、K 向量通常有很多维。对于偶数维 $d$，可以把相邻维度分成 $(x_0,x_1)$、$(x_2,x_3)$ 等二维组，分别旋转。第 $i$ 组的频率常取：

$$
\theta_i=10000^{-2i/d},\qquad i=0,1,\ldots,d/2-1.
$$

位置为 $m$ 时，第 $i$ 组的旋转角度就是 **$m\theta_i$**。不同组的频率不同，因此能以不同尺度变化。完整的 $R_m$ 可以看作由这些二维旋转组成的分块矩阵，每一组都满足上一节的相对位置关系。（来源：[科学空间：旋转式位置编码](https://spaces.ac.cn/archives/8265)）

### 4.2 手算四维向量

取 $d=4$，位置 $m=1$，输入向量为 $(1,0,1,0)$。拆分为 $(1,0)$ 和 $(1,0)$：

| 维度组 | 频率 $\theta_i$ | 位置 1 的角度 | 旋转后（约值） |
|---|---:|---:|---|
| 第 0、1 维 | $10000^0=1$ | 1 弧度 | $(0.5403,0.8415)$ |
| 第 2、3 维 | $10000^{-1/2}=0.01$ | 0.01 弧度 | $(0.99995,0.0100)$ |

两组按原来的顺序拼回去：

$$
(1,0,1,0)\longrightarrow(0.5403,0.8415,0.99995,0.0100).
$$

位置为 0 时，各组角度都是 0，因此旋转结果就是原向量。

## 五、PyTorch 实现

下面将刚才的步骤写成代码，采用**相邻维度配对**。输入形状约定为 `[batch, heads, seq_len, head_dim]`：分别表示批次大小、注意力头数、序列长度和每个头的维度。`head_dim` 必须为偶数。

```python
import torch


def apply_rope(
    x: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    base: float = 10000.0,
) -> torch.Tensor:
    """对 Q 或 K 应用 RoPE，返回与输入形状和 dtype 相同的张量。

    x: [batch, heads, seq_len, head_dim]，浮点张量。
    position_ids: [seq_len] 或 [batch, seq_len]；默认从 0 开始。
    """
    if x.ndim != 4:
        raise ValueError("x 必须是 [batch, heads, seq_len, head_dim]")
    if not x.is_floating_point():
        raise ValueError("x 必须是浮点张量")

    batch, _, seq_len, head_dim = x.shape
    if head_dim == 0 or head_dim % 2 != 0:
        raise ValueError("head_dim 必须是正偶数")
    if base <= 0:
        raise ValueError("base 必须大于 0")

    # 1. 拆分相邻维度，例如 [a, b, c, d] -> [a, c] 和 [b, d]。
    # 使用 float32 计算，最后再转回输入 dtype。
    x_float = x.float()
    x_even = x_float[..., 0::2]  # 下标 0、2、4...
    x_odd = x_float[..., 1::2]   # 下标 1、3、5...

    # 2. 计算每组的频率。d=4 时 dim_indices=[0, 2]，
    # 因而 inv_freq=[1, 0.01]。
    dim_indices = torch.arange(
        0, head_dim, 2, device=x.device, dtype=torch.float32
    )
    inv_freq = base ** (-dim_indices / head_dim)  # [head_dim / 2]

    # 默认位置为 0 到 seq_len - 1；也可以由调用者指定。
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=x.device)
    position_ids = position_ids.to(device=x.device, dtype=torch.float32)

    if position_ids.ndim == 1:
        if position_ids.shape[0] != seq_len:
            raise ValueError("一维 position_ids 的长度必须等于 seq_len")
        position_ids = position_ids.unsqueeze(0)  # [1, seq_len]
    elif position_ids.ndim == 2:
        if position_ids.shape not in ((1, seq_len), (batch, seq_len)):
            raise ValueError("position_ids 必须为 [1, seq_len] 或 [batch, seq_len]")
    else:
        raise ValueError("position_ids 必须是一维或二维")

    # 位置 × 频率，得到每个位置和每组维度的角度。
    # [batch 或 1, seq_len, 1] × [head_dim / 2]
    # -> [batch 或 1, seq_len, head_dim / 2]
    angles = position_ids.unsqueeze(-1) * inv_freq

    # 3. 添加 heads 维度，让同一位置的角度广播到各个头。
    # cos、sin 的形状：[batch 或 1, 1, seq_len, head_dim / 2]
    cos = angles.cos().unsqueeze(1)
    sin = angles.sin().unsqueeze(1)

    # 4. 对每一对数字使用二维旋转公式。
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    # 5. [[a', b'], [c', d']] -> [a', b', c', d']。
    paired = torch.stack((rotated_even, rotated_odd), dim=-1)
    out = paired.flatten(-2)
    return out.to(dtype=x.dtype)
```

### 5.1 运行四维示例

将下面的代码接在函数后面执行：

```python
# 1 个样本、1 个头、1 个 token、4 个维度。
x = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
position_ids = torch.tensor([1])  # 这个 token 位于位置 1

print(apply_rope(x, position_ids))
```

输出约为：

```text
tensor([[[[0.5403, 0.8415, 1.0000, 0.0100]]]])
```

第三个数字实际约为 $0.99995$，显示时保留四位小数为 `1.0000`。

### 5.2 放到注意力里

```python
# 示例中 Q、K、V 用随机数表示；实际模型中它们来自输入的线性投影。
batch, heads, seq_len, head_dim = 2, 8, 128, 64
q = torch.randn(batch, heads, seq_len, head_dim)
k = torch.randn(batch, heads, seq_len, head_dim)
v = torch.randn(batch, heads, seq_len, head_dim)

# 分别对 Q、K 使用对应 token 的位置。
q = apply_rope(q)
k = apply_rope(k)

# 计算缩放点积注意力。
scores = q @ k.transpose(-2, -1) / (head_dim ** 0.5)
weights = torch.softmax(scores, dim=-1)
output = weights @ v
```

这里为了突出 RoPE，省略了因果掩码等完整模型可能需要的步骤。

## 六、使用时容易混淆的地方

1. **相对位置相同不意味着所有分数相同。** 还要看原始 Q、K 是否相同；真实 token 的内容向量通常不同。
2. **点积通常会因旋转而改变。** 两个向量按相同角度共同旋转时点积不变；按不同位置旋转时，点积通常改变。
3. **位置不能在增量生成时重新归零。** 使用 KV Cache 时，新 token 应使用完整序列中的实际位置；例如已有 128 个 token，下一 token 通常使用位置 128。已旋转并缓存的 K 不应再次旋转。
4. **配对规则要与模型一致。** 本文把相邻维度配对；有的实现将前半部分和后半部分配对。加载预训练权重时，还需遵循原模型的基数、旋转维度和其他配置。
5. **这是教学实现。** 它每次调用都会重新计算正弦和余弦，并以 float32 进行旋转；生产实现可缓存相应结果，也应根据模型和精度要求调整。

## 参考资料

- 苏剑林：[Transformer 升级之路：2、博采众长的旋转式位置编码](https://spaces.ac.cn/archives/8265)。
- Su 等：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)。
