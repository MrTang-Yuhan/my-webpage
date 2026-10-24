---
layout: post.njk
post_id: 2026-10-24-大模型中的数学-003-为什么每个-token-的-kv-会受到上下文影响
archive: 数学原理
title: " 大模型中的数学（003）：为什么每个 Token 的 KV 会受到上下文影响？"
date: 2026-10-24
updated: 2026-10-24
tags:
  - post
---
# 从一层到两层：KV 为什么会带上上下文？

讨论 KV Cache 时，很容易产生一个疑问：每个 token 不都是各自计算自己的 Key 和 Value 吗？既然如此，**同一个 token 出现在不同句子里，为什么它的 KV 还会不同？**

## 先看第一层

设第 $i$ 个 token 的初始表示为 $x_i^{(0)}$。它通常由 token embedding 得到；位置相关的信息如何加入，则取决于模型采用的位置编码方式。

以常见的 Pre-LN Transformer 为例，第一层先对输入做归一化，再计算 Key 和 Value：

$$
k_i^{(1)} = \operatorname{LN}(x_i^{(0)})W_K^{(1)}
$$

$$
v_i^{(1)} = \operatorname{LN}(x_i^{(0)})W_V^{(1)}
$$

先注意一个细节：这两个公式里没有出现其他 token 的隐藏状态。因此，如果模型参数、当前 token 的初始表示以及位置条件都相同，那么第一层的 K、V 可以相同，即使前面的文本不同。

不过，“位置条件相同”不能省略。以 RoPE 为例，投影得到的 Query 和 Key 还会经过位置相关的旋转；如果 token 的位置变了，实际用于 Attention 的 Key 也可能随之变化。所以下文比较不同上下文时，先假设目标 token 的位置相同，暂时排除位置差异的影响。

第一层的 KV 看起来确实可以只由当前位置的初始表示决定。上下文是从哪里进入计算的呢？关键在 Attention。

## 上下文是怎样进入隐藏状态的？

考虑因果 Attention。位置 $i$ 只能读取位置 $1,\ldots,i$，不能读取未来 token。对单个注意力头，位置 $i$ 对位置 $j$ 的注意力权重为

$$
\alpha_{ij}^{(1)}
=
\frac{
\exp\left(q_i^{(1)}(k_j^{(1)})^\top/\sqrt{d_k}\right)
}{
\sum_{m\leq i}
\exp\left(q_i^{(1)}(k_m^{(1)})^\top/\sqrt{d_k}\right)
}
$$

于是，位置 $i$ 的 Attention 输出是

$$
a_i^{(1)}
=
\sum_{j\leq i}\alpha_{ij}^{(1)}v_j^{(1)}
$$

这一步才发生了跨 token 的信息交互。$a_i^{(1)}$ 是多个位置的 Value 的加权和，权重则由 Query 和各个 Key 的匹配程度决定。它不只是当前位置自己的信息，也可能包含前面 token 提供的信息。

例如，假设我们关注句子末尾的“苹果”。在“我喜欢苹果”中，它可以读取“我”和“喜欢”；在前缀不同、但目标 token 位置相同的另一句话中，它能够读取的内容或分配的注意力权重可能不同。即使“苹果”的初始表示一样，Attention 输出也不必一样。

实际模型通常使用多头 Attention。不同头会分别计算注意力，再把结果拼接并经过输出投影。为了看清信息依赖，上面的单头公式已经足够；多头结构不会改变这个结论。

## Attention 输出如何影响下一层？

Attention 输出并不是直接替换原来的隐藏状态。常见的 Pre-LN Block 还带有残差连接和 FFN。把第一层写完整：

$$
z_i^{(1)} = x_i^{(0)} + a_i^{(1)}
$$

$$
x_i^{(1)}
=
z_i^{(1)}
+
\operatorname{FFN}_1\left(\operatorname{LN}(z_i^{(1)})\right)
$$

第一条公式把 Attention 提取的上下文信息加回原始表示；第二条公式再对这个结果逐位置地做 FFN 变换。FFN 通常不会直接混合不同位置的向量，跨 token 的信息交换主要发生在 Attention 中。不过，FFN 的输入已经含有上下文信息，所以它的输出也会保留并进一步变换这些信息。

将 Attention 的表达式代入，可以把第一层的输出理解为

$$
x_i^{(1)}
=
x_i^{(0)}
+
\sum_{j\leq i}\alpha_{ij}^{(1)}v_j^{(1)}
+
\operatorname{FFN}_1\left(
\operatorname{LN}\left(
x_i^{(0)}+
\sum_{j\leq i}\alpha_{ij}^{(1)}v_j^{(1)}
\right)
\right)
$$

这里省略了多头 Attention 的输出投影，只保留单头形式来展示依赖关系。公式不必逐项计算，重点是：**$x_i^{(1)}$ 已经可能依赖前面多个位置的表示。**

## 把第二层的 KV 展开

到了第二层，模型使用的输入不再是最初的 $x_i^{(0)}$，而是第一层输出的 $x_i^{(1)}$。因此，第二层的 Key 和 Value 为

$$
k_i^{(2)} = \operatorname{LN}(x_i^{(1)})W_K^{(2)}
$$

$$
v_i^{(2)} = \operatorname{LN}(x_i^{(1)})W_V^{(2)}
$$

把前面两条残差公式代入 Key 的表达式，就得到

$$
\begin{aligned}
k_i^{(2)}
={}&\operatorname{LN}\Bigg(
x_i^{(0)} + a_i^{(1)} \\
&\quad+
\operatorname{FFN}_1\left(
\operatorname{LN}(x_i^{(0)}+a_i^{(1)})
\right)
\Bigg)W_K^{(2)}
\end{aligned}
$$

而

$$
a_i^{(1)}=\sum_{j\leq i}\alpha_{ij}^{(1)}v_j^{(1)}
$$

所以，**第二层的 Key 通过 $a_i^{(1)}$ 间接依赖前面 token 的 Value 和注意力权重；第二层的 Value 也有同样的依赖关系，因为它同样由 $x_i^{(1)}$ 投影而来。**

这就是“后续层的 KV 会受到上下文影响”的具体含义。不是说模型在计算 $k_i^{(2)}$ 时，额外把所有前文 token ID 拼进了一个公式；而是前文已经通过第一层 Attention 影响了 $x_i^{(1)}$，第二层再从这个新表示计算 K、V。

上面的公式采用 Pre-LN 结构。其他 Transformer 变体的归一化位置和残差形式可能不同，但只要某一层的隐藏状态经过了上下文 Attention，后续层从该隐藏状态生成的 KV 就可能带有上下文信息。

## 这为什么会影响 KV Cache 复用？

假设 Chunk 2 单独运行时，前面没有 Chunk 1；而在完整输入中，顺序变成 `Chunk 1 + Chunk 2`。

单独运行时，Chunk 2 的 token 无法关注 Chunk 1。合并后，因果 Attention 允许 Chunk 2 的 token 读取 Chunk 1。于是，从第二层开始，Chunk 2 的隐藏状态就可能不同，对应的 K、V 也可能不同。


这也是几种缓存策略之间的关键差别：前缀缓存复用的是相同前缀在相同计算条件下得到的 KV；直接拼接各个独立文本块的 KV，则可能漏掉块与块之间的上下文交互。

最后可以把整个逻辑压缩成一句话：**每个位置确实有自己的 K、V，但后续层的 K、V 是从已经经过 Attention 更新的隐藏状态计算出来的，因此不能只看 token ID 判断 KV 是否相同。**

