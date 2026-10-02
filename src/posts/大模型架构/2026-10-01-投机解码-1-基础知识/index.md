---
layout: post.njk
post_id: 2026-10-01-投机解码-1-基础知识
archive: 大模型架构
title: 投机解码 (1)：拒绝采样机制
date: 2026-10-01
updated: 2026-10-02
tags:
  - post
---

## 0. 证明目标

目标：**从任意初始历史 $h_0$ 出发，Speculative Sampling 算法从草稿模型生成完整序列的分布，与目标模型自回归采样完全一致。**

## 1. 问题设定与动机

自回归语言模型按 token 逐步生成。给定已经生成的历史 $h$，下一个 token 的条件分布由模型决定。**目标模型**（target model）记为 $p$，**草稿模型**（draft model）记为 $q$。

目标模型通常质量更高，但单次前向延迟 $\tau_p$ 较大；草稿模型的单 token 生成延迟 $\tau_q$ 较小，通常满足 $\tau_q\ll\tau_p$，但其分布 $q$ 与目标分布 $p$ 可能不同。

**Speculative Sampling**（也称 **Speculative Decoding**）的基本做法是：

1. 先由草稿模型连续生成一小段候选 token；
2. 再由目标模型并行计算这段候选的条件分布；
3. 从左到右逐个接受或拒绝候选；
4. 首次拒绝时使用残差分布重新采样；
5. 如果所有候选都被接受，再从目标模型生成一个额外 token。

只要接受概率和残差分布按下面的规则定义，最终生成结果的分布就与直接使用目标模型自回归采样的分布一致。这里的“一致”是严格的概率分布一致，而不是逐次运行时一定得到相同的随机 token。该算法及其无损性质见 Leviathan 等人的工作和 Chen 等人的工作。[Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)；[Chen et al., 2023](https://arxiv.org/abs/2302.01318)

为避免符号混乱，本文统一采用：

- $\mathcal V$：词表，包含普通 token 和 **EOS token**；
- $h$：当前已经输出的历史上下文；
- $p(x\mid h)$：目标模型在上下文 $h$ 下输出 token $x$ 的概率；
- $q(x\mid h)$：草稿模型在上下文 $h$ 下输出 token $x$ 的概率；
- **$\gamma$**：每轮由草稿模型预生成的候选 token 数量；
- **$\tau_p$**：目标模型单次常规自回归前向的延迟；
- **$\tau_q$**：草稿模型生成一个 token 的平均延迟。
- $Pr$ : Speculative Sample 算法的输出概率。

除非特别说明，所有分布都定义在同一个词表 $\mathcal V$ 上，并满足

$$
\sum_{x\in\mathcal V}p(x\mid h)
=
\sum_{x\in\mathcal V}q(x\mid h)
=
1.
$$

---

## 2. 单步 Speculative Sampling

先固定一个上下文 $h$，只研究如何生成下一个 token。

### 2.1 算法定义

首先从草稿分布采样候选 token：

$$
\tilde X\sim q(\cdot\mid h).
$$

对于候选值 $x$，定义接受概率

$$
a(x\mid h)
=
\min\left(1,\frac{p(x\mid h)}{q(x\mid h)}\right),
$$

其中仅在 $q(x\mid h)>0$ 时需要计算比值。若 $q(x\mid h)=0$，该候选被采样到的概率本身为零，因此可以任意定义 $a(x\mid h)$，例如令其为 $1$。

然后执行：

1. 以概率 $a(\tilde X\mid h)$ 接受候选，并输出 $\tilde X$；
2. 以剩余概率拒绝候选，并从残差分布 $\mu(\cdot\mid h)$ 中重新采样一个 token 输出。

$[\cdot]_+$ 表示**正部函数（Positive Part Function）**，具体定义为：

$$
[z]_+ \coloneqq \max(0,z),
$$

以及残差总质量

$$
\beta(h)
\coloneqq
\sum_{x\in\mathcal V}
\bigl[p(x\mid h)-q(x\mid h)\bigr]_+.
$$

当 $\beta(h)>0$ 时，残差分布为

$$
\mu(x\mid h)
=
\frac{\bigl[p(x\mid h)-q(x\mid h)\bigr]_+}{\beta(h)}.
$$

当 $\beta(h)=0$ 时，有 $p(\cdot\mid h)=q(\cdot\mid h)$，拒绝事件的概率为零，因此无需实际调用残差分布。

### 2.2 接受路径的概率质量

对任意 $x\in\mathcal V$，候选等于 $x$ 且被接受的概率为

$$
\begin{aligned}
\Pr(\text{accept},X=x\mid h)
&=
q(x\mid h)\,
\min\left(1,\frac{p(x\mid h)}{q(x\mid h)}\right)\\
&=
\min\bigl(q(x\mid h),p(x\mid h)\bigr).
\end{aligned}
$$


这个等式可以直接按两种情况验证：

- 若 $p(x\mid h)\ge q(x\mid h)$，接受概率为 $1$，接受路径质量为 $q(x\mid h)$；
- 若 $p(x\mid h)<q(x\mid h)$，接受概率为 $p(x\mid h)/q(x\mid h)$，接受路径质量为 $p(x\mid h)$。

因此，

$$
\boxed{
\Pr(\text{accept},X=x\mid h)
=
\min\bigl(q(x\mid h),p(x\mid h)\bigr).
}
$$

### 2.3 拒绝概率与残差归一化常数

总接受概率为

$$
\sum_{x\in\mathcal V}
\min\bigl(q(x\mid h),p(x\mid h)\bigr).
$$

因此，总拒绝概率为

$$
\begin{aligned}
\Pr(\text{reject}\mid h)
&=
1-
\sum_{x\in\mathcal V}
\min\bigl(q(x\mid h),p(x\mid h)\bigr)\\
&=
\sum_{x\in\mathcal V}
\bigl[q(x\mid h)-p(x\mid h)\bigr]_+.
\end{aligned}
$$

最后一个等式使用了

$$
u-\min(u,v)=[u-v]_+.
$$

> 证明 $u-\min(u,v)=[u-v]_+$ 使用分类讨论：（1）$u \ge v$； （2） $u < v$。

另一方面，由于 $p(\cdot\mid h)$ 与 $q(\cdot\mid h)$ 都是归一化分布（所有概率求和为 1），

$$
\sum_{x\in\mathcal V}
\left(p(x\mid h)-q(x\mid h)\right)
=0.
$$

对于任意实数 $u,v$，有

$$
u-v=[u-v]_+-[v-u]_+.
$$

> 证明 $u-v=[u-v]_+-[v-u]_+$ 使用分类讨论：（1）$u \ge v$； （2） $u < v$。

令 $u=p(x\mid h)$、$v=q(x\mid h)$，并对 $x$ 求和，可得

$$
\sum_{x\in\mathcal V}
\bigl[p(x\mid h)-q(x\mid h)\bigr]_+
=
\sum_{x\in\mathcal V}
\bigl[q(x\mid h)-p(x\mid h)\bigr]_+.
$$

所以

$$
\boxed{
\Pr(\text{reject}\mid h)=\beta(h).
}
$$

这说明残差分布的归一化常数正好等于拒绝概率。进一步，

$$
\beta(h)
=
\frac12\sum_{x\in\mathcal V}
\left|p(x\mid h)-q(x\mid h)\right|
=
d_{\mathrm{TV}}
\bigl(p(\cdot\mid h),q(\cdot\mid h)\bigr).
$$

> $d_{\text{TV}}$ 代表 **Total Variation Distance（全变差距离）**。它是概率论和统计学中用来衡量**两个概率分布之间差异程度**的一种度量（距离）。
> 定义为：
> $$d_{\text{TV}}(p(\cdot \mid h), q(\cdot \mid h)) = \frac{1}{2} \sum_{x \in \mathcal{V}} |p(x \mid h) - q(x \mid h)|$$

因此，$\beta(h)$ 同时表示拒绝概率和两个条件分布之间的总变差距离。

### 2.4 拒绝路径的概率质量

拒绝后从 $\mu(\cdot\mid h)$ 重新采样，因此对任意 $x$，

$$
\begin{aligned}
\Pr(\text{reject},X=x\mid h)
&=
\Pr(\text{reject}\mid h)\,
\mu(x\mid h)\\
&=
\beta(h)\,
\frac{\bigl[p(x\mid h)-q(x\mid h)\bigr]_+}{\beta(h)}\\
&=
\bigl[p(x\mid h)-q(x\mid h)\bigr]_+.
\end{aligned}
$$

当 $\beta(h)=0$ 时，拒绝路径概率本身为零，上式可理解为不需要实际执行残差采样。

### 2.5 单步输出分布严格等于目标分布

最终输出 $X=x$ 的概率是接受路径和拒绝路径之和：

$$
\begin{aligned}
\Pr(X=x\mid h)
&=
\min\bigl(q(x\mid h),p(x\mid h)\bigr)
+
\bigl[p(x\mid h)-q(x\mid h)\bigr]_+\\
&=
p(x\mid h).
\end{aligned}
$$

因此得到单步结论：

$$
\boxed{
\Pr(X=x\mid h)=p(x\mid h),
\qquad \forall x\in\mathcal V.
}
$$

这说明草稿模型的作用只是提出候选。草稿模型越接近目标模型，通常接受率越高；即使草稿模型较差，只会增加拒绝次数和计算开销，**不会改变单步输出分布**。

---

## 3. 块级 Speculative Decoding 的流程

设当前已经输出的历史为 $h$，每轮草稿长度为 $\gamma$。

### 3.1 预生成草稿

草稿模型按自身的自回归分布依次生成

$$
\tilde X_i
\sim
q\bigl(\cdot\mid h,\tilde X_{1:i-1}\bigr),
\qquad
i=1,\ldots,\gamma.
$$

记

$$
q_i(x)
\coloneqq
q\bigl(x\mid h,\tilde X_{1:i-1}\bigr),
$$

并记目标模型在同一草稿前缀下的条件分布为

$$
p_i(x)
\coloneqq
p\bigl(x\mid h,\tilde X_{1:i-1}\bigr).
$$

为进行概率证明，假设每个草稿位置使用给定前缀下的新鲜采样随机数；验收随机数、残差采样随机数和 bonus 采样随机数彼此独立，并且目标模型前向在给定输入后是确定的。目标模型一次前向可以并行计算这些位置的分布，并额外计算全接受情况下所需的 bonus 分布

$$
p_{\gamma+1}(x)
\coloneqq
p\bigl(x\mid h,\tilde X_{1:\gamma}\bigr).
$$

### 3.2 从左到右验证

对 $i=1,\ldots,\gamma$，依次执行：

1. 使用接受概率

   $$
   a_i(x)=\min\left(1,\frac{p_i(x)}{q_i(x)}\right)
   $$

   判断候选 $\tilde X_i$ 是否接受；这里沿用第 2.1 节的约定：若 $q_i(x)=0$，接受概率可任意定义，例如置为 $1$。
2. 如果接受，就输出 $\tilde X_i$，继续验证下一位置；
3. 如果拒绝，就从

   $$
   \mu_i(x)
   =
   \frac{[p_i(x)-q_i(x)]_+}
   {\sum_{z\in\mathcal V}[p_i(z)-q_i(z)]_+}
   $$

   中采样一个 token 输出，并立即结束本轮；
4. 如果 $\gamma$ 个候选全部被接受，且此前没有输出 EOS，就输出这 $\gamma$ 个候选，再从 $p_{\gamma+1}$ 中采样一个 bonus token。若残差分布的归一化常数为零，则该位置的拒绝概率为零，不需要执行残差采样。

首次拒绝后，后面的预生成草稿会被丢弃，因为它们依赖于尚未被接受的候选前缀。若某次输出为 EOS，则在该位置停止生成。


---

### 4. 序列层面的一致性

**目标**：**证明从任意初始历史 $h_0$ 出发，Speculative Decoding 算法 $\mathcal{A}$ 生成完整序列的分布，与目标模型自回归采样完全一致。**

**证明**：

**第一步：明确单步引理（复用第2节结论）**

由第2节单步 Speculative Sampling 的证明可知：
对于任意给定的历史 $h$，算法 $\mathcal{A}$ 在生成下一个 token 时，其输出分布严格等于目标模型的条件分布。即：
$$
\Pr\nolimits(X=x \mid h) = p(x \mid h), \qquad \forall x \in \mathcal{V}.
$$
*注意：这个引理涵盖了第3节块级流程中的所有情况（无论是验证位置被接受、被拒绝后残差采样、还是全接受后的 bonus token）。只要算法在历史 $h$ 下输出下一个 token，其边缘分布就是 $p(\cdot \mid h)$。*

**第二步：定义算法生成的随机过程**

令算法生成的 token 序列为 $Y_1, Y_2, \ldots$，实际历史为 $H_0 = h_0$，$H_t = (h_0, Y_{1:t})$。
定义停止时间 $T = \min\{t \ge 1 : Y_t = \mathrm{EOS}\}$。

**第三步：应用概率链式法则**

对于任意以 EOS 结尾的序列 $y_{1:T}$（其中 $y_T = \mathrm{EOS}$），算法生成该序列的概率为：
$$
\Pr\nolimits(Y_{1:T} = y_{1:T} \mid h_0) = \prod_{t=1}^{T} \Pr\nolimits(Y_t = y_t \mid H_{t-1}).
$$

> 根据概率论的基本乘法公式：
>
> $$P(A, B) = P(A) \cdot P(B \mid A)$$
> 
> 推广到多个事件：
>
> $$P(Y_1, Y_2, \dots, Y_T) = P(Y_1) \cdot P(Y_2 \mid Y_1) \cdot P(Y_3 \mid Y_1, Y_2) \cdots P(Y_T \mid Y_1, \dots, Y_{T-1})$$

**第四步：代入单步引理**

因为 $H_{t-1}$ 是算法生成 $Y_t$ 时的实际历史，根据第一步的引理，我们有：
$$
\Pr\nolimits(Y_t = y_t \mid H_{t-1}) = p(y_t \mid H_{t-1}) = p(y_t \mid h_0, y_{1:t-1}).
$$
将其代入第三步的连乘式中：
$$
\Pr\nolimits(Y_{1:T} = y_{1:T} \mid h_0) = \prod_{t=1}^{T} p(y_t \mid h_0, y_{1:t-1}).
$$

**第五步：与目标模型对比**

目标模型自回归采样从同一初始历史 $h_0$ 生成同一序列的概率恰好为：
$$
P_{\mathrm{tar}}(Y_{1:T} = y_{1:T} \mid h_0) = \prod_{t=1}^{T} p(y_t \mid h_0, y_{1:t-1}).
$$

因此：
$$
\boxed{
\Pr\nolimits(Y_{1:T} = y_{1:T} \mid h_0) = P_{\mathrm{tar}}(Y_{1:T} = y_{1:T} \mid h_0).
}
$$
证毕。

---

