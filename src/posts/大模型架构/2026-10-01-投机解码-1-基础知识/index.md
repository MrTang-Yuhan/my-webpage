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
# Speculative Sampling：从单步正确性到序列级无损生成

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

定义

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

另一方面，由于 $p(\cdot\mid h)$ 与 $q(\cdot\mid h)$ 都是归一化分布，

$$
\sum_{x\in\mathcal V}
\left(p(x\mid h)-q(x\mid h)\right)
=0.
$$

对于任意实数 $u,v$，有

$$
u-v=[u-v]_+-[v-u]_+.
$$

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

这说明草稿模型的作用只是提出候选。草稿模型越接近目标模型，通常接受率越高；即使草稿模型较差，只会增加拒绝次数和计算开销，不会改变单步输出分布。

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

## 4. 块级算法为何仍然严格无损

原始的序列证明容易出现一个问题：草稿模型已经预先生成了整块候选，不能直接把“下一个候选 token”不加条件地写成 $q(\cdot\mid\text{当前真实前缀})$。正确做法是对**已经到达的验证阶段**进行条件化。

### 4.1 到达第 $i$ 个验证位置时的条件分布

在固定轮次起点历史 $h$ 和已揭示前缀 $\tilde X_{1:i-1}$ 下，定义事件

$$
E_i
=
\{\text{前 }i-1\text{ 个草稿 token 全部被接受}\}.
$$

在事件 $E_i$ 发生时，当前已经输出的历史正好是

$$
h,\tilde X_1,\ldots,\tilde X_{i-1}.
$$

第 $i$ 个候选 $\tilde X_i$ 在预生成时只依赖于草稿前缀 $\tilde X_{1:i-1}$，而 $E_i$ 只涉及前 $i-1$ 个候选及其接受随机数。因此，在给定当前前缀后，

$$
\boxed{
\Pr\left(
\tilde X_i=x
\mid
h,\tilde X_{1:i-1},E_i
\right)
=
q_i(x).
}
$$

这一步只对“当前已经到达的阶段”进行条件化，不能把完整的未来草稿值一起固定后再声称其仍服从 $q_i$。更直观地，可以把预生成理解为“延迟揭示”：验证第 $i$ 位之前，只揭示已经接受的前缀和当前所需的候选，未来候选全部边缘化。因为草稿序列的联合分布满足

$$
q(\tilde x_{1:\gamma}\mid h)
=
\prod_{i=1}^{\gamma}q\bigl(\tilde x_i\mid h,\tilde x_{1:i-1}\bigr),
$$

所以预先生成整块与按需揭示具有相同的相关分布。

### 4.2 验证阶段的下一个输出 token

在给定事件 $E_i$ 和当前历史 $h,\tilde X_{1:i-1}$ 后，第 $i$ 个验证位置完全符合第 2 节的单步算法：

- 候选分布是 $q_i$；
- 目标分布是 $p_i$；
- 接受概率是 $\min(1,p_i/q_i)$；
- 拒绝后使用残差分布 $\mu_i$。

因此，由单步结论，

$$
\boxed{
\Pr(\text{该阶段输出 }x
\mid h,\tilde X_{1:i-1},E_i)
=
p_i(x).
}
$$

由于 $E_i$ 表示前面的草稿都已被接受，

$$
h,\tilde X_{1:i-1}
$$

就是当前真实输出历史。注意这里的结论是把当前候选的“接受”和“拒绝后残差”两条路径一起混合后的结论；如果单独条件在接受路径或拒绝路径上，分布一般不再等于 $p_i$。

因此，上式正是目标模型在当前真实历史下的条件分布。若同一真实历史可以通过不同的轮次或验证阶段到达，则对这些到达方式再使用全概率公式，混合结果仍然是同一个目标条件分布。

### 4.3 全部接受后的 bonus 阶段

如果所有 $\gamma$ 个草稿 token 都被接受且此前没有输出 EOS，则进入 bonus 阶段。下文将 $E_{\gamma+1}$ 理解为“前 $\gamma$ 个候选全部接受且此前没有输出 EOS”的事件，此时当前真实历史为

$$
h,\tilde X_{1:\gamma}.
$$

bonus token 直接从

$$
p_{\gamma+1}(\cdot)
=
p\bigl(\cdot\mid h,\tilde X_{1:\gamma}\bigr)
$$

中采样，因此

$$
\boxed{
\Pr(\text{bonus}=x
\mid
h,\tilde X_{1:\gamma},E_{\gamma+1})
=
p_{\gamma+1}(x).
}
$$

因此，在验证阶段，必须把当前候选的接受路径和拒绝后的残差路径合并后，才能得到目标分布；不能在额外条件化于某一条路径后，单独声称该路径仍服从 $p$。全接受后的 bonus token 则直接从目标分布采样。

### 4.4 序列级 lossless 结论

令 $\mathcal A$ 表示块级 Speculative Decoding 算法，令 $Y_t$ 表示算法输出的第 $t$ 个 token，令

$$
H_t=Y_{<t}
$$

表示生成第 $t$ 个 token 前已经输出的真实历史。再令 $S_t$ 表示算法内部当前所处的阶段，例如某个验证位置、bonus 阶段或新一轮的起点。这里的阶段状态只记录当前真实历史、阶段索引和已经公开的接受结果，不把尚未揭示的草稿尾部作为条件信息。

对任意可达的阶段 $s$，上一节的阶段条件化结论都给出

$$
\Pr^{\mathcal A}(Y_t=y\mid H_t=h,S_t=s)
=
p(y\mid h).
$$

因此，对隐藏阶段变量使用全概率公式，得到

$$
\begin{aligned}
\Pr^{\mathcal A}(Y_t=y\mid H_t=h)
&=
\sum_s
\Pr^{\mathcal A}(Y_t=y\mid H_t=h,S_t=s)
\Pr(S_t=s\mid H_t=h)\\
&=
\sum_s p(y\mid h)\Pr(S_t=s\mid H_t=h)\\
&=
 p(y\mid h).
\end{aligned}
$$

所以，对任意真实输出历史 $y_{<t}$，都有

$$
\boxed{
\Pr^{\mathcal A}(Y_t=y\mid Y_{<t}=y_{<t})
=
p(y\mid y_{<t}).
}
$$

再由链式法则，对于任意长度为 $T$ 的 token 序列 $y_{1:T}$，

$$
\begin{aligned}
\Pr^{\mathcal A}(Y_{1:T}=y_{1:T})
&=
\prod_{t=1}^{T}
\Pr^{\mathcal A}(Y_t=y_t\mid Y_{<t}=y_{<t})\\
&=
\prod_{t=1}^{T}
p(y_t\mid y_{<t})\\
&=
p(y_{1:T}).
\end{aligned}
$$

因此，

$$
\boxed{
\Pr^{\mathcal A}(Y_{1:T}=y_{1:T})
=
p(y_{1:T}).
}
$$

如果把 EOS 视为词表中的普通 token，那么对于在 EOS 处终止的序列，上式应理解为一直相乘到 EOS 所在位置；一旦输出 EOS，就停止后续生成。该结论说明 Speculative Decoding 在理想数学条件下与直接运行目标模型具有相同的序列分布。[Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)

---

## 5. 接受率与总变差距离

### 5.1 固定上下文下的平均接受率

需要区分两种量：

- $a(x\mid h)$：给定候选 token $x$ 时的接受概率；
- $\alpha(h)$：先从草稿分布采样，再对候选取平均后的接受率。

定义

$$
\begin{aligned}
\alpha(h)
&\coloneqq
\sum_{x\in\mathcal V}
q(x\mid h)a(x\mid h)\\
&=
\sum_{x\in\mathcal V}
\min\bigl(p(x\mid h),q(x\mid h)\bigr).
\end{aligned}
$$

利用

$$
\min(u,v)
=
\frac{u+v-|u-v|}{2},
$$

并使用 $p$ 与 $q$ 都归一化这一事实，有

$$
\begin{aligned}
\alpha(h)
&=
1-
\frac12
\sum_{x\in\mathcal V}
|p(x\mid h)-q(x\mid h)|\\
&=
1-
d_{\mathrm{TV}}
\bigl(p(\cdot\mid h),q(\cdot\mid h)\bigr),
\end{aligned}
$$

其中

$$
d_{\mathrm{TV}}(p,q)
\coloneqq
\frac12\sum_{x\in\mathcal V}|p(x)-q(x)|
$$

是总变差距离。因此，对每个固定上下文 $h$，都有精确关系

$$
\boxed{
\alpha(h)
=
1-
d_{\mathrm{TV}}
\bigl(p(\cdot\mid h),q(\cdot\mid h)\bigr).
}
$$

在真实自回归生成中，上下文会不断变化，因此接受率通常也是上下文相关的，不能默认存在一个对所有位置都完全相同的常数 $\alpha$。

---

## 6. 每轮输出 token 数与期望值

### 6.1 精确计数

设 $A\in\{0,1,\ldots,\gamma\}$ 表示本轮从第一个位置开始连续接受的草稿 token 数量。

无论本轮在哪个位置首次拒绝，还是所有草稿都被接受，本轮都会额外输出一个 token：

- 首次拒绝时，额外输出一个残差采样 token；
- 全部接受时，额外输出一个 bonus token。

因此，在暂时忽略 EOS 和最大长度截断时，本轮输出 token 总数 $N$ 满足

$$
\boxed{
N=A+1.
}
$$

由非负整数随机变量的尾和公式（以下期望均条件于本轮起点历史 $h$），

$$
\mathbb E[A\mid h]
=
\sum_{i=1}^{\gamma}
\Pr(A\ge i\mid h),
$$

从而

$$
\boxed{
\mathbb E[N\mid h]
=
1+
\sum_{i=1}^{\gamma}
\Pr(A\ge i\mid h).
}
$$

这里的事件 $A\ge i$ 等价于“前 $i$ 个草稿 token 全部被接受”。这个公式不需要独立性假设。

### 6.2 一般的条件接受率表达式

固定本轮起点历史 $h$，令

$$
r_i(h)
\coloneqq
\Pr\bigl(
\text{第 }i\text{ 个草稿被接受}
\mid
h,\ \text{前 }i-1\text{ 个草稿均被接受}
\bigr).
$$

由条件概率的乘法公式，

$$
\Pr(A\ge i\mid h)
=
\prod_{j=1}^{i}r_j(h).
$$

因此，精确地有

$$
\boxed{
\mathbb E[N\mid h]
=
1+
\sum_{i=1}^{\gamma}
\prod_{j=1}^{i}r_j(h).
}
$$

在真实模型中，$r_i(h)$ 会受到当前已接受前缀和上下文的影响，因此通常随 $i$ 变化。

### 6.3 齐次接受率近似

为了得到简洁的闭式，可以作一个明确的性能近似：沿着连续接受路径，各位置的条件接受率近似相同，即

$$
r_i(h)\approx\alpha,
\qquad i=1,\ldots,\gamma.
$$

于是

$$
\Pr(A\ge i\mid h)\approx\alpha^i.
$$

当 $0\le\alpha<1$ 时，

$$
\boxed{
\mathbb E[A\mid h]
\approx
\sum_{i=1}^{\gamma}\alpha^i
=
\frac{1-\alpha^{\gamma+1}}{1-\alpha}-1,
}
$$

而每轮实际输出 token 数的期望为

$$
\boxed{
\mathbb E[N\mid h]
\approx
\sum_{i=0}^{\gamma}\alpha^i
=
\frac{1-\alpha^{\gamma+1}}{1-\alpha}.
}
$$

两个边界情况为：

- $\alpha=0$ 时，$\mathbb E[A\mid h]=0$，但 $\mathbb E[N\mid h]=1$；
- $\alpha=1$ 时，$\mathbb E[A\mid h]=\gamma$，且 $\mathbb E[N\mid h]=\gamma+1$。

因此，$\mathbb E[A\mid h]$ 不能直接作为每轮吞吐量的分子，因为它漏掉了每轮必然输出的残差 token 或 bonus token。仅知道单步平均接受率，或知道各位置边际接受率相同，也不足以严格推出 $\Pr(A\ge i\mid h)=\alpha^i$；该等式需要额外的齐次条件接受率近似。

---

## 7. 加速比与时间开销

### 7.1 一般时间模型

每轮计算时间可以拆成三部分：

- 草稿模型生成 $\gamma$ 个候选的时间；
- 目标模型并行验证候选并计算 bonus 分布的时间；
- 接受判断、残差采样和缓存管理等额外时间。

记

- $\tau_{\mathrm{ver}}(\gamma)$：一次目标模型验证长度为 $\gamma$ 的草稿并计算 bonus 分布的延迟；
- $\tau_{\mathrm{extra}}(\gamma)$：其他额外开销。

则一轮平均计算时间可写为

$$
C_\gamma
\approx
\gamma\tau_q
+
\tau_{\mathrm{ver}}(\gamma)
+
\tau_{\mathrm{extra}}(\gamma).
$$

直接使用目标模型逐 token 生成时，每输出一个 token 的平均时间约为 $\tau_p$。下式中的 $\mathbb E[N]$ 表示对实际轮次起点历史 $h$ 的长期平均；若固定某个 $h$，则使用 $\mathbb E[N\mid h]$。因此，以长期平均吞吐量估计，加速比为

$$
\boxed{
\mathrm{Speedup}
\approx
\frac{\mathbb E[N]\tau_p}
{\gamma\tau_q
+\tau_{\mathrm{ver}}(\gamma)
+\tau_{\mathrm{extra}}(\gamma)}.
}
$$

### 7.2 理想化闭式

如果进一步采用经典理想化假设

$$
\tau_{\mathrm{ver}}(\gamma)\approx\tau_p,
\qquad
\tau_{\mathrm{extra}}(\gamma)\approx 0,
$$

并令

$$
c\coloneqq\frac{\tau_q}{\tau_p},
$$

则

$$
\mathrm{Speedup}
\approx
\frac{\mathbb E[N]}{1+\gamma c}.
$$

在齐次接受率近似下，当 $0\le\alpha<1$ 时，

$$
\boxed{
\mathrm{Speedup}
\approx
\frac{\displaystyle\sum_{i=0}^{\gamma}\alpha^i}
{1+\gamma c}
=
\frac{1-\alpha^{\gamma+1}}
{(1-\alpha)(1+\gamma c)}.
}
$$

边界情况为

$$
\boxed{
\mathrm{Speedup}
\approx
\frac{\gamma+1}{1+\gamma c},
\qquad \alpha=1,
}
$$

以及

$$
\boxed{
\mathrm{Speedup}
\approx
\frac{1}{1+\gamma c},
\qquad \alpha=0.
}
$$

这里的闭式是明确假设下的性能估计，不是所有硬件和实现中的精确公式。尤其是一次并行验证的延迟不一定严格等于一次单 token 目标模型前向，$\tau_{\mathrm{ver}}(\gamma)$ 可能随 $\gamma$、硬件、批大小和缓存状态变化。[Chen et al., 2023](https://arxiv.org/abs/2302.01318)

---

## 8. 精确结论、近似结论与边界条件

### 8.1 精确结论

在以下条件成立时，Speculative Sampling 的输出分布严格等于目标模型分布：

1. $p$ 和 $q$ 使用相同的词表；
2. 接受概率使用

   $$
   \min\left(1,\frac{p}{q}\right);
   $$

3. 拒绝后使用归一化残差分布

   $$
   \frac{[p-q]_+}{\sum_x[p(x)-q(x)]_+};
   $$

4. 块内按照从左到右的验证阶段执行；
5. 首次拒绝后丢弃后续未验证草稿；
6. 全部接受后从目标分布生成 bonus token。

### 8.2 近似结论

下列结论需要额外近似：

- $\Pr(A\ge i\mid h)\approx\alpha^i$；
- $\mathbb E[N]\approx(1-\alpha^{\gamma+1})/(1-\alpha)$；
- $\tau_{\mathrm{ver}}(\gamma)\approx\tau_p$；
- 用单一常数 $\alpha$ 代表所有上下文和所有验证位置的条件接受率。

因此，**无损性是概率分布层面的精确结论，而加速比闭式是带有性能建模假设的估计**。

### 8.3 数值实现中的限制

实际实现使用有限精度浮点数，可能出现极小的数值误差。因此工程文献通常将无损性表述为在数值精度范围内保持目标分布，而数学推导对应的是理想精确算术。[Chen et al., 2023](https://arxiv.org/abs/2302.01318)

---

## 参考资料

1. Yaniv Leviathan, Matan Kalman, Yossi Matias. **Fast Inference from Transformers via Speculative Decoding**. ICML 2023.  
   <https://proceedings.mlr.press/v202/leviathan23a.html>

2. Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper. **Accelerating Large Language Model Decoding with Speculative Sampling**. 2023.  
   <https://arxiv.org/abs/2302.01318>
