---
layout: post.njk
post_id: 2026-07-03-概率密度函数-pdf-与累积分布函数-cdf
archive: 数学原理
title: 概率密度函数（PDF）与累积分布函数（CDF）
date: 2026-07-03
tags:
  - post
---
# 概率密度函数（PDF）与累积分布函数（CDF）完整推导

---

## 一、公式作用概述

概率密度函数（Probability Density Function, PDF）和累积分布函数（Cumulative Distribution Function, CDF）是描述**连续随机变量**统计行为的两个核心工具。PDF 描述了随机变量在各个取值点附近的"相对可能性密度"——曲线下方面积代表概率；CDF 则描述了随机变量取值不超过某一阈值的累积概率，是从 $-\infty$ 到该点的"面积累加器"。二者通过微积分基本定理紧密相连：**CDF 是 PDF 的积分，PDF 是 CDF 的导数**。在机器学习领域，从正态分布的似然函数到变分推断中的 ELBO，从神经网络的权重初始化到扩散模型中的噪声调度，PDF 与 CDF 无处不在。

---

## 二、从离散到连续：为什么需要 PDF 和 CDF

### 2.1 离散随机变量的概率质量函数（PMF）

> **【知识卡片：随机变量】**
> - **定义**：随机变量是一个将随机试验的每个可能结果映射为一个实数的函数，记作 $X: \Omega \to \mathbb{R}$，其中 $\Omega$ 是样本空间。
> - **公式**：$X(\omega) \in \mathbb{R}, \quad \omega \in \Omega$。
> - **本步作用**：为我们提供一个用数值描述随机现象的数学对象。

> **【小例子：随机变量】**
> 掷一枚公平骰子，样本空间 $\Omega = \{1, 2, 3, 4, 5, 6\}$。定义随机变量 $X$ 为骰子的点数，则 $X(1) = 1, X(2) = 2, \ldots, X(6) = 6$。这就是一个离散随机变量。

对于**离散随机变量**（取值有限或可列无穷），我们用**概率质量函数**（Probability Mass Function, PMF）来描述其分布：

$$
p_X(x) \triangleq P(X = x), \quad x \in \{x_1, x_2, x_3, \ldots\}
$$

PMF 满足两个基本性质：

$$
\text{(非负性)} \quad p_X(x) \geq 0, \quad \forall x \in \mathcal{X}
$$

$$
\text{(归一化)} \quad \sum_{x \in \mathcal{X}} p_X(x) = 1
$$

> **【知识卡片：概率质量函数（PMF）】**
> - **定义**：PMF 给出了离散随机变量取某个特定值的概率，即"质量"集中在离散点上。
> - **公式**：$p_X(x) = P(X = x)$，满足 $p_X(x) \geq 0$ 且 $\sum_{x} p_X(x) = 1$。
> - **本步作用**：描述离散随机变量的概率分布，是理解连续随机变量的起点。

> **【小例子：PMF】**
> 公平骰子的 PMF 为 $p_X(x) = \frac{1}{6}$，$x \in \{1, 2, 3, 4, 5, 6\}$。则 $P(X = 3) = \frac{1}{6}$，且 $\sum_{x=1}^{6} \frac{1}{6} = 1$。

**关键问题**：当我们尝试将离散随机变量的取值变得越来越密集（例如从整数点扩展到有理数点，再到所有实数），PMF 会面临什么困境？

### 2.2 连续随机变量的困境：单点概率为零

> **【知识卡片：连续随机变量】**
> - **定义**：连续随机变量是在一个不可数无穷集合（通常是实数区间）上取值的随机变量，其可能取值无法一一列举。
> - **公式**：$X: \Omega \to \mathbb{R}$，且 $X$ 的取值构成一个连续区间，如 $X(\omega) \in [a, b]$ 或 $\mathbb{R}$。
> - **本步作用**：将随机变量的概念从离散点扩展到连续区间，为描述物理量（如温度、长度、时间）等提供数学框架。

考虑一个连续随机变量 $X$ 在区间 $[0, 1]$ 上"均匀"取值。由于取值可能性相等且不可数无穷，对于任意一个具体值 $x \in [0, 1]$：

$$
P(X = x) = 0
$$

**推导依据**：假设 $P(X = x) = \epsilon > 0$ 对所有 $x \in [0, 1]$ 成立，则对任意 $N$ 个不同点 $x_1, \ldots, x_N \in [0, 1]$，由概率的有限可加性（见下方知识卡片）：

$$
P\left(\bigcup_{i=1}^{N} \{X = x_i\}\right) = \sum_{i=1}^{N} P(X = x_i) = N \cdot \epsilon
$$

取 $N > 1/\epsilon$ 即可使右端超过 1，与概率公理矛盾。因此对单点 $x$ 必有 $P(X = x) = 0$。

**困境总结**：对连续随机变量，PMF 退化为处处零的平凡函数，完全失去了描述能力。我们需要一种新的工具。

> **【知识卡片：概率的可数可加性（Kolmogorov 第三公理）】**
> - **定义**：对于可数无穷多个两两互斥的事件 $\{A_i\}_{i=1}^{\infty}$（即 $A_i \cap A_j = \emptyset$ 对 $i \neq j$），其并集的概率等于各事件概率之和。
> - **公式**：$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$。
> - **本步作用**：确保概率测度的数学自洽性，是推导连续随机变量单点概率为零的关键依据。

> **【小例子：可数可加性】**
> 设 $A_i = \{X = x_i\}$ 且 $x_i \neq x_j$（$i \neq j$），则 $P(X \in \{x_1, x_2, \ldots\}) = \sum_{i=1}^{\infty} P(X = x_i)$。若每个 $P(X = x_i) > 0$，则无穷和可能发散到大于 1，违反概率公理。

---

## 三、累积分布函数（CDF）：定义与核心性质

### 3.1 CDF 的定义

CDF 是同时适用于**离散**和**连续**随机变量的统一描述工具。它的核心思想是：不追问"取某一点的概率是多少"，而是问"取值不超过 $x$ 的概率是多少"。

> **【知识卡片：累积分布函数（CDF）】**
> - **定义**：CDF 是随机变量 $X$ 取值小于或等于某实数 $x$ 的概率，记作 $F_X(x)$。它对所有类型的随机变量（离散、连续、混合）都有良好定义。
> - **公式**：$F_X(x) \triangleq P(X \leq x), \quad x \in \mathbb{R}$。
> - **本步作用**：提供一个统一的概率描述框架，无论随机变量是离散还是连续，CDF 始终有意义。

> **【小例子：CDF】**
> 公平骰子的 CDF：$F_X(2.5) = P(X \leq 2.5) = P(X \in \{1, 2\}) = \frac{2}{6} = \frac{1}{3}$。注意 CDF 的自变量 $x$ 可以是任意实数，不限于 $X$ 的取值点。

**定义（累积分布函数）**：设 $X$ 是定义在概率空间 $(\Omega, \mathcal{F}, P)$ 上的随机变量，其 CDF 定义为：

$$
F_X(x) \triangleq P(X \leq x) = P(\{\omega \in \Omega : X(\omega) \leq x\}), \quad \forall x \in \mathbb{R}
$$

### 3.2 CDF 的四大基本性质

**定理**：任意随机变量（离散、连续或混合）的 CDF $F_X: \mathbb{R} \to [0, 1]$ 满足以下四条性质：

**性质 1（有界性 / 极限行为）**：

$$
\lim_{x \to -\infty} F_X(x) = 0, \quad \lim_{x \to +\infty} F_X(x) = 1
$$

**推导依据**：事件 $\{X \leq -\infty\} = \bigcap_{n=1}^{\infty} \{X \leq -n\}$ 是递减事件列的极限，由概率的**上连续性**（continuity from above）知 $P(\bigcap_{n=1}^{\infty} A_n) = \lim_{n \to \infty} P(A_n)$。由于 $\{X \leq -n\} \downarrow \emptyset$，故极限为 $0$。同理 $\{X \leq +n\} \uparrow \Omega$，故极限为 $1$。

> **【知识卡片：概率的连续性（Continuity of Probability）】**
> - **定义**：若事件列 $\{A_n\}$ 单调递增（$A_n \uparrow A$），则 $P(A_n) \uparrow P(A)$；若单调递减（$A_n \downarrow A$），则 $P(A_n) \downarrow P(A)$。
> - **公式**：$A_n \uparrow A \Rightarrow \lim_{n \to \infty} P(A_n) = P(A)$；$A_n \downarrow A \Rightarrow \lim_{n \to \infty} P(A_n) = P(A)$。
> - **本步作用**：将离散求和/极限操作与概率测度结合，用于证明 CDF 在无穷远处的极限行为。

**性质 2（单调不减）**：

$$
\forall x_1 < x_2 \in \mathbb{R}: \quad F_X(x_1) \leq F_X(x_2)
$$

**推导依据**：设 $x_1 < x_2$，则 $\{X \leq x_1\} \subseteq \{X \leq x_2\}$（若 $X$ 不超过 $x_1$，则必然不超过更大的 $x_2$）。由概率的**单调性**（monotonicity），$A \subseteq B \Rightarrow P(A) \leq P(B)$，即得证。

> **【知识卡片：概率的单调性】**
> - **定义**：若事件 $A$ 是事件 $B$ 的子集（$A \subseteq B$），则 $A$ 的概率不超过 $B$ 的概率。
> - **公式**：$A \subseteq B \Rightarrow P(A) \leq P(B)$。
> - **本步作用**：建立集合包含关系与概率大小之间的对应，是证明 CDF 单调不减的直接工具。

**性质 3（右连续性）**：

$$
\lim_{h \to 0^+} F_X(x + h) = F_X(x), \quad \forall x \in \mathbb{R}
$$

**推导依据**：当 $h \to 0^+$ 时，事件 $\{X \leq x + h\} \downarrow \{X \leq x\}$（递减趋于）。由概率的上连续性，$\lim_{h \to 0^+} P(X \leq x + h) = P(X \leq x)$。

**性质 4（区间概率计算）**：

$$
\forall a < b \in \mathbb{R}: \quad P(a < X \leq b) = F_X(b) - F_X(a)
$$

**推导依据**：事件 $\{X \leq b\} = \{X \leq a\} \cup \{a < X \leq b\}$，且右侧两事件互斥。由概率的**有限可加性**：

$$
P(X \leq b) = P(X \leq a) + P(a < X \leq b)
$$

整理即得 $P(a < X \leq b) = F_X(b) - F_X(a)$。

![CDF的几何意义：左图展示PDF下从负无穷到a的面积即为F(a)；中图展示CDF曲线本身；右图展示CDF的四大关键性质](./fig_2_cdf_geometric_meaning.png)

> **【小例子：CDF 计算区间概率】**
> 标准正态分布中，$F_X(0) = 0.5$，$F_X(1) \approx 0.841$。则 $P(0 < X \leq 1) = F_X(1) - F_X(0) = 0.841 - 0.5 = 0.341$。即 $X$ 落在 $(0, 1]$ 区间内的概率约为 $34.1\%$。

---

## 四、概率密度函数（PDF）：从 CDF 到密度

### 4.1 为什么 CDF 还不够？

CDF 已经能够完整描述随机变量的概率分布，但在实际应用中，我们还需要回答更精细的问题：

- 随机变量在 $x = 0$ 附近比在 $x = 5$ 附近"更可能出现"，这种**相对可能性**如何用数学表达？
- 如何直观地"看出"分布的形状（峰值、对称性、尾部厚薄）？
- 在机器学习中，如何写出连续随机变量的**似然函数**用于参数估计？

这些问题引导我们引入概率密度函数。

### 4.2 PDF 的严格定义

> **【知识卡片：几乎处处（Almost Everywhere, a.e.）】**
> - **定义**：一个性质"几乎处处成立"是指该性质在除了一个测度为零的集合之外的所有点上都成立。直观地说，"例外点"太少，不影响积分结果。
> - **公式**：性质 $P(x)$ 几乎处处成立 $\Leftrightarrow$ $P(\{x : P(x) \text{ 不成立}\}) = 0$。
> - **本步作用**：允许我们在不改变 CDF 的前提下对 PDF 进行逐点定义，因为修改单个点的密度值不影响任何区间概率。

**定义（概率密度函数）**：设 $X$ 为连续随机变量，其 CDF 为 $F_X(x)$。若存在一个非负可积函数 $f_X: \mathbb{R} \to [0, +\infty)$，使得：

$$
F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt, \quad \forall x \in \mathbb{R}
$$

则称 $f_X(x)$ 为 $X$ 的**概率密度函数**（PDF）。

**等价表述**：若 $F_X(x)$ 在点 $x$ 处可导，则：

$$
f_X(x) = \frac{d}{dx} F_X(x) = \lim_{h \to 0} \frac{F_X(x + h) - F_X(x)}{h}
$$

> **【知识卡片：微积分基本定理（Fundamental Theorem of Calculus, FTC）】**
> - **定义**：FTC 由两部分组成：(1) 若 $f$ 连续，则 $F(x) = \int_{a}^{x} f(t) dt$ 的导数是 $f(x)$；(2) 若 $F$ 是 $f$ 的原函数，则 $\int_{a}^{b} f(x) dx = F(b) - F(a)$。
> - **公式**：$\frac{d}{dx} \int_{a}^{x} f(t) \, dt = f(x)$；$\int_{a}^{b} f(x) \, dx = F(b) - F(a)$。
> - **本步作用**：建立 CDF 与 PDF 之间的微积分桥梁——PDF 是 CDF 的导数，CDF 是 PDF 的积分。

### 4.3 PDF 的核心性质

由定义可直接推出 PDF 的三条基本性质：

**性质 1（非负性）**：

$$
f_X(x) \geq 0, \quad \forall x \in \mathbb{R}
$$

**推导依据**：由 CDF 的单调不减性，$F_X(x + h) \geq F_X(x)$ 对 $h > 0$ 成立，故差商 $\frac{F_X(x+h) - F_X(x)}{h} \geq 0$。取极限 $h \to 0^+$ 即得 $f_X(x) = F_X'(x) \geq 0$。

**性质 2（归一化）**：

$$
\int_{-\infty}^{+\infty} f_X(x) \, dx = 1
$$

**推导依据**：由 CDF 定义及性质 1 的极限行为：

$$
\int_{-\infty}^{+\infty} f_X(x) \, dx = \lim_{b \to +\infty} \int_{-\infty}^{b} f_X(x) \, dx = \lim_{b \to +\infty} F_X(b) = 1
$$

**性质 3（区间概率 = 面积）**：对任意 $a < b$：

$$
P(a \leq X \leq b) = \int_{a}^{b} f_X(x) \, dx = F_X(b) - F_X(a)
$$

**推导依据**：

$$
P(a \leq X \leq b) = P(X \leq b) - P(X < a) = F_X(b) - F_X(a) = \int_{a}^{b} f_X(x) \, dx
$$

其中第一步由 CDF 的区间概率公式（性质 4），第二步由 FTC 的第二部分。

### 4.4 关键直观理解：密度 ≠ 概率

> **【知识卡片：概率 vs 概率密度】**
> - **定义**：概率 $P(X \in A)$ 是一个无量纲的数（在 $[0, 1]$ 内）；概率密度 $f_X(x)$ 是一个有量纲的量，单位是"概率 / 单位长度"。
> - **公式**：$P(x \leq X \leq x + dx) = f_X(x) \, dx$（对无穷小区间）。
> - **本步作用**：澄清一个常见误解——PDF 在某点的高度**不是**该点的概率，而是该点附近"单位区间内的概率集中度"。

> **【小例子：密度 vs 概率】**
> 设 $X \sim \mathcal{N}(0, 1)$，则 $f_X(0) = \frac{1}{\sqrt{2\pi}} \approx 0.399$。这**不是** $P(X = 0)$（实际上 $P(X = 0) = 0$），而是说在 $0$ 附近一个极窄区间 $[-0.001, 0.001]$ 内，概率约为 $f_X(0) \times 0.002 \approx 0.399 \times 0.002 \approx 0.0008$。

![从离散PMF到连续PDF的过渡：左图为离散PMF柱状图，中图为更密集的离散点，右图为连续PDF曲线。注意在连续情况下，曲线高度不等于概率，曲线下方面积才是概率](./fig_1_pmf_to_pdf_transition.png)

---

## 五、PDF 与 CDF 的完整互推关系

### 5.1 关系定理

**定理（PDF-CDF 等价关系）**：设 $X$ 为连续随机变量，CDF 为 $F_X(x)$，PDF 为 $f_X(x)$，则：

$$
\boxed{F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt \quad \Longleftrightarrow \quad f_X(x) = \frac{d}{dx} F_X(x)}
$$

**证明（正向 $\Rightarrow$）**：由 PDF 的定义 $F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt$，若 $f_X$ 在 $x$ 处连续，直接应用 FTC 第一部分得 $F_X'(x) = f_X(x)$。

**证明（反向 $\Leftarrow$）**：由 $f_X(x) = F_X'(x)$，两边从 $-\infty$ 到 $x$ 积分：

$$
\int_{-\infty}^{x} f_X(t) \, dt = \int_{-\infty}^{x} F_X'(t) \, dt = F_X(x) - \underbrace{F_X(-\infty)}_{= 0} = F_X(x)
$$

**推导依据**：微积分基本定理第二部分，以及 CDF 性质 1（$F_X(-\infty) = 0$）。

### 5.2 区间概率的两种计算方式

对任意 $a < b$，区间概率可通过 CDF 或 PDF 两种方式计算：

**方式一（CDF 差分）**：

$$
P(a \leq X \leq b) = F_X(b) - F_X(a)
$$

**方式二（PDF 积分）**：

$$
P(a \leq X \leq b) = \int_{a}^{b} f_X(x) \, dx
$$

两种方式的等价性由 FTC 保证：

$$
\int_{a}^{b} f_X(x) \, dx = \int_{-\infty}^{b} f_X(x) \, dx - \int_{-\infty}^{a} f_X(x) \, dx = F_X(b) - F_X(a)
$$

![PDF与CDF的微积分关系：左上展示PDF和CDF同轴对比；右上展示PDF等于CDF的导数（数值验证）；左下展示CDF等于PDF的积分（数值验证）；右下展示区间概率P(a<=X<=b)=F(b)-F(a)的几何解释](./fig_3_pdf_cdf_calculus_relation.png)

> **【小例子：PDF-CDF 互推】**
> 设指数分布的 CDF 为 $F_X(x) = 1 - e^{-\lambda x}$（$x \geq 0$），则其 PDF 为：
> $$f_X(x) = \frac{d}{dx}(1 - e^{-\lambda x}) = \lambda e^{-\lambda x}$$
> 反过来验证：$\int_{0}^{x} \lambda e^{-\lambda t} dt = [-e^{-\lambda t}]_{0}^{x} = 1 - e^{-\lambda x} = F_X(x)$。

---

## 六、典型连续分布的 PDF 与 CDF 实例

![六种常见概率分布的PDF（蓝色实线）和CDF（绿色虚线）对比：正态分布、柯西分布、指数分布、均匀分布、拉普拉斯分布、逻辑分布](./fig_4_common_distributions.png)

### 6.1 正态（高斯）分布 $\mathcal{N}(\mu, \sigma^2)$

正态分布是机器学习中最核心的连续分布，广泛用于权重初始化、噪声建模、变分推断等场景。

**PDF**：

$$
f_X(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}, \; \sigma^2 > 0
$$

**CDF**：

$$
F_X(x \mid \mu, \sigma^2) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x - \mu}{\sigma\sqrt{2}}\right)\right]
$$

其中 $\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} dt$ 是**误差函数**，没有初等闭式表达。

> **【知识卡片：误差函数（Error Function, erf）】**
> - **定义**：erf 是高斯函数 $e^{-t^2}$ 从 $0$ 到 $z$ 的定积分的归一化形式，用于表达正态分布的 CDF。
> - **公式**：$\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} \, dt$，满足 $\text{erf}(0) = 0$，$\text{erf}(+\infty) = 1$。
> - **本步作用**：将正态分布的 CDF 表示为可数值计算的函数形式，是许多统计检验和采样的基础。

### 6.2 均匀分布 $\mathcal{U}(a, b)$

**PDF**：

$$
f_X(x \mid a, b) = \begin{cases} \displaystyle\frac{1}{b - a}, & a \leq x \leq b \\[8pt] 0, & \text{otherwise} \end{cases}
$$

**CDF**：

$$
F_X(x \mid a, b) = \begin{cases} 0, & x < a \\[8pt] \displaystyle\frac{x - a}{b - a}, & a \leq x \leq b \\[8pt] 1, & x > b \end{cases}
$$

### 6.3 指数分布 $\text{Exp}(\lambda)$

常用于描述等待时间、寿命分析，也是泊松过程中事件间隔时间的分布。

**PDF**：

$$
f_X(x \mid \lambda) = \lambda e^{-\lambda x}, \quad x \geq 0, \; \lambda > 0
$$

**CDF**：

$$
F_X(x \mid \lambda) = 1 - e^{-\lambda x}, \quad x \geq 0
$$

### 6.4 分布之间的关系速查

| 分布 | PDF $f_X(x)$ | CDF $F_X(x)$ | 典型应用场景 |
|------|-------------|-------------|-------------|
| 正态 $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}} e^{-(x-\mu)^2/(2\sigma^2)}$ | $\frac{1}{2}[1 + \text{erf}(\frac{x-\mu}{\sigma\sqrt{2}})]$ | 中心极限定理、噪声建模、VAE 先验 |
| 均匀 $\mathcal{U}(a, b)$ | $\frac{1}{b-a}$（区间上） | $\frac{x-a}{b-a}$（区间上） | 随机初始化、随机采样 |
| 指数 $\text{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$ | $1 - e^{-\lambda x}$ | 等待时间、可靠性分析 |
| 拉普拉斯 $\text{Lap}(\mu, b)$ | $\frac{1}{2b} e^{-\vert x-\mu \vert/b}$ | $\frac{1}{2} + \frac{1}{2}\text{sgn}(x-\mu)\bigl(1 - e^{-\vert x-\mu \vert/b}\bigr)$ | L1 正则先验、稀疏编码 |

---

## 七、涉及的基本数学知识清单

| 概念名称 | 在本推导中的具体作用 | 一句话定义或公式表达 |
|---------|---------------------|---------------------|
| 随机变量 | 将随机试验映射为实数的函数，是 PDF/CDF 的自变量 | $X: \Omega \to \mathbb{R}$ |
| 概率质量函数 (PMF) | 描述离散随机变量的概率分布，是理解 PDF 的起点 | $p_X(x) = P(X = x)$ |
| 累积分布函数 (CDF) | 统一描述所有类型随机变量的"累积概率" | $F_X(x) = P(X \leq x)$ |
| 概率密度函数 (PDF) | 描述连续随机变量的"概率密度"，曲线下面积 = 概率 | $f_X(x) = \frac{d}{dx}F_X(x)$ |
| 概率的单调性 | 证明 CDF 单调不减 | $A \subseteq B \Rightarrow P(A) \leq P(B)$ |
| 概率的可数可加性 | 推导连续随机变量单点概率为零 | $P(\bigcup_{i=1}^{\infty} A_i) = \sum_{i=1}^{\infty} P(A_i)$ |
| 概率的连续性 | 证明 CDF 的极限行为和右连续性 | $A_n \uparrow A \Rightarrow P(A_n) \uparrow P(A)$ |
| 微积分基本定理 (FTC) | 建立 PDF 与 CDF 之间的微积分互推关系 | $\frac{d}{dx}\int_a^x f(t)dt = f(x)$ |
| 几乎处处 (a.e.) | 允许在零测集上修改 PDF 而不影响分布 | 性质在除零测集外的所有点成立 |
| 误差函数 (erf) | 表达正态分布 CDF 的不可初等积分 | $\text{erf}(z) = \frac{2}{\sqrt{\pi}}\int_0^z e^{-t^2}dt$ |

---


