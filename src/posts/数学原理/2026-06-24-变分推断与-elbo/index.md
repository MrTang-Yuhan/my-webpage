---
layout: post.njk
post_id: 2026-06-24-变分推断与-elbo
archive: 数学原理
title: 变分推断与 ELBO
date: 2026-06-24
tags:
  - post
---
# 变分推断与 ELBO (Evidence Lower Bound)

## 一、问题设定：为什么需要 ELBO？

我们有一个**生成模型**：
- 观测数据 $X$（如图像、文本、传感器读数）
- 隐变量 $Z$（如潜在表示、类别标签、未观测的状态）

模型定义了：
- 先验 $p(Z)$：隐变量的自然分布
- 似然 $p(X \mid Z)$：给定隐变量时生成观测数据的概率

**目标**：计算**证据**（evidence），即观测数据的对数边际似然：
$$
\log p(X) = \log \int p(X \mid Z)  p(Z)  dZ
$$

**困难**：积分通常涉及高维空间，$p(X \mid Z)$ 和 $p(Z)$ 的乘积对 $Z$ 积分**没有解析解**，直接数值积分也**不可行**（维度灾难）。

**解决方案**：引入一个**可计算的近似分布** $q(Z \mid X)$，用 ELBO 作为 $\log p(X)$ 的**可计算下界**。

> **这个公式在回答："我的模型生成这组观测数据 X  的概率有多大？"**
>
> 为什么这个公式"看着简单，算不出来"？
>
>这就是整个变分推断存在的意义：
>
> $$
> \int p(X \mid Z) p(Z) dZ
> $$
>
> - **高维积分**：$Z$ 通常是几百维甚至几千维的向量，数值积分（如网格采样）维度灾难；
> - **$p(X \mid Z)$ 复杂**：如果是神经网络（如 VAE 的 decoder），没有解析形式；
> - **$p(Z \mid X)$ 不可求**：由贝叶斯定理 $p(Z \mid X) = \frac{p(X \mid Z) p(Z)}{p(X)}$，分母正是这个积分——死锁。
> 
> **所以**：我们**无法直接计算** $\log p(X)$，才需要引入 ELBO 作为它的**下界近似**。




---

## 二、前置数学知识

### 1. 概率三兄弟的恒等关系

对任意两个随机变量 $X$ 和 $Z$：

| 名称 | 公式 | 备注 |
|------|------|------|
| **联合概率** | $p(X, Z)$ | 同时发生的概率 |
| **条件概率** | $p(X \mid Z) = \frac{p(X, Z)}{p(Z)}$ | 贝叶斯定义 |
| **边缘概率** | $p(X) = \int p(X, Z) dZ$ | 消去 $Z$ |

**核心恒等式**（推导中反复使用）：
$$
p(X, Z) = p(X \mid Z)  p(Z) = p(Z \mid X)p(X)
$$

### 2. 期望（Expectation）

若 $Z \sim q(Z)$，则：
$$
\mathbb{E}_{q(Z)}[f(Z)] = \int q(Z) f(Z)  dZ
$$

**关键性质**：
- **线性性**：$\mathbb{E}[a f + b g] = a \mathbb{E}[f] + b \mathbb{E}[g]$
- **常数提取**：若 $c$ 与 $Z$ 无关，则 $\mathbb{E}_{q}[c] = c \cdot \int q(Z) dZ = c$

### 3. 对数运算

$$
\log \frac{a}{b} = \log a - \log b, \qquad \log(ab) = \log a + \log b
$$

### 4. KL 散度（Kullback-Leibler Divergence）

衡量分布 $q$ 相对于 $p$ 的差异：

$$
\text{KL}(q \parallel p) = \int q(x) \log \frac{q(x)}{p(x)} dx = \mathbb{E}_{q}\left[\log \frac{q}{p}\right]
$$

**核心性质**：
$$
\text{KL}(q \parallel p) \geq 0, \quad \text{等号成立} \iff q = p
$$

> **证明概要**：[KL 散度](https://my-webpage-adu.pages.dev/posts/%E6%95%B0%E5%AD%A6%E5%8E%9F%E7%90%86/2026-06-01-kl-%E6%95%A3%E5%BA%A6/)

---

## 三、符号表（通用版）

| 符号 | 含义 |
|------|------|
| $X$ | 观测数据 |
| $Z$ | 隐变量（latent variable） |
| $p(X \mid Z)$ | 似然（likelihood） |
| $p(Z)$ | 先验（prior） |
| $p(Z \mid X)$ | **真实后验**（true posterior），通常不可计算 |
| $q(Z \mid X)$ | **变分后验**（variational posterior），可计算的近似分布 |
| $\log p(X)$ | **证据**（evidence），即对数边际似然，目标函数 |

---

## 四、核心推导：恒等变形法

> **目标**：证明 $\log p(X) \geq \mathbb{E}_{q}[\log p(X \mid Z)] - \text{KL}(q(Z \mid X) \parallel p(Z))$

---

### Step 1：将 $\log p(X)$ 写成关于 $q$ 的期望

$$
\log p(X) = \int q(Z \mid X)  \log p(X)  dZ
$$

**数学依据**：常数提取性质。因为 $\int q(Z \mid X) dZ = 1$（概率密度积分为1），且 $\log p(X)$ 与 $Z$ 无关，所以：
$$
\log p(X) = \log p(X) \cdot 1 = \log p(X) \cdot \int q(Z \mid X) dZ = \int q(Z \mid X) \log p(X) dZ
$$

---

### Step 2：用条件概率替换 $p(X)$

由联合概率与条件概率的关系 $p(X, Z) = p(Z \mid X) p(X)$，得：
$$
p(X) = \frac{p(X, Z)}{p(Z \mid X)}
$$

代入 Step 1：

$$
\log p(X) = \int q(Z \mid X)  \log \frac{p(X, Z)}{p(Z \mid X)}  dZ
$$

**数学依据**：概率恒等式 $p(X) = \frac{p(X,Z)}{p(Z|X)}$。

---

### Step 3：强行插入 $q/q$（凑出 KL 散度的形式）

$$
\log \frac{p(X, Z)}{p(Z \mid X)} = \log \left( \frac{p(X, Z)}{q(Z \mid X)} \cdot \frac{q(Z \mid X)}{p(Z \mid X)} \right) = \log \frac{p(X, Z)}{q(Z \mid X)} + \log \frac{q(Z \mid X)}{p(Z \mid X)}
$$

**数学依据**：对数乘法性质 $\log(ab) = \log a + \log b$。

---

### Step 4：拆成两个积分（积分的线性性）

$$
\log p(X) = \underbrace{\int q(Z \mid X) \log \frac{p(X, Z)}{q(Z \mid X)} dZ}_{\text{第一项}} + \underbrace{\int q(Z \mid X) \log \frac{q(Z \mid X)}{p(Z \mid X)} dZ}_{\text{第二项}}
$$

**数学依据**：$\int [f(Z) + g(Z)] dZ = \int f(Z) dZ + \int g(Z) dZ$。

---

### Step 5：识别两项的身份

**第二项** = **KL 散度**（直接对照定义）：
$$
\int q(Z \mid X) \log \frac{q(Z \mid X)}{p(Z \mid X)} dZ = \text{KL}(q(Z \mid X) \parallel p(Z \mid X))
$$

**第一项** = **ELBO**（Evidence Lower Bound）：
$$
\text{ELBO} = \int q(Z \mid X) \log \frac{p(X, Z)}{q(Z \mid X)} dZ = \mathbb{E}_{q}[\log p(X, Z)] - \mathbb{E}_{q}[\log q(Z \mid X)]
$$

**得到核心恒等式**（精确相等，不是不等式）：
$$
\boxed{\log p(X) = \text{ELBO} + \text{KL}(q(Z \mid X) \parallel p(Z \mid X))}
$$

---

### Step 6：利用 KL 的非负性得到下界

由 KL 散度的基本性质：
$$
\text{KL}(q(Z \mid X) \parallel p(Z \mid X)) \geq 0
$$

代入恒等式：
$$
\log p(X) = \text{ELBO} + \underbrace{\text{KL}(q \parallel p(Z \mid X))}_{\geq 0} \geq \text{ELBO}
$$

所以：
$$
\log p(X) \geq \text{ELBO}
$$

---

### Step 7：将 ELBO 展开为最终形式

$$
\text{ELBO} = \mathbb{E}_{q}[\log p(X, Z)] - \mathbb{E}_{q}[\log q(Z \mid X)]
$$

利用 $p(X, Z) = p(X \mid Z) p(Z)$：
$$
\begin{aligned}
\text{ELBO}=\mathbb{E}_{q}[\log p(X \mid Z)] + \mathbb{E}_{q}[\log p(Z)] - \mathbb{E}_{q}[\log q(Z \mid X)] \\
= \mathbb{E}_{q}[\log p(X \mid Z)] - \underbrace{\left( \mathbb{E}_{q}[\log q(Z \mid X)] - \mathbb{E}_{q}[\log p(Z)] \right)}_{\text{KL}(q(Z \mid X) \parallel p(Z))}
\end{aligned}
$$

最终得到标准 ELBO 公式：

$$
\boxed{\log p(X) \geq \mathbb{E}_{q(Z \mid X)}[\log p(X \mid Z)] - \text{KL}(q(Z \mid X) \parallel p(Z))}
$$



---

## 五、关键结论与洞察

### 1. 为什么叫 "Lower Bound"？
因为 $\log p(X) \geq \text{ELBO}$，ELBO 是真实对数边际似然的**下界**。我们无法直接算 $\log p(X)$，但可以算 ELBO，通过最大化 ELBO 来**间接最大化** $\log p(X)$。

### 2. 近似质量由谁决定？
从恒等式：
$$
\log p(X) = \text{ELBO} + \text{KL}(q \parallel p(Z \mid X))
$$

- $\log p(X)$ 是**与 $q$ 无关的常数**（它就是真实数据似然）；
- 因此 $\max_q \text{ELBO} \iff \min_q \text{KL}(q(Z \mid X) \parallel p(Z \mid X))$；
> 因为 $\log p(X)$ 是真实数据的对数似然，它就是客观存在的那个数，所以 KL 越小，ELBO 就越大

- **变分推断的本质**：用可计算的 $q$ 去逼近不可计算的真实后验 $p(Z \mid X)$。

### 3. ELBO 两项的通用解释

| ELBO 中的项 | 通用名称 | 通用含义 |
|-------------|---------|---------|
| $\mathbb{E}_{q}[\log p(X \mid Z)]$ | **期望对数似然 / 重构项** | $q$ 采样出的隐变量 $Z$ 对解释观测 $X$ 有多好 |
| $-\text{KL}(q(Z \mid X) \parallel p(Z))$ | **正则项 / 先验约束** | 惩罚 $q$ 偏离先验 $p(Z)$，防止后验过拟合到特定观测 |

**直觉**：ELBO 鼓励 $q$ 做两件事：
1. **解释数据**：采样出的 $Z$ 要能让 $p(X \mid Z)$ 高（重构好）；
2. **保持简洁**：$q$ 不能离先验 $p(Z)$ 太远（避免过拟合）。

---

## 六、变分推断的优化视角

在实际应用中（如 VAE、主题模型、潜变量序列模型），我们通常：

1. **参数化** $q(Z \mid X)$ 为一个神经网络 $q_\phi(Z \mid X)$（变分编码器）；
2. **参数化** $p(X \mid Z)$ 为另一个神经网络 $p_\theta(X \mid Z)$（解码器）；
3. **最大化 ELBO** 关于 $\phi$（变分参数）和 $\theta$（生成参数）：
   $$
\max_{\phi, \theta} \; \mathbb{E}_{q_\phi}[\log p_\theta(X \mid Z)] - \text{KL}(q_\phi(Z \mid X) \parallel p(Z))
$$

**注意**：若 $q$ 被简化为确定性映射（如最近邻、硬分配），KL 项消失或退化，ELBO 退化为单纯的期望对数似然——这在工程上常见，但理论上损失了后验近似的严谨性。

---

## 七、一句话总结

> **ELBO 是数学上的"替身策略"**：因为真实后验 $p(Z \mid X)$ 无法直接计算，我们通过恒等变形把 $\log p(X)$ 拆成"好算的部分（ELBO）"与"替身和真身的差距（KL）"。最大化 ELBO 等价于让变分分布 $q$ 尽可能逼近真实后验，从而在无法直接优化目标时，优化一个可控的下界。
