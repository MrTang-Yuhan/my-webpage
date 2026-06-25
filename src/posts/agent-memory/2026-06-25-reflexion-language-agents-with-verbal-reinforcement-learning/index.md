---
layout: post.njk
post_id: 2026-06-25-reflexion-language-agents-with-verbal-reinforcement-learning
archive: agent-memory
title: "Reflexion: Language Agents with Verbal Reinforcement Learning"
date: 2026-06-25
tags:
  - post
---
# Reflexion: Language Agents with Verbal Reinforcement Learning — 阅读笔记

> **论文信息**
> - **标题**: Reflexion: Language Agents with Verbal Reinforcement Learning
> - **作者**: Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao
> - **机构**: Northeastern University, MIT, Princeton University
> - **arXiv ID**: 2303.11366
> - **代码仓库**: https://github.com/noahshinn024/reflexion

---

## 一、研究背景与核心问题

### 1.1 研究动机

近年来，以大语言模型（Large Language Model, LLM）为核心构建的自主决策智能体（autonomous decision-making agents）展现出巨大潜力。ReAct [30]、SayCan [1]、Toolformer [22]、HuggingGPT [23]、generative agents [19] 以及 WebGPT [17] 等工作已证明，基于 LLM 的 Agent 能够生成文本与"动作"（actions），并通过 API 调用在外部环境（如游戏、编译器、网页）中执行这些动作，从而完成目标驱动的任务。

然而，这类方法面临一个根本性瓶颈：**它们依赖于海量参数的预训练模型，却仅能通过上下文示例（in-context examples）来"教学"**。更传统的优化方案——如基于梯度下降的强化学习（Reinforcement Learning, RL）——需要大量训练样本与昂贵的模型微调计算成本，这使得在保留 LLM 通用能力的同时让其快速适应特定任务变得极为困难。

### 1.2 核心问题

> **如何设计一种轻量级的强化机制，使 LLM-based Agent 能够从试错中快速学习，且无需更新模型权重？**

该问题的挑战在于：
1. **学习效率**: 传统 RL 方法需要大量交互样本才能收敛，而 LLM 推理成本高昂；
2. **反馈利用**: 环境提供的反馈往往是稀疏的（如二元成功/失败信号），如何将其转化为可操作的改进方向涉及"信用分配问题"（credit assignment problem [25]）；
3. **知识持久化**: 如何让 Agent 在多轮试验中积累并有效利用经验，避免重复犯相同错误。

### 1.3 核心思想

Reflexion 的核心洞察是：**将环境的二元或标量反馈转化为自然语言形式的文本摘要（verbal feedback），作为额外上下文注入 LLM Agent 的下一次试验中**。这种自我反思式反馈充当"语义梯度信号"（semantic gradient signal），为 Agent 提供具体的改进方向——类似于人类通过反思过往失败来制定更优的下次尝试计划。

![图1: Reflexion 在决策、编程和推理任务上的应用概览](img/fig1_overview.png)

**图1**展示了 Reflexion 的统一框架如何应用于三类截然不同的任务：
- **决策任务**（左列）: Agent 在房间中寻找物品并操作，通过反思修正行动计划；
- **编程任务**（中列）: Agent 根据自然语言描述生成代码，通过单元测试结果反思并修复实现错误；
- **推理任务**（右列）: Agent 回答知识密集型问题，通过反思修正推理路径。

---

## 二、方法框架：Reflexion

### 2.1 总体架构

Reflexion 采用模块化设计，由三个不同的模型组成，它们协同工作形成一个迭代优化闭环：

| 组件 | 符号 | 功能 |
|------|------|------|
| **Actor** | $M_a$ | 基于 LLM，生成文本与动作 |
| **Evaluator** | $M_e$ | 评估 Actor 输出，计算奖励分数 |
| **Self-Reflection** | $M_{sr}$ | 生成语言形式的强化线索，辅助 Actor 自我改进 |

![图2: Reflexion 框架图与强化算法伪代码](img/fig2_framework.png)

**图2(a)** 直观展示了三个组件之间的信息流：
- Actor 与环境交互产生"轨迹"（Trajectory，短期记忆）；
- Evaluator 接收外部/内部反馈对轨迹进行评分；
- Self-Reflection 模型将评估信号"放大"为自然语言经验摘要，存入"经验"（Experience，长期记忆）；
- Actor 在下一轮决策中同时 conditioning 于短期轨迹与长期经验。

### 2.2 算法流程

Reflexion 被形式化为一个迭代优化过程（**图2(b)** 伪代码）：

**初始化阶段**:
- 初始化 Actor $M_a$、Evaluator $M_e$、Self-Reflection $M_{sr}$
- 初始化策略 $\pi_\theta(a_i \mid s_i)$，其中参数 $\theta = \\{M_a, \text{mem}\\}$
- 生成初始轨迹 $\tau_0$ 并用 $M_e$ 评估
- 用 $M_{sr}$ 生成初始自我反思 $sr_0$，存入记忆 $\text{mem} \leftarrow [sr_0]$

**迭代循环** (设试验轮次 $t = 0, 1, 2, \ldots$):
1. **策略生成**: 使用当前策略 $\pi_\theta$ 生成轨迹 $\tau_t = [a_0, o_0, \ldots, a_i, o_i]$，其中 $a_i$ 为动作，$o_i$ 为环境观测；
2. **评估**: 用 Evaluator $M_e$ 对 $\tau_t$ 评分，得到标量奖励 $r_t = M_e(\tau_t)$；
3. **自我反思**: Self-Reflection 模型 $M_{sr}$ 分析 $\\{\tau_t, r_t\\}$，生成语言反馈 $sr_t$；
4. **记忆更新**: 将 $sr_t$ 追加至长期记忆 $\text{mem}$；
5. **终止判断**: 若 $M_e$ 判定 $\tau_t$ 正确或 $t$ 达到最大试验次数，则停止。

> **关键约束**: 实际实现中，记忆容量受限于 LLM 的最大上下文长度，因此设置最大存储经验数 $\Omega$（通常取 1–3），采用滑动窗口机制保留最近的自我反思。

### 2.3 三个核心组件详解

#### 2.3.1 Actor ($M_a$)

Actor 建立在大型语言模型之上，通过提示工程（prompting）使其基于状态观测生成必要的文本与动作。类比传统基于策略的强化学习设定，Actor 在时刻 $t$ 从当前策略 $\pi_\theta$ 中采样动作 $a_t$，并从环境接收观测 $o_t$。

论文探索了多种 Actor 模型变体：
- **Chain-of-Thought (CoT)** [26]: 通过"逐步思考"提示实现逐步推理；
- **ReAct** [30]: 将推理（Reasoning）与动作（Acting）交错进行，在 long-horizon 决策中表现优异。

Actor 还配备了一个记忆组件 $\text{mem}$，提供额外上下文。该设计灵感来源于 Brooks 等人 [3] 提出的上下文策略迭代（in-context policy iteration）方法。

#### 2.3.2 Evaluator ($M_e$)

Evaluator 负责评估 Actor 生成输出的质量。由于为语义空间定义有效的价值函数与奖励函数具有挑战性，论文探索了 Evaluator 的多种变体：

| 任务类型 | 评估方式 | 说明 |
|---------|---------|------|
| 推理任务 | 精确匹配评分 (Exact Match, EM) | 检查生成输出是否与预期答案一致 |
| 决策任务 | 预定义启发式函数 | 针对特定评估标准定制 |
| 通用 | LLM 自身作为 Evaluator | 用另一个 LLM 实例为决策与编程任务生成奖励 |

#### 2.3.3 Self-Reflection ($M_{sr}$)

Self-Reflection 模型（同样实例化为 LLM）是 Reflexion 框架的关键创新。给定稀疏的奖励信号（如二元成功/失败状态）、当前轨迹 $\tau_t$ 及其持久记忆 $\text{mem}$，该模型生成细致且具体的语言反馈。

**关键机制**: 在多步决策任务中，当 Agent 接收到失败信号时，$M_{sr}$ 可以推断出特定动作 $a_i$ 导致了后续的错误动作 $a_{i+1}$ 和 $a_{i+2}$。随后 Agent 可以用自然语言陈述"本应采取不同动作 $a'_i$"，并将这一经验存入记忆。在后续试验中，Agent 可以利用过往经验在时刻 $t$ 选择动作 $a'_i$。

### 2.4 记忆机制

Reflexion 的记忆系统包含两个核心组件：

- **短期记忆（Short-term memory）**: 轨迹历史（trajectory history），即在当前试验中 Agent 执行的动作序列与观测序列，类似于人类对近期细节的记忆；
- **长期记忆（Long-term memory）**: Self-Reflection 模型的输出，即自然语言形式的反思经验，类似于人类从过往经历中提炼的重要教训。

两者协同工作，为 Actor 提供既具体又受多轮试验学习影响的上下文——这是 Reflexion Agent 相较于其他 LLM 动作选择方法的关键优势。

> **长期记忆容量限制**: 由于 LLM 存在最大上下文长度限制，记忆 $\text{mem}$ 通常被限制为最多存储 $\Omega \in \\{1, 2, 3\\}$ 条反思经验，超出时采用滑动窗口丢弃最早的经验。

---

## 三、核心概念深度拆解

### 3.1 策略参数化与"语义梯度"

> 正文首次提及位置：第1节 Introduction

Reflexion 提出了一种全新的策略参数化方式：策略不仅由 LLM 的权重参数定义，还由 Agent 的记忆编码（memory encoding）共同决定。（其形式化定义与原理详见下方补充框。）

> **补充框：Reflexion 策略参数化的四层次解析**
>
> **① 形式化定义（它是什么）**
>
> 传统强化学习中，策略通常表示为条件概率分布 $\pi_\theta(a \mid s)$，其中 $\theta$ 为可学习的参数向量（如神经网络权重）。Reflexion 将策略参数化为：
> $$\pi_\theta(a_i \mid s_i), \quad \theta = \\{M_a, \text{mem}\\}$$
> 其中 $M_a$ 为固定的预训练 LLM，$\text{mem} = [sr_0, sr_1, \ldots, sr_{t-1}]$ 为自然语言形式的反思经验序列。策略的输出同时依赖于模型 $M_a$ 的内部知识与外部记忆 $\text{mem}$ 中的经验。
>
> **② 工程变形（实际怎么做）**
>
> 在具体实现中，策略执行等价于构造一个包含任务指令、历史轨迹（短期记忆）和反思经验（长期记忆）的提示文本 $p$，然后调用 LLM 生成下一动作：
> 

$$ 
a_t \sim M_a(\, \cdot \mid p = [\text{task\_instruction}, \tau_{t-1}, \text{mem}] \,) 
$$

> 这里 $\text{mem}$ 被限制为最近 $\Omega$ 条经验（通常 $\Omega = 1, 2, 3$）。没有梯度更新，没有权重微调——"学习"完全发生在提示层面。
>
> **③ 系统可行性（为什么能跑）**
>
> 从系统角度，Reflexion 的轻量性体现在三个维度：
> - **计算**: 每次试验仅需一次 LLM 推理调用（加上 Evaluator 和 Self-Reflection 的调用），计算复杂度为 $O(T \cdot C_{\text{LLM}})$，其中 $T$ 为试验轮数，$C_{\text{LLM}}$ 为单次推理成本；
> - **存储**: 长期记忆仅需存储文本字符串，远低于存储梯度或值函数所需的空间；
> - **可扩展性**: 由于不修改模型权重，Reflexion 可无缝应用于任何可通过 API 访问的 LLM，无需模型托管或训练基础设施。
>
> **④ 第一性原理（为什么必须这样）**
>
> 从信息论视角看，Reflexion 将稀疏的标量反馈 $r_t \in \\{0, 1\\}$ 转化为高维自然语言信号 $sr_t \in \mathcal{V}^*$（$\mathcal{V}$ 为词表），这一过程可视为一个"信息放大"操作。根据**信道容量理论**，标量奖励的信道容量极低（二元信道容量为 1 bit），难以承载复杂的信用分配信息；而自然语言的信道容量高得多，可以编码"哪个动作错了、为什么错、应该如何修正"等结构化信息。因此，从反馈效率的角度，将稀疏信号"放大"为语言反馈是一种必然选择。

---

### 3.2 轨迹（Trajectory）与奖励信号

> 正文首次提及位置：第3节 The Reflexion process

Reflexion 中的轨迹与奖励信号沿袭了标准 RL 的形式化，但具有其独特语义。（其形式化定义与原理详见下方补充框。）

> **补充框：轨迹与奖励信号的四层次解析**
>
> **① 形式化定义（它是什么）**
>
> **轨迹**（Trajectory）定义为 Agent 与环境交互的动作-观测序列：
> $$\tau_t = [a_0, o_0, a_1, o_1, \ldots, a_i, o_i]$$
> 其中 $a_k \in \mathcal{A}$ 为第 $k$ 步的动作（可以是文本生成、API 调用或决策动作），$o_k \in \mathcal{O}$ 为环境返回的观测（observation），$i$ 为轨迹长度（可能因任务而异）。
>
> **标量奖励**由 Evaluator 模型计算：
> $$r_t = M_e(\tau_t)$$
> 其中 $r_t \in \mathbb{R}$（实践中通常为二元值 $\\{0, 1\\}$ 或有界标量），$t$ 为试验轮次索引。
>
> **② 工程变形（实际怎么做）**
>
> 在不同任务中，轨迹与奖励的实现形式各异：
> - **决策任务（ALFWorld）**: 轨迹是文本形式的 Agent 思考-动作序列（如 `>think: I need to find a mug` $\to$ `>go to desk 1` $\to$ `Observation: On the desk you see...`），奖励为二元值（任务完成/失败）；
> - **编程任务**: 轨迹是代码实现与单元测试执行结果的拼接，奖励为二元值（所有测试通过/未通过）；
> - **推理任务**: 轨迹是问题-推理链-答案三元组，奖励由精确匹配（EM）评分决定。
>
> **③ 系统可行性（为什么能跑）**
>
> 标量奖励的计算成本极低（一次字符串比较或一次 LLM 调用），使得快速迭代成为可能。在 12 次试验的设置下，单个任务的总调用成本约为 $12 \times (C_{\text{actor}} + C_{\text{evaluator}} + C_{\text{reflection}})$。由于 Actor 与 Evaluator 可共享同一个底层 LLM（通过不同提示区分），实际部署时可复用计算资源。
>
> **④ 第一性原理（为什么必须这样）**
>
> 从**马尔可夫决策过程（Markov Decision Process, MDP）**的角度，Agent 的决策问题可形式化为五元组 $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$，其中：
> - $\mathcal{S}$ 为状态空间；
> - $\mathcal{A}$ 为动作空间；
> - $\mathcal{P}: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$ 为状态转移概率；
> - $\mathcal{R}: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ 为奖励函数；
> - $\gamma \in [0, 1]$ 为折扣因子。
>
> Reflexion 的创新在于：它将 MDP 中的策略改进步骤（通常通过策略梯度或值函数更新实现）替换为**基于自然语言的策略修正**。其理论合理性来源于**非参数策略迭代**（non-parametric policy iteration）框架：经验 $\text{mem}$ 充当了一个非参数化的价值函数近似器，而 Self-Reflection 则扮演了策略改进步骤的角色。

---

### 3.3 Self-Reflection 的信息放大机制

> 正文首次提及位置：第1节 Introduction 与第3节 Self-reflection

Reflexion 的核心创新是将稀疏的标量反馈"放大"（amplify）为丰富的自然语言经验摘要。这一信息放大机制是理解 Reflexion 有效性的关键。（其形式化定义与原理详见下方补充框。）

> **补充框：信息放大机制的四层次解析**
>
> **① 形式化定义（它是什么）**
>
> 信息放大操作定义为从 $\\{\tau_t, r_t\\}$ 到自然语言反思 $sr_t$ 的映射：
> $$sr_t = M_{sr}(\tau_t, r_t, \text{mem})$$
> 其中 $M_{sr}$ 为 Self-Reflection 模型（实例化为 LLM），$\tau_t$ 为当前轨迹，$r_t$ 为标量奖励，$\text{mem}$ 为历史反思记忆。输出 $sr_t$ 是描述失败原因与改进建议的自由格式文本。
>
> **② 工程变形（实际怎么做）**
>
> 在实际实现中，$M_{sr}$ 通过精心设计的提示工程完成上述映射。对于编程任务，典型的 $sr_t$ 格式为：
> ```
> The function failed on test case X because [reason]. 
> In the next trial, I should [actionable suggestion].
> ```
> 对于决策任务，$sr_t$ 可能描述为：
> ```
> I should have looked for the desklamp first, then the mug. 
> In the next trial, I will go to desk 1, find the lamp, then look for the mug.
> ```
>
> **③ 系统可行性（为什么能跑）**
>
> 信息放大机制的可行性依赖于 LLM 强大的上下文学习与推理能力：
> - 现代 LLM（GPT-3.5/GPT-4）能够理解长文本轨迹并定位关键错误点；
> - 自然语言的开放性使得反馈可以包含多维度的改进建议（如计划修正、工具使用策略调整、推理路径修正等）；
> - 记忆容量的限制（$\Omega \leq 3$）确保提示长度始终处于 LLM 的有效上下文窗口内。
>
> **④ 第一性原理（为什么必须这样）**
>
> 从**强化学习理论**的角度，传统方法面临一个根本性约束：**信用分配问题**（credit assignment problem）。当 Agent 在 100 步的轨迹中仅收到最终的二元奖励时，标准 RL 方法需要通过时间差分（TD）学习或策略梯度来估计每一步动作的贡献度——这在稀疏奖励设置下效率极低。
>
> Reflexion 的信息放大机制可视为一种**基于模型的信用分配**（model-based credit assignment）：LLM 利用其预训练获得的"世界知识"直接推断出错误原因，绕过了传统方法中需要大量样本估计值函数的步骤。从**贝叶斯推理**的角度看，$M_{sr}$ 实质上是在执行后验推断——给定观测到的失败轨迹，推断最可能的错误假设并生成修正策略。这使得 Reflexion 在仅有 $T \leq 12$ 次试验的条件下就能实现有效学习，而传统 RL 方法可能需要数万次交互。

---

## 四、实验验证

论文在三类不同任务上评估了 Reflexion：
1. **决策任务**（ALFWorld）: 测试长轨迹上的序列动作选择；
2. **推理任务**（HotPotQA）: 测试知识密集型单步生成改进；
3. **编程任务**（HumanEval, MBPP, LeetcodeHard）: 测试 Agent 有效使用编译器和解释器的能力。

### 4.1 序列决策：ALFWorld

#### 实验设置

ALFWorld [24] 是一组基于 TextWorld [8] 的文本环境，要求 Agent 在多步交互中完成家庭环境中的任务，如：
- 寻找隐藏物品（如在抽屉中寻找锅铲）；
- 移动物品（如将刀移至砧板）；
- 使用物品操作其他物品（如将番茄放入冰箱冷藏）。

实验在 134 个 ALFWorld 环境上进行，涵盖 6 种不同任务类型。Agent 采用 ReAct [30] 作为动作生成器。为实现完全自主行为，论文实现了两种自我评估技术：
- **LLM 自然语言分类**: 用 LLM 对任务完成状态进行二元分类；
- **手写启发式**: 若 Agent 连续 3 次执行相同动作并收到相同响应，或当前环境动作数超过 30（低效规划），则触发自我反思。

基线运行中，若建议自我反思则跳过反思过程、重置环境并开始新试验；Reflexion 运行中，Agent 利用自我反思找出错误、更新记忆、重置环境并开始新试验。记忆截断为最近 3 条自我反思。

#### 实验结果

![图3: ALFWorld 实验结果](img/fig3_alfworld_results.png)

**图3(a)** 显示了 134 个任务上的累积解决比例：

| 方法 | 12 次试验后完成的任务数 | 相对提升 |
|------|----------------------|---------|
| ReAct only | ~104 | 基线 |
| ReAct + Reflexion (Heuristic) | 130 | **+22%** |
| ReAct + Reflexion (GPT) | ~128 | **+19%** |

ReAct + Reflexion 在 12 次连续试验中学习解决额外任务，而 ReAct-only 的性能在第 6–7 次试验之间停止增长。

**图3(b)** 展示了失败轨迹的分类分析：
- **幻觉（Hallucination）**: ReAct-only Agent 收敛于 22% 的幻觉率且无长期恢复迹象；
- **低效规划（Inefficient planning）**: Reflexion 几乎完全消除了此类错误。

#### 分析

基线失败轨迹中的常见错误是：Agent 认为自己拥有某物品，但实际上并未持有。Agent 在长轨迹中继续执行多个动作，却无法回溯找到错误根源。Reflexion 通过将长失败轨迹蒸馏为可用作"自我提示"（self-hints）的相关经验来消除绝大部分此类错误。

长期记忆在 ALFWorld 中有两种主要帮助方式：
1. **早期错误识别**: 在长轨迹中定位早期错误，Agent 可建议新的动作选择甚至新的长期计划；
2. **系统性搜索**: 当房间中有过多表面/容器需要检查时，Agent 可利用多轮试验的经验记忆来彻底搜索房间。

学习曲线表明学习过程发生在若干经验积累之后——Agent 成功平衡了上述两种情况：在前两次试验之间出现即时跃升，随后 11 次试验中稳步增长至接近完美的表现。

![图5: ALFWorld 轨迹示例](img/fig5_alfworld_trajectory.png)

**图5**展示了一个具体的 ALFWorld 轨迹示例。Trial #1 中 Agent 因低效规划而失败（先找杯子再找台灯，而非按任务要求"用台灯检查杯子"）。在反思中，Agent 识别出应先寻找台灯再寻找杯子。Trial #2 中 Agent 成功修正推理路径并以简洁方式执行动作序列。

---

### 4.2 推理任务：HotPotQA

#### 实验设置

HotPotQA [28] 是一个基于 Wikipedia 的数据集，包含 113K 问答对，要求 Agent 解析内容并在多个支撑文档上进行推理。

为隔离推理能力的改进效果，论文实现了 Reflexion + Chain-of-Thought (CoT) [26] 的两种配置：
- $Q \to A$: 直接从问题生成答案；
- $Q, C_{gt} \to A$: 给定问题与真实上下文（ground truth context）生成答案。

由于 CoT 不是多步决策技术，论文将真实上下文 $C_{gt}$ 提供给 Agent，以隔离在长文本上的推理行为。

为测试整体问答能力（需要推理与动作选择），还实现了 Reflexion + ReAct Agent，该 Agent 使用 Wikipedia API 检索相关上下文，并通过逐步显式思考推断答案。

提示设置：
- CoT 实现：6-shot prompting
- ReAct：2-shot prompting
- Self-reflection：2-shot prompting

评估使用精确匹配（EM）评分提供二元成功信号。每次试验后使用自我反思循环放大二元信号，记忆大小为 3 条经验。

#### 实验结果

![图4: HotPotQA 实验结果](img/fig4_hotpotqa_results.png)

**图4(a)** 比较了 Reflexion 与 CoT 和 ReAct 的结合效果：

| 方法 | 首次试验准确率 | 多次试验后提升 |
|------|-------------|-------------|
| CoT only | ~0.35 | 无提升 |
| CoT + Reflexion | ~0.35 | **显著提升至 ~0.55** |
| ReAct only | ~0.38 | 无提升 |
| ReAct + Reflexion | ~0.38 | **显著提升至 ~0.55** |

关键发现：ReAct-only、CoT-only 和 CoT(GT)-only 实现在多次试验中**未能概率性地改进任何任务**，即基线方法在温度 0.7 设置下无法通过重新采样解决首次试验中失败的任务。

**图4(b)** 展示了 CoT(GT) 的推理专用改进：
- CoT(GT) only: 首次试验准确率约 0.68，后续无提升；
- CoT(GT) + Reflexion: 准确率提升至约 0.80（**绝对提升 12%**）。

值得注意的是，CoT(GT) Agent 即使拥有真实上下文，仍有 39% 的问题无法正确推断答案，但 Reflexion 帮助 Agent 在**无真实答案访问权限**的情况下修正错误，将准确率提升约 14%。

#### 消融实验：自我反思 vs 情景记忆

**图4(c)** 展示了关键的消融实验结果：

| 方法 | 描述 | 准确率 |
|------|------|--------|
| CoT (GT) only | 基线 | ~0.68 |
| CoT (GT) EPM | 加入情景记忆（仅最近轨迹） | ~0.73 (+5%) |
| CoT (GT) EPM + Reflexion | 加入完整自我反思 | **~0.80 (+12%)** |

实验设置：
- **EPM（Episodic Memory）**: 在提示中包含最近一次的完整轨迹，但不包含自我反思生成的语言解释；
- **EPM + Reflexion**: 在 EPM 基础上加入标准自我反思步骤。

结果分析：自我反思较情景记忆学习优势带来了 **8% 的绝对提升**，支持了"仅优化（refinement-only）方法不如自我反思引导的优化方法有效"的论点。这表明**用语言写成的第一人称口头解释对于迭代学习至关重要**—— Agent 通过自我解释来更好地识别错误并制定改进策略。

![图7: HotPotQA 推理示例](img/fig7_hotpotqa_example.png)

**图7**展示了一个 HotPotQA 的两试验示例。Trial #1 中 Agent 搜索了错误的查询 `"'Allo 'Allo!"` 导致无结果，反思后识别出应搜索该剧主角 Gorden Kaye 来找到其最知名的角色。Trial #2 中 Agent 成功修正搜索策略并给出正确答案。

---

### 4.3 编程任务

#### 实验设置

论文在 Python 和 Rust 两种语言上评估了基线与 Reflexion 方法，基准测试包括：

| 基准测试 | 语言 | 描述 |
|---------|------|------|
| HumanEval [6] | Python | 手写编程问题，评估函数体生成 |
| HumanEval-RS | Rust | 最难的 50 个 HumanEval Python 问题经 MultiPL-E [4] 翻译 |
| MBPP [2] | Python |  Mostly Basic Python Programming |
| MBPP-RS | Rust | MBPP 子集经 MultiPL-E 翻译 |
| LeetcodeHardGym (新) | Python | 40 道 LeetCode Hard 难度题目，发布于 GPT-4 训练数据截止日期（2022年10月8日）之后 |

编程任务为 Reflexion 提供了独特的自我评估机会：**自生成单元测试套件**。这使得 Reflexion 编程 Agent 符合 pass@1 准确率报告条件。

**测试套件生成流程**:
1. 使用 Chain-of-Thought prompting [26] 生成多样化、全面的测试及其自然语言描述；
2. 通过尝试为每个候选测试构建有效的抽象语法树（AST）来过滤语法有效的测试语句；
3. 从生成的单元测试集合中采样 $n$ 个测试组成测试套件 $T = \\{t_0, t_1, \ldots, t_n\\}$，$n$ 最大为 6。

Reflexion 编程 Agent 的学习循环与推理和决策 Agent 相同，最大记忆限制为 1 条经验。

#### 实验结果

![表1: 各基准测试 Pass@1 准确率](img/table1_pass_at_1.png)

**表1**汇总了不同模型-策略-语言组合下的 Pass@1 准确率：

| 基准 + 语言 | 先前 SOTA Pass@1 | SOTA Pass@1 | Reflexion Pass@1 |
|------------|-----------------|-------------|-----------------|
| HumanEval (PY) | 65.8 (CodeT + GPT-3.5) | 80.1 (GPT-4) | **91.0** |
| HumanEval (RS) | — | 60.0 (GPT-4) | **68.0** |
| MBPP (PY) | 67.7 (CodeT + Codex) | 80.1 (GPT-4) | 77.1 |
| MBPP (RS) | — | 70.9 (GPT-4) | **75.4** |
| LeetcodeHard (PY) | — | 7.5 (GPT-4) | **15.0** |

Reflexion 在 Python 和 Rust 的所有基准测试上均超越了所有基线准确率并设立了新的最先进水平，**唯一例外是 MBPP Python**。

![表2: 整体准确率与测试生成性能](img/table2_test_performance.png)

**表2**提供了超越 Pass@1 的详细分析，包括四种条件：
- **TP（True Positive）**: 单元测试通过，解决方案正确；
- **FN（False Negative）**: 单元测试失败，解决方案正确；
- **FP（False Positive）**: 单元测试通过，解决方案错误；
- **TN（True Negative）**: 单元测试失败，解决方案错误。

**关键发现**——MBPP Python 上 Reflexion 表现不佳的原因：

HumanEval 和 MBPP Python 的基线 Pass@1 准确率相近（82% vs 80%），但 MBPP Python 的**误报测试执行率高达 16.3%**，而 HumanEval Python 仅为 1.4%。误报率高意味着单元测试通过但实际上解决方案错误，导致 Agent 过早报告无效提交。

在 Reflexion 的实现中，**假阴性优于假阳性**——Agent 可能通过自我反思识别不正确的测试并提示自己保持原始代码不变；但如果无效测试套件返回假阳性（所有内部测试通过但实现错误），Agent 将过早报告无效提交。

#### 消融实验

![表3: HumanEval Rust 消融实验](img/table3_ablation.png)

**表3**在 HumanEval Rust 最难的 50 个问题上测试了 Reflexion 的复合方法：

| 方法 | 测试生成 | 自我反思 | Pass@1 (Acc) |
|------|---------|---------|-------------|
| 基线模型 | False | False | 0.60 |
| 省略测试生成 | False | True | **0.52** |
| 省略自我反思 | True | False | 0.60 |
| **Reflexion** | **True** | **True** | **0.68** |

关键发现：
1. **省略测试生成**（仅保留自我反思）导致准确率从 60% 降至 52%——表明 Agent 无法在没有单元测试的情况下确定当前实现是否正确，必须在所有迭代中持续参与，对实现进行有害的编辑；
2. **省略自我反思**（仅保留测试生成）准确率与基线持平（60%）——测试生成和代码编译能够捕获语法和逻辑错误，但实现修复未能反映这些指示；
3. **完整 Reflexion** 达到 68%——测试生成与自我反思的协同至关重要。

这些实证结果表明，**没有自我反思的盲目试错调试技术在编写 Rust 等复杂程序等更困难任务上是无效的**。

---

### 4.4 与先前工作的对比

![相关工作对比表](img/related_work_tables.png)

上表对比了 Reflexion 与推理/决策和编程领域的相关工作：

**推理与决策领域**:
- Self-refine [15]: 支持自我优化但不支持隐藏约束、决策、二元奖励和记忆；
- Beam search [27]: 支持自我优化、隐藏约束、决策和二元奖励，但不支持记忆；
- **Reflexion**: 同时支持自我优化、隐藏约束、决策、二元奖励和记忆——是唯一全覆盖的方法。

**编程领域**:
- AlphaCode [14]: 支持测试执行和调试，但不支持自我生成测试、多语言和自反思；
- CodeT [5]: 支持测试执行和自我生成测试，但不支持调试、多语言和自反思；
- Self-debugging [7]: 支持测试执行和调试，但不支持自我生成测试、多语言和自反思；
- CodeRL [12]: 支持测试执行和调试，但不支持自我生成测试、多语言和自反思；
- **Reflexion**: 唯一同时支持测试执行、调试、自我生成测试、多语言和自反思的方法。

---

## 五、局限性与未来方向

### 5.1 主要局限

1. **局部最优解**: Reflexion 本质上是使用自然语言进行策略优化的技术，仍可能陷入非最优的局部极小解；
2. **内存结构限制**: 当前长期记忆采用最大容量的滑动窗口，未来可扩展为向量嵌入数据库或传统 SQL 数据库等更先进的结构；
3. **测试驱动开发的约束**: 对于代码生成任务，存在许多实际限制：
   - 非确定性生成器函数；
   - 与 API 交互的纯函数（impure functions）；
   - 根据硬件规格变化输出的函数；
   - 调用并行或并发行为难以预测的函数。
4. **对 LLM 自评估能力的依赖**: Reflexion 的效果受限于 LLM 的自我评估能力（或启发式函数的质量），且没有形式化的成功保证。

### 5.2 在 WebShop 上的失败案例

![图6: WebShop 实验结果](img/fig6_webshop_results.png)

**图6**展示了 Reflexion 在 WebShop [29] 上的实验结果。WebShop 是一个基于 Web 的问题解决基准，测试 Agent 在电子商务网站上导航以定位和购买产品的能力。

实验在 100 个环境中测试了两轮 ReAct + Reflexion Agent。然而，仅经过 4 次试验后实验即终止——Agent 未显示出改善迹象，且在失败后未能生成有用的直观自我反思。

**失败原因分析**: WebShop 要求极高多样性和探索性的行为，而 Reflexion 难以应对。在 ALFWorld 中，Agent 能够充分探索新环境，因为可允许的动作可以在观测中看到；在 HotPotQA 中，Agent 面对类似的搜索查询任务但更为成功，因为 Wikipedia 文章的搜索空间更加多样且需要不那么精确的搜索查询。电子商务搜索引擎面临的共同问题是自然语言搜索解释中的歧义处理——WebShop 要求 Reflexion Agent 表现出非常多样化和独特的行为。

### 5.3 模型能力的影响

附录中的额外实验（**表4**和**表5**）表明，自我纠正的能力是**更强、更大模型的涌现特性**。

![表4: Starchat-beta 模型上的结果](img/table4_starchat.png)

在 HumanEval Python 上使用 starchat-beta [13] 模型时，Reflexion 与基线的 Pass@1 准确率相同（0.26），标准差分别为 0.00305 和 0.00481——较弱模型无法从自我反思中获益。

![表5: 不同模型在 HotPotQA 上的结果](img/table5_various_models.png)

**表5**比较了不同模型在 100 个 HotPotQA 问题上的表现：

| 模型 | 基线准确率 | Reflexion 准确率 | 绝对提升 |
|------|----------|----------------|---------|
| CoT(GT) + text-davinci-003 | 0.60 | 0.77 | +0.17 |
| CoT(GT) + gpt-3.5-turbo | 0.57 | 0.71 | +0.14 |
| CoT(GT) + gpt-4 | 0.68 | **0.80** | **+0.12** |
| ReAct + text-davinci-003 | 0.30 | 0.55 | +0.25 |
| ReAct + gpt-3.5-turbo | 0.26 | 0.38 | +0.12 |
| ReAct + gpt-4 | 0.39 | **0.51** | **+0.12** |

规律：更强的模型（GPT-4）基线更高，Reflexion 带来的绝对提升可能略小，但最终准确率更高。

---

## 六、结论

### 6.1 核心贡献总结

Reflexion 提出了一种全新的"语言强化学习"范式，其核心贡献可概括为四点：

1. **新范式**: 提出 Reflexion，一种"语言"强化学习范式，将策略参数化为 Agent 的记忆编码与 LLM 参数选择的组合；
2. **自我反思的涌现特性**: 探索并实证证明 LLM 中的自我反思是一种极其有用的涌现能力，可在少量试验中学习复杂任务；
3. **新基准**: 引入 LeetcodeHardGym，一个包含 40 道 LeetCode Hard 难度编程题的代码生成 RL gym 环境，支持 19 种编程语言；
4. **全面超越基线**: 在多个任务上取得超越强基线的改进，并在各类代码生成基准上达到最先进水平。

### 6.2 关键定量结果

| 任务 | 基准 | 改进幅度 |
|------|------|---------|
| 决策 | ALFWorld | **+22%** (12 次迭代学习) |
| 推理 | HotPotQA | **+20%** |
| 编程 | HumanEval | **+11%** (Pass@1 从 80% 提升至 91%) |

### 6.3 更广泛的影响

从正面看，Reflexion 可能解决传统 RL 中一些长期存在的问题：
- **可解释性**: 传统 RL 的黑盒策略和优化设置使可解释性和对齐变得困难，而"语言"强化学习使自主 Agent 更加可解释和可诊断；
- **安全性监控**: 在工具使用场景中（可能过于复杂难以理解），可以监控自我反思以确保在使用工具之前的意图正确。

从负面看：
- 大型语言模型越来越多地用于与外部环境（互联网、软件、机器人等）和人类交互；
- 这项工作有强化和赋能这些 Agent 实现更大自动化和工作效率的潜力，但也会放大这些 Agent 被误用的风险；
- 这一研究方向需要更多安全和伦理方面的考量。

### 6.4 未来方向

1. **值学习**: 将 Reflexion 用于传统 RL 中已深入研究的更先进技术，如自然语言中的值学习（value learning in natural language）；
2. **离策略探索**: 引入离策略探索技术（off-policy exploration techniques）；
3. **高级记忆结构**: 使用向量嵌入数据库或传统 SQL 数据库存储和检索反思经验；
4. **更复杂的反馈信号**: 探索结构化的、层次化的反馈形式。

---

## 七、关键公式汇总与详解

本节对论文中出现的核心符号与公式进行统一梳理，确保符号完备性。

### 7.1 策略定义

$$\pi_\theta(a_i \mid s_i), \quad \theta = \\{M_a, \text{mem}\\}$$

| 符号 | 含义 | 空间/类型 |
|------|------|----------|
| $\pi_\theta$ | 参数化策略 | 映射 $\mathcal{S} \times \mathcal{H} \to \mathcal{P}(\mathcal{A})$ |
| $a_i \in \mathcal{A}$ | 第 $i$ 步的动作 | 动作空间（文本/API 调用/决策） |
| $s_i \in \mathcal{S}$ | 第 $i$ 步的状态观测 | 状态空间（环境返回的文本观测） |
| $\theta$ | 策略参数 | 元组 $\\{M_a, \text{mem}\\}$ |
| $M_a$ | Actor 模型（LLM） | 预训练语言模型 |
| $\text{mem}$ | 长期记忆（反思经验） | 文本序列列表 $[sr_0, sr_1, \ldots]$ |

### 7.2 轨迹定义

$$\tau_t = [a_0, o_0, a_1, o_1, \ldots, a_i, o_i]$$

| 符号 | 含义 | 空间/类型 |
|------|------|----------|
| $\tau_t$ | 第 $t$ 次试验的轨迹 | 动作-观测交替序列 |
| $a_k \in \mathcal{A}$ | 第 $k$ 步的动作 | 文本/命令 |
| $o_k \in \mathcal{O}$ | 第 $k$ 步的环境观测 | 文本反馈 |
| $i \in \mathbb{N}$ | 轨迹长度（步数） | 正整数 |
| $t \in \mathbb{N}_0$ | 试验轮次索引 | 非负整数 |

### 7.3 奖励函数

$$r_t = M_e(\tau_t)$$

| 符号 | 含义 | 空间/类型 |
|------|------|----------|
| $r_t \in \mathbb{R}$ | 第 $t$ 次试验的标量奖励 | 实数（实践中通常为 $\\{0, 1\\}$） |
| $M_e$ | Evaluator 模型 | 评分模型（LLM 或启发式函数） |
| $\tau_t$ | 第 $t$ 次试验的轨迹 | 见 7.2 |

### 7.4 自我反思生成

$$sr_t = M_{sr}(\tau_t, r_t, \text{mem})$$

| 符号 | 含义 | 空间/类型 |
|------|------|----------|
| $sr_t$ | 第 $t$ 次试验的自我反思文本 | 自由格式自然语言文本 |
| $M_{sr}$ | Self-Reflection 模型（LLM） | 预训练语言模型 |
| $\tau_t$ | 当前试验轨迹 | 见 7.2 |
| $r_t$ | 当前试验奖励 | 见 7.3 |
| $\text{mem}$ | 历史反思记忆 | $[sr_0, sr_1, \ldots, sr_{t-1}]$ |

### 7.5 记忆更新

$$\text{mem} \leftarrow \text{mem} \oplus [sr_t], \quad |\text{mem}| \leq \Omega$$

| 符号 | 含义 | 空间/类型 |
|------|------|----------|
| $\oplus$ | 序列追加操作 | 列表拼接 |
| $\Omega \in \\{1, 2, 3\\}$ | 最大记忆容量 | 正整数 |
| $|\text{mem}|$ | 当前记忆中的经验数量 | 非负整数 |

### 7.6 测试套件（编程任务）

$$T = \\{t_0, t_1, \ldots, t_n\\}, \quad n \leq 6$$

| 符号 | 含义 | 空间/类型 |
|------|------|----------|
| $T$ | 单元测试套件 | 测试函数集合 |
| $t_k$ | 第 $k$ 个单元测试 | 可执行测试函数 |
| $n$ | 测试套件中的测试数量 | 整数，$0 \leq n \leq 6$ |

---

## 八、术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| 大语言模型 | Large Language Model (LLM) | 基于 Transformer 架构、参数量巨大（通常 > 1B）的预训练语言模型 |
| 上下文学习 | In-context Learning | LLM 通过提示中的示例学习新任务，无需更新模型参数 |
| 信用分配问题 | Credit Assignment Problem | 在序列决策中确定哪个动作对最终结果的贡献（正面或负面）的困难 |
| 语义梯度 | Semantic Gradient | Reflexion 中将语言反馈类比为传统 RL 中梯度信号的概念 |
| 情景记忆 | Episodic Memory | 对特定事件/经历的记忆，在 Reflexion 中指存储的自我反思经验 |
| Pass@1 | Pass at 1 | 编程任务评估指标：单次生成通过所有测试的概率 |
| 精确匹配 | Exact Match (EM) | 评估指标：生成输出与参考答案的字符串完全匹配 |
| 策略迭代 | Policy Iteration | RL 中交替执行策略评估和策略改进步骤的算法 |
| 抽象语法树 | Abstract Syntax Tree (AST) | 源代码的树形表示，用于语法分析 |
| 假阳性/假阴性 | False Positive / False Negative | 二元分类中的两种错误类型：FP 为错误预测为正，FN 为错误预测为负 |

---

> **免责声明**: 本笔记严格基于论文原文提炼，所有实验数据、图表与结论均来自 Shinn 等人的原始论文 [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)。笔记中的公式符号保持与原文一致，概念详解中的扩展内容已明确标注为推导性分析。
