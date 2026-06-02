---
layout: post.njk
post_id: 2026-06-02-rlhf-学习路线
archive: RLHF
title: RLHF 学习路线
date: 2026-06-02
tags:
  - post
---
推荐先从 [huggingface 强化学习教程](https://huggingface.co/learn/deep-rl-course/unit0/introduction)入门。

[图解大模型RLHF系列之：人人都能看懂的PPO原理与源码解读](https://zhuanlan.zhihu.com/p/677607581)




----

**分界线，以下内容由 AI 生成**


下面给你一版更“工程 + 科研”两条线并行的 RLHF 学习路线。定位是：你已经懂一些 LLM/SFT/Transformer，但强化学习基础较弱，目标是进入 **AI Infra 里的 RLHF/RLVR/Alignment training 系统**。

检索时间：2026-06-02。

**0. 先建立总图**
RLHF 可以先理解成一个“把人类偏好变成训练信号”的系统：先用 SFT 让模型会回答，再用偏好数据训练 Reward Model，最后用 PPO/DPO/GRPO 等方法让模型更偏向人类喜欢的回答。经典 RLHF 流程可从综述和 InstructGPT 线索入手：  
- RLHF Survey：<https://arxiv.org/abs/2312.14925>[^1]  
- DeepSpeed-Chat 论文，偏工程系统视角：<https://arxiv.org/abs/2308.01320>[^2]  
- RLHF 资源合集：<https://github.com/WeiXiongUST/awesome-RLHF>[^3]

**1. RLHF 是什么时候提出的，作用是什么**
从学术脉络看，RLHF 起源可以追到 preference-based RL，也就是不直接手写 reward，而是让人类比较“哪个输出更好”。2017 年 OpenAI/DeepMind 的人类偏好强化学习工作是关键节点；2022 年 InstructGPT/ChatGPT 把它推成 LLM 对齐主流方案。它的作用不是单纯增加知识，而是让模型在“多个都能说得通的答案”中更倾向有用、真实、安全、符合用户意图的答案。  
推荐链接：  
- RLHF Survey：<https://arxiv.org/abs/2312.14925>[^1]  
- OpenRLHF 论文/框架介绍：<https://arxiv.org/abs/2405.11143>[^4]  
- DeepSpeed-Chat 工程论文：<https://arxiv.org/abs/2308.01320>[^2]

**2. 需要哪些 RL 前置知识**
你不需要一上来学完整 Sutton & Barto。对 LLM RLHF 来说，先掌握这几个概念就够开工：

- Policy：语言模型本身就是策略，给定 prompt/prefix，选择下一个 token。  
  参考 PPO 实现细节：<https://arxiv.org/abs/2403.17031>[^5]

- Reward：Reward Model 对完整 response 打分，工程里还会加 KL penalty，防止模型偏离参考模型太远。  
  参考 DeepSpeed-Chat：<https://arxiv.org/abs/2308.01320>[^2]

- Value/Critic：PPO 里需要估计未来回报，用于降低方差。  
  参考 PPO 复现细节：<https://arxiv.org/abs/2403.17031>[^5]

- KL constraint：RLHF 的核心稳定器，避免 reward hacking。  
  参考 DPO/RLHF 理论理解：<https://arxiv.org/abs/2310.12036>[^6]

- GAE/Advantage：知道它是“估计这一步动作比平均水平好多少”的工具即可，不必先推公式。  
  参考 PPO 实现细节：<https://arxiv.org/abs/2403.17031>[^5]

建议学习顺序：先看 PPO 在 LLM 里的工程实现，再回头补传统 RL；这样比较不容易被 MDP/Bellman 公式劝退。

**3. 从科研学术视角看 RLHF 发展脉络**
可以按算法范式分成 5 个阶段：

1. Preference-based RL / Reward Modeling  
   人类给 pairwise preference，训练 reward model，再用 RL 优化策略。  
   链接：<https://arxiv.org/abs/2312.14925>[^1]

2. PPO-RLHF  
   InstructGPT/ChatGPT 风格主线，典型四模型系统：Actor、Reference、Reward、Critic。  
   链接：<https://arxiv.org/abs/2403.17031>[^5]

3. DPO  
   直接用偏好数据优化模型，绕过显式 Reward Model 和在线 RL loop，工程上更简单。  
   论文链接：<https://arxiv.org/abs/2305.18290>[^7]  
   理论分析链接：<https://arxiv.org/abs/2310.12036>[^6]

4. DPO 后续与理论化  
   研究者开始分析 DPO 与 PPO/RL 的关系、隐式 reward 泛化能力、主动偏好采样等。  
   DeepMind 理论文章：<https://deepmind.google/research/publications/54918/>[^8]  
   Apple 隐式 reward 泛化：<https://machinelearning.apple.com/research/reward-generalization>[^9]  
   Active DPO：<https://arxiv.org/abs/2503.01076>[^10]

5. RLVR / Reasoning RL / GRPO/RLOO  
   面向数学、代码、可验证任务，用规则/verifier 替代人工 reward，减少标注成本。OpenRLHF 已经把 PPO、DPO、KTO、GRPO 等放进统一框架。  
   链接：<https://github.com/OpenRLHF/OpenRLHF>[^11]

**4. 从 AI Infra 工程视角看 RLHF 系统**
RLHF 工程难点不在“写一个 loss”，而在多模型、多阶段、多资源类型之间调度：

- 多角色模型：Actor 生成，Reference 算 KL，Reward 打分，Critic 估 value。  
  参考：<https://arxiv.org/abs/2308.01320>[^2]

- 生成阶段很贵：PPO 训练中大量时间花在 rollout/generation，所以 vLLM、Ray、batching、pipeline overlap 很关键。  
  OpenRLHF README：<https://github.com/OpenRLHF/OpenRLHF/blob/895e8089dc0b1db230316207ca702d5133ae18fd/README_zh.md>[^12]

- 显存压力高：PPO 常常同时涉及 4 个模型，需要 ZeRO、FSDP、offload、LoRA、QLoRA、FlashAttention。  
  DeepSpeed-Chat：<https://arxiv.org/abs/2308.01320>[^2]  
  OpenRLHF：<https://github.com/OpenRLHF/OpenRLHF>[^11]

- 训练与推理交替：RLHF 不是普通 pretrain/SFT，先 rollout，再算 reward/value，再反传更新。系统需要在 inference engine 和 training engine 之间切换。  
  DeepSpeed-Chat 相关介绍：<https://cloud.tencent.com/developer/article/2317877>[^13]

- 数据闭环：prompt 采样、response 生成、人工/模型偏好标注、reward 训练、policy 优化、eval，再回到数据。  
  RLHF Survey：<https://arxiv.org/abs/2312.14925>[^1]

**5. 适合 review 的 RLHF 代码和工程大项目**
建议按“从简单到工业级”顺序看：

1. [TRL (Hugging Face)](https://github.com/huggingface/trl) —— 最适合入门

1. OpenRLHF  
   适合看现代 RLHF 工程主干。支持 Ray、DeepSpeed、vLLM、PPO、DPO、KTO、GRPO、LoRA/QLoRA，工程味很足。  
   链接：<https://github.com/OpenRLHF/OpenRLHF>[^11]  
   中文 README：<https://github.com/OpenRLHF/OpenRLHF/blob/895e8089dc0b1db230316207ca702d5133ae18fd/README_zh.md>[^12]  
   阅读重点：`examples/scripts`、`openrlhf/trainer`、`openrlhf/models`、Ray actor 调度逻辑。

2. DeepSpeed-Chat  
   适合理解早期工业级 PPO-RLHF 系统：SFT、Reward Model、RLHF 三阶段非常清楚。  
   论文：<https://arxiv.org/abs/2308.01320>[^2]  
   介绍：<https://cloud.tencent.com/developer/article/2317877>[^13]  
   阅读重点：RLHF engine、PPO rollout、reward + KL、ZeRO/offload 设计。

3. PPO 复现项目/论文  
   如果你想真正吃透 PPO-RLHF 细节，优先看 2024 这篇复现实验，它列了 20+ 个实现细节，非常适合“科研 + 工程”交叉学习。  
   链接：<https://arxiv.org/abs/2403.17031>[^5]

4. DPO 官方实现  
   DPO 是入门 preference optimization 最友好的路线，比 PPO 少很多系统复杂度。  
   论文：<https://arxiv.org/abs/2305.18290>[^7]  
   代码：<https://github.com/eric-mitchell/direct-preference-optimization>[^7]

5. Awesome RLHF  
   适合长期追论文、代码、benchmark。  
   链接：<https://github.com/WeiXiongUST/awesome-RLHF>[^3]

**6. 推荐学习路线**
第 1 阶段：搞懂 pipeline  
读 RLHF Survey + DeepSpeed-Chat 论文，只要求能画出 SFT → RM → PPO 的数据流。  
链接：<https://arxiv.org/abs/2312.14925>[^1]，<https://arxiv.org/abs/2308.01320>[^2]

第 2 阶段：先学 DPO，再学 PPO  
DPO 更接近普通 supervised fine-tuning，容易建立偏好学习直觉。  
链接：<https://arxiv.org/abs/2305.18290>[^7]

第 3 阶段：精读 PPO-RLHF 实现细节  
重点看 reward shaping、KL penalty、advantage、value loss、rollout batch。  
链接：<https://arxiv.org/abs/2403.17031>[^5]

第 4 阶段：进入 Infra  
看 OpenRLHF/DeepSpeed-Chat 的分布式调度、显存优化、vLLM 生成加速、Ray actor 编排。  
链接：<https://github.com/OpenRLHF/OpenRLHF>[^11]，<https://arxiv.org/abs/2308.01320>[^2]

第 5 阶段：跟前沿  
关注 DPO 理论、隐式 reward、RLVR、multi-turn RLHF、sample-efficient RLHF。  
链接：<https://arxiv.org/abs/2310.12036>[^6]，<https://machinelearning.apple.com/research/reward-generalization>[^9]，<https://arxiv.org/abs/2502.05434>[^14]

**7. 最小知识清单**
学完后你应该能回答这些问题：

- 为什么 RLHF 需要 Reward Model？链接：<https://arxiv.org/abs/2312.14925>[^1]
- 为什么 PPO-RLHF 需要 Reference Model？链接：<https://arxiv.org/abs/2403.17031>[^5]
- 为什么 DPO 不显式训练 Reward Model？链接：<https://arxiv.org/abs/2305.18290>[^7]
- 为什么 RLHF 系统吞吐比 SFT 低很多？链接：<https://arxiv.org/abs/2308.01320>[^2]
- 为什么 OpenRLHF 要用 Ray + vLLM？链接：<https://github.com/OpenRLHF/OpenRLHF/blob/895e8089dc0b1db230316207ca702d5133ae18fd/README_zh.md>[^12]

**参考链接**
[^1]: [A Survey of Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2312.14925)  
[^2]: [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales](https://arxiv.org/abs/2308.01320)  
[^3]: [awesome-RLHF](https://github.com/WeiXiongUST/awesome-RLHF)  
[^4]: [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/abs/2405.11143)  
[^5]: [The N+ Implementation Details of RLHF with PPO](https://arxiv.org/abs/2403.17031)  
[^6]: [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)  
[^7]: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) / [DPO Code](https://github.com/eric-mitchell/direct-preference-optimization)  
[^8]: [Understanding Learning from Human Preferences - Google DeepMind](https://deepmind.google/research/publications/54918/)  
[^9]: [On the Limited Generalization Capability of the Implicit Reward Model Induced by DPO - Apple](https://machinelearning.apple.com/research/reward-generalization)  
[^10]: [Active Learning for Direct Preference Optimization](https://arxiv.org/abs/2503.01076)  
[^11]: [OpenRLHF GitHub](https://github.com/OpenRLHF/OpenRLHF)  
[^12]: [OpenRLHF 中文 README](https://github.com/OpenRLHF/OpenRLHF/blob/895e8089dc0b1db230316207ca702d5133ae18fd/README_zh.md)  
[^13]: [DeepSpeed-Chat 介绍](https://cloud.tencent.com/developer/article/2317877)  
[^14]: [Sample-Efficient RLHF via Information-Directed Sampling](https://arxiv.org/abs/2502.05434)
