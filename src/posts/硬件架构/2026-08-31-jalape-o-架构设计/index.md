---
layout: post.njk
post_id: 2026-08-31-jalape-o-架构设计
archive: 硬件架构
title: Jalapeño 架构设计
date: 2026-08-31
tags:
  - post
---
## 附录：参考资源

### 第三方分析

[重新设计推理芯片： 从 Nvidia GPU 的缺陷到 OpenAI Jalapeño](https://mp.weixin.qq.com/s/wTW2KyhDBMmWP1tkwNQt7A?scene=1)

[Jalapeño 拆解：OpenAI 的第一代推理 ASIC，Prefill-Decode 分离不再必要了？](https://mp.weixin.qq.com/s/DnvsiPBSyb6BTrjZHQmqoA?scene=1&click_id=1826631586)


## 一、背景：从带宽墙到开销墙（The Overhead Wall）

现代 AI 推理的瓶颈已经发生迁移。在训练或大 batch 场景下，瓶颈是**算力**（FLOPS）或**内存带宽**（GB/s）；但在服务化推理（serving）场景——尤其是低延迟、小 batch、短序列、稀疏 MoE 模型下——真正的瓶颈是**固定开销**：

- Kernel launch 与提交开销
- 全局 barrier 与同步开销
- 深缓存层级的寻址与仲裁开销
- 数据跨芯片/跨层级的搬运与排队开销

Jalapeño 的全部设计动作，本质上都是在**用体系结构手段消除这些固定开销**，而不是简单地堆砌更多算力或带宽。

---

## 二、四个关键设计动作的深层原因

### 设计动作 1：Core Slice 与 HBM Slice 一一配对
**问题根因：统一内存体系的争用与不可预测性**

GPU 采用 Temporal 执行模型，数千线程共享统一寻址空间。为了掩盖 DRAM 延迟，GPU 必须依赖深层缓存（L0/L1/L2）、大容量 shared memory 和大量 warp 切换。这导致：
- **集中式仲裁瓶颈**：多个 SM 同时访问 HBM 时，需经过 crossbar、arbiter、全局 memory fence，延迟随并发数非线性增长。
- **shape 依赖的效率**：GPU 必须凑够大 tile（如 128×128）才能让单个 SM 执行足够长时间，以摊销内存往返和 kernel launch 开销。小 shape（如 batch=1）效率暴跌。
- **缓存一致性开销**：统一视图需要硬件维护 coherency，增加了面积、功耗和延迟。

**Jalapeño 解法：Spatial Partitioning**
将 64 个 core slice 与 64 个 HBM slice **严格一一绑定**，形成 64 个独立的本地内存域。每个 core slice 对自己的 HBM slice 拥有**专属的物理通路**。



---

### 设计动作 2：双网络分离（Collective Network + General NoC）
**问题根因：混合流量导致的资源竞争与 QoS 失效**

在 GPU 或统一 NoC 的加速器中，所有通信（all-reduce、all-to-all、host 控制、debug）走同一套网络。这导致：
- **关键路径被污染**：TP（Tensor Parallel）的同步 all-reduce 是延迟敏感且在关键路径上的，但它会被 EP（Expert Parallel）的突发 all-to-all 或 host 控制包挤占。
- **过度设计**：通用 NoC 必须为最坏情况（不规则路由、拥塞）预留缓冲和仲裁资源，增加了面积和功耗。

**Jalapeño 解法：Traffic Isolation by Construction**
- **Collective Network**：物理上为规则集合通信（all-reduce、all-gather、reduce-scatter）优化。由于通信图在编译期完全已知（ring/tree），网络可做**电路交换式预配置**，无需逐包路由。
- **General NoC**：处理不规则、非预测流量（host 控制、debug、动态 EP 路由）。带宽低、延迟高，但灵活。

---

### 设计动作 3：Weight-Stationary Systolic Array 支持小维度
**问题根因：Shape Mismatch 导致的利用率悬崖**

Weight-Stationary（WS）脉动阵列的优势在于权重只从 HBM 读取一次，之后完全在阵列内部复用，极大降低内存带宽压力。但传统 WS 阵列有一个致命弱点：
- **刚性尺寸**：如果 matmul 维度（M, N, K）不是阵列物理尺寸（如 128×128）的整数倍，边缘 PE 大量空闲。例如 130×130 的 matmul 映射到 128×128 阵列，利用率骤降至约 60%。

**Jalapeño 解法：细粒度可重构阵列**
通过位切片（bit-slicing）或子阵列动态组合，支持更小维度的 WS 执行。阵列不再是刚性的一块，而是可以**根据 workload shape 动态重构**。


---

### 设计动作 4：OoO 标量核 + L1 Cache
**问题根因：软件管理内存的控制开销与动态流处理困难**

传统加速器（TPU、Groq）采用 software-managed scratchpad + 异步 DMA：
- 编译器必须显式指定每个数据块的搬运时机和地址。
- 遇到不规则控制流（如 MoE 动态路由、变长序列）时，静态调度几乎不可能。
- Barrier 同步是硬 stall：一个 core 慢了，所有 core 等。

**Jalapeño 解法：把"控制"交给硬件，把"映射"交给 AI**
- **OoO（Out-of-Order）标量核**：跑在 matrix engine 前面。它自动重排指令，遇到内存未命中时不 stall 整个流水线，而是先执行后续无关指令，同时预取数据。matrix engine 就绪时，数据已在队列中等待。
- **L1 Cache**：自动管理数据局部性。编译器/AI 不需要知道每个 byte 的物理地址，正常读写即可，硬件自动缓存常用数据。



---

## 三、PD 分离（Prefill-Decode Disaggregation）详解

### 3.1 什么是 PD 分离？
大模型推理分为两个阶段：

| 阶段 | 计算特征 | 资源瓶颈 | 优化目标 |
|------|---------|---------|---------|
| **Prefill** | 处理输入上下文，Attention 计算量 ∝ seq_len² | **算力**（Compute-bound） | TTFT（首字延迟） |
| **Decode** | 自回归生成，每步 1 个 token，计算量 ∝ seq_len | **内存带宽**（Memory-bound） | TPOT/TBT（逐字延迟） |

**PD 分离**：将两个阶段拆分到**不同的物理设备池**执行，中间通过跨网络传输 KV cache。

### 3.2 为什么 GPU 必须做 PD 分离？
GPU 缺乏细粒度的 request-level 抢占与隔离机制：
- **Stream Priority**：无法任意切片正在运行的大 kernel。
- **Compute Preemption**：有状态保存/恢复开销。
- **MIG（Multi-Instance GPU）**：静态分区，无法动态响应负载变化。

因此，GPU 上 prefill 和 decode 共置时会产生**双向干扰**：
1. 大 batch prefill 突发占用 GPU → decode 请求的 TPOT 劣化。
2. 长时间 decode 占用 GPU → 新请求 TTFT 爆炸。

PD 分离是**在 GPU 调度粒度限制下的有效妥协**：用物理隔离换取可预测的 SLO（Service Level Objective）。

### 3.3 PD 分离的代价
- **KV 搬运开销**：长上下文下 KV cache 可达几十 GB，跨芯片搬运消耗大量带宽和延迟。
- **池失衡（Pool Skew）**：Prefill 需求激增时，Prefill 池爆满而 Decode 池空闲；反之亦然。需持续预测比例并预留冗余。
- **资源碎片**：两侧都必须预留 headroom（余量），全局利用率下降。

---

## 四、为什么 Jalapeño 可以不做 PD 分离：根因分析

### 根因 1：巨大的 SLO Headroom（最关键）
Jalapeño 的 min TBT（最小 token 间隔）远低于 GPU，导致在相同 TPOT SLO 下，能容忍更大的 prefill 干扰：

| TPOT SLO | Jalapeño 可容忍干扰 | GB200 可容忍干扰 | Jalapeño 优势 |
|---------|-------------------|----------------|--------------|
| 10 ms | 93.1% | 81.3% | 1.15× |
| 5 ms | 86.2% | 62.6% | 1.38× |
| **3 ms** | **77.0%** | **37.7%** | **2.04×** |

**计算逻辑**：可容忍干扰 = `(SLO - min_TBT) / SLO`
- Jalapeño min TBT = **0.69 ms**（GPT-OSS）
- GB200 min TBT = **1.87 ms**

**专业解读**：在 3 ms SLO 下，Jalapeño 允许 prefill 将当前 step 延迟 2.31 ms 而不超标；GB200 只允许延迟 1.13 ms。当 headroom 大于实际干扰量时，PD 分离的隔离收益趋近于零，而其成本（KV 搬运、池失衡、碎片）却恒定存在。**净收益为负。**

### 根因 2：局部性消除了 KV 搬运的动机
PD 分离必然打破 locality，导致 KV cache 跨网络搬运。Jalapeño 从三层消除这一需求：
- **Slice 级局部性**：KV cache 生成于本地 HBM slice，decode 时同一 core slice 直接复用。
- **显式放置（TensorInfo）**：编译器显式编码 tensor 的物理布局，KV 位置编译期确定，无需运行时动态路由。
- **同构可替换池（Fungible Fleet）**：任意芯片可执行任意阶段，请求可在同一芯片完成 prefill→decode，无需"跨池搬家"。

> 官方原话：`Locality is king: keep KV local`

### 根因 3：可预测编程模型支持细粒度交错
- **Persistent Kernel / Gigakernel**：常驻设备端，无需反复 launch，消除了 kernel 提交开销。
- **极简内存层级 + 可预测同步**：使得 chunked prefill（将长 prefill 切小块）与 decode step 的细粒度交错调度开销极低。
- 编译器可以精确建模每一步的延迟，无需像 GPU 那样靠大 batch 来"摊销调度不确定性"。

### 根因 4：投机解码（MTP）与分离的结构性冲突
OpenAI 使用 MTP=7（7 轮 draft + 1 次 verify）。这是一个**极度延迟敏感的紧耦合循环**：
- Draft 模型必须极快地产出候选 token；
- Verifier 必须立即批量验证；
- 两者往返延迟直接决定加速比。

若将 draft 和 verifier 拆到不同池，等于把这个 tight loop 变成**分布式事务**——网络往返、序列化、同步的 overhead 可能直接吃掉投机解码的收益。

