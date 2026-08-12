---
layout: post.njk
post_id: 2026-08-12-apogee-adaptive-prefetching-on-gpus-for-energy-efficiency
archive: GPU预取技术
title: "APOGEE: Adaptive Prefetching On GPUs for Energy Efficiency"
date: 2026-08-12
tags:
  - post
---
- 论文链接：[APOGEE: Adaptive Prefetching On GPUs for Energy Efficiency](https://cccp.eecs.umich.edu/papers/asethia-pact13.pdf)

# APOGEE 核心方法

## FOA 预取器：

> **一个 warp 里的线程同时暴露地址规律，预取器瞬间学会"步长"，下次按公式批量预取。**

---

### 学习（Training）

**场景：** 8 线程的 warp 执行 `load A[tid]`

| 线程 | 索引 | 地址 |
|------|------|------|
| tid0 | 0 | `0x1000` |
| tid1 | 1 | `0x1004` |
| tid2 | 2 | `0x1008` |
| tid3 | 3 | `0x100C` |

**硬件并行计算：**
- `(0x1004 - 0x1000) / (1-0) = 4`
- `(0x1008 - 0x1004) / (2-1) = 4`
- `(0x100C - 0x1008) / (3-2) = 4`

**判定：** 所有相邻对算出的 offset 都是 4 → **学会 FOA，offset = 4**

写入预取表：
```
PC=0x253ad, Address=0x1000, Offset=4, Confidence=高, Distance=1
```

---

### 预取（Prefetching）

**公式：** 下次地址 = 上次地址 + offset × 总线程数

- 上次 tid0 地址：`0x1000`
- 总线程数 = 8
- 下次 tid0 地址 = `0x1000 + 4×8 = 0x1020`

**整个 warp 下次要访问的地址：**
- tid0: `0x1020`, tid1: `0x1024`, tid2: `0x1028` ... tid7: `0x103C`

合并同一缓存行后，发 **2 个预取请求**到内存。

---

### 动态距离调整（及时性校正）

预取表每项带 2-bit 状态机：

| 状态 | 含义 |
|------|------|
| `00` | 刚执行这条 load |
| `01` | 预取已发，数据还没回来 |
| `10` | 数据已回缓存 |

**太慢：** 状态 `01` 时，这条 load 又执行了（缓存未命中）→ **Distance + 1**（提前更多）

**太早：** 状态 `10` 时，这条 load 执行但缓存未命中（数据被挤掉了）→ **Distance - 1**（提前少一点）

**刚好：** 状态 `10` 时，缓存命中 → **Distance 不变**

---

## TIA 预取器：

> **所有线程读同一个地址，预取器不需要算地址，只需要找一条"够早"的 load 当哨兵，提前把数据喊回来。**

---

### 场景

循环体里：
```
PC0: load &LightColor      ← 所有线程读地址 0xFACE（TIA）
PC1: load &Normal[tid]     ← FOA
PC2: load &Position[tid]   ← FOA
PC3: load &TexCoord[tid]  ← FOA
...几百条指令...
回到 PC0
```

**问题：** PC0 第一次执行后，LightColor 在缓存里。但中间几百条指令把缓存挤爆，回到 PC0 时 0xFACE 已被逐出 → **又要等 400 周期去内存读。**

---

### 学习

Warp 执行 PC0，预取器看到 8 个线程的地址全是 `0xFACE`。

offset = 0，所有线程一致 → **判定为 TIA。**

预取表写入：
```
PF PC=PC0, Address=0xFACE, Offset=0, Load PC=PC1, Slow Bit=0
```
（Load PC 设为 PC0 之后**最近执行过的 load**，即 PC1）

---

### 预取与 LAL 链调整

**第一轮：**
- 执行到 PC1 时，预取器发现"我是 PC0 的哨兵"
- 发预取：`prefetch 0xFACE`
- 但 PC1 离 PC0 太近（只隔 50 周期），数据没回来 → **PC0 未命中**

**动作：** Slow Bit = 1，Load PC 前移一位 → **PC2**

**第二轮：**
- 执行到 PC2 时发预取
- PC2 离 PC0 有 150 周期，还是不够 → **PC0 未命中**

**动作：** Load PC 再前移 → **PC3**

**第三轮：**
- 执行到 PC3 时发预取
- PC3 离 PC0 有 500 周期 > 400 周期内存延迟
- 回到 PC0 时，数据已在缓存 → **命中！**

**动作：** Load PC 固定为 PC3。以后每次执行 PC3，自动为 PC0 预取 `0xFACE`。

---

## 核心差异对比

| | FOA | TIA |
|--|-----|-----|
| **地址** | 每个线程不同，要算 | 所有线程相同，已知 |
| **难点** | 算对下次地址 | 找够早的触发时机 |
| **学习** | 算相邻线程 offset | 发现 offset = 0 |
| **预取** | 按公式批量算地址 | 用 LAL 链找哨兵提前触发 |
| **调优** | Distance 增减（状态机） | Load PC 前移（Slow Bit） |
