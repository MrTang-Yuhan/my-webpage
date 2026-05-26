---
layout: post.njk
post_id: 2026-05-26-triosim-模拟器-四-内存和网络模型
archive: triosim模拟器
title: TrioSim 模拟器 （四）：内存和网络模型
date: 2026-05-26
tags:
  - post
---
# 内存模型

以 `main.cpp` 中 `PlayTrace()` 函数为例：

```c++
void PlayTrace(triosim::Trace& trace,
               akita::sim::SerialEngine* engine,
               triosim::TimeEstimator* timeEstimator) {

...
    // 构建硬件平台：GPU MemoryRegions + Remote Memory + 网络
    auto [remoteMemRegion, remotePort, gpuPorts] = BuildHardwarePlatform(tracePlayer, engine);
        
    // 设置默认内存区域为 Remote（GPU 本地内存不足时使用）
    tracePlayer->SetDefaultMemoryRegion(remoteMemRegion);  
    // 将 tensors 加入 default_memory_region_
    tracePlayer->SetTrace(trace, config.batch_size_sim);
...
}
```

这段代码同时完成了内存模型的建模。

以下是润色后的版本，表述更加自然流畅：

---

## 内存建模细节

内存模型由多个 GPU 本地内存（显存）和一份远程内存组成。

程序根据运行时用户指定的 GPU 数量，为每个 GPU 分配独立的本地内存；远程内存则全局共享一份。整体架构如下图所示。

### GPU 本地内存

![](img/local_memory.png)

- 容量有限
- 端口带宽有限
- 初始状态为空，不含任何张量数据（trace）

### 远程内存

![](img/remote_memory.png)

- 容量视为无限
- 端口带宽有限
- 初始张量数据（trace）全部存放于此

---







