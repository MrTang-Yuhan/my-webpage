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

## 内存建模细节

内存建模为多个 GPU 本地内存（显存） + 远程内存。

程序根据执行时用户制定的 GPU 数量参数，为每个 GPU 分配本地内存，远程内存只有一个。大致架构如下图。 

### GPU 本地内存

![](img/local_memory.png)

- GPU 本地内存是容量有限的。
- GPU 的端口容量是有限的。


### 远端内存

- 远端内存是容量无限的。
- 远端内存的端口容量是有限的。
- 

![](img/remote_memory.png)






