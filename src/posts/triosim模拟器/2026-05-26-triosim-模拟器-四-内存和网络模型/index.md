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

    // 创建 InferenceTracePlayer，用于回放 operators
    auto* tracePlayer = new traceplayer::InferenceTracePlayer(
        "Player", engine, engine, timeEstimator);           // 注意传入的 ITimeTeller 和 IEventScheduler 参数均是 SerialEngine

    // 构建硬件平台：GPU MemoryRegions + Remote Memory + 网络
    auto [remoteMemRegion, remotePort, gpuPorts] = BuildHardwarePlatform(tracePlayer, engine);
        
    // 设置默认内存区域为 Remote（GPU 本地内存不足时使用）
    tracePlayer->SetDefaultMemoryRegion(remoteMemRegion);  
    // 将 tensors 加入 default_memory_region_
    tracePlayer->SetTrace(trace, config.batch_size_sim);

    // 根据 interconnect 类型选择网络模型
    if (config.interconnects == 1) {
        // 光学网络：使用 OpticalNetworkModel
        SetupOpticalNetwork(engine, remotePort, gpuPorts);
    } else {
        // 电气网络：使用 PacketSwitchingNetworkModel
        // 等效总线带宽计算：PTP_bw * 2 * (N-1) / N, 公式推导如下：
        // N 个 GPU 进行 Ring AllReduce，拆成 Reduce-Scatter + All-Gather 2个阶段。
        // 每个阶段进行 N - 1 轮通信，每轮发送 S / N 个数据 (每个 GPU 上有大小为 S 的
        // 数据, 被切分成 N 块，故每块大小是 S / N) 
        double busbandwidth = config.ptp_bandwidth * 2 * (config.gpu_number - 1.0) / config.gpu_number;
        auto* networkModel = SetupPacketSwitchingNetwork(engine, remotePort, gpuPorts, busbandwidth);
        tracePlayer->SetNetworkModel(networkModel);
    }

    // 启动 trace 回放（发送初始事件）
    tracePlayer->KickStart();

    // 运行事件驱动引擎，直到所有事件处理完毕
    auto err = engine->Run();
    if (err) {
        printf("Simulation error code: %d\n", err);
    }
    printf("Estimated execution time ms, %.10f\n", engine->CurrentTime() * 1000);
}
```

