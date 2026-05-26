---
layout: post.njk
post_id: 2026-05-26-triosim-模拟器-三-traceplayer
archive: triosim模拟器
title: TrioSim 模拟器 （三）：TracePlayer
date: 2026-05-26
tags:
  - post
---
# TracePlayer 介绍

在 `main.cpp` 的 `int main(int argc, char* argv[])` 函数中，可以看到如下代码：

```c++
int main(int argc, char* argv[])
{
...
    switch (config.case_num) {
        case 0:
            PlayTrace(trace, engine, &timeEstimator);
            break;
        case 1:
            PlayTrace(trace, engine, &timeEstimator);
            PlayTraceWithAllReduce(trace, engine, &timeEstimator);
            break;
        case 2:
            PlayDataTrace(trace, engine, &timeEstimator);
            break;
        case 3:
            PlayTensorTrace(trace, engine, &timeEstimator);
            break;
        case 4:
            PlayPipeTrace(trace, engine, &timeEstimator);
            break;
        case 5:
            PlayTraceWithHop(trace, engine, &timeEstimator);
            break;
        default:
            PlayTrace(trace, engine, &timeEstimator);
            break;
    }
...
}
```

以 case 0 为例，进入函数 `void PlayTrace(triosim::Trace& trace, akila::sim::SerialEngine* engine, triosim::TimeEstimator* timeEstimator)` 后，首先会看到 `auto* tracePlayer = new traceplayer::InferenceTracePlayer("Player", engine, engine, timeEstimator)`，即创建一个 TracePlayer 对象。

可以将 TracePlayer 理解为 **trace 的执行器或解释器**。

在 TrioSim 模拟器中，trace 本身只是静态数据（包含层、张量、时间信息），而 TracePlayer 负责将其转化为动态仿真过程，具体包括：

1. 按顺序逐层推进 layer；
2. 管理每个 GPU 内存区的状态；
3. 触发计算与通信事件；
4. 在 `engine->Run()` 的事件循环中持续调度下一步。

# 详解 TracePlayer



