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

---

# 详解 TracePlayer

仍以 case 0 为例，函数 `auto* tracePlayer = new traceplayer::InferenceTracePlayer("Player", engine, engine, timeEstimator)` 的定义如下：

```c++
InferenceTracePlayer::InferenceTracePlayer(
    const std::string& name,
    akita::sim::ITimeTeller* tt,
    akita::sim::IEventScheduler* es,
    TimeEstimator* time_estimator)
    : akita::sim::ComponentBase(name)
    , time_teller_(tt)
    , event_scheduler_(es)
    , time_estimator_(time_estimator)
    , batch_size_(0)
    , reduce_layer_(0)
    , send_to_finish_(0)
    , scatter_step_(0)
    , gather_step_(0)
    , network_model_(nullptr) {
}
```

实际传入的参数中：
- `engine` 的类型为 `class SerialEngine : public Engine`，负责实现串行模拟。它分别被传递给形参 `akita::sim::ITimeTeller* tt` 和 `akita::sim::IEventScheduler* es`。<br>
由于 `class SerialEngine : public Engine`， `class Engine : public Hookable, public TimeTeller, public EventScheduler, public ITimeTeller, public IEventScheduler`，因此 `class SerialEngine` 实际上是 `class ITimeTeller` 和 `class IEventScheduler` 的派生类。
- `timeEstimator` 的类型为 `class RecordedTimeEstimator : public TimeEstimator`，被传递给形参 `TimeEstimator* time_estimator`。其作用是记录 trace 中每一层的执行时间，并在需要时直接读取该时间。

关键点在于实参 `engine`：

- **`class ITimeTeller` 的功能在 `class SerialEngine` 中被重载为直接报告当前的模拟器全局时间（注意：此处为模拟时间，而非墙钟时间）。**
- **`class IEventScheduler` 的功能在 `class SerialEngine` 中被重载为将事件放入调度队列。**

---

# 如何实现事件的调度和执行

接上文，**`class IEventScheduler` 的功能在 `class SerialEngine` 中被重载为将事件放入调度队列。**

在代码中，经常能看到类似下面的写法：

```c++
ScheduleEvent(MakePlayNextEvent(CurrentTime(), this, item->gpu_id));
```

涉及的相关函数定义如下：

```c++
...
PlayNextEvent* InferenceTracePlayer::MakePlayNextEvent(akita::sim::VTimeInSec time, InferenceTracePlayer* handler, int gpu_id) {
    return new PlayNextEvent(time, handler, gpu_id);
}
...
void ScheduleEvent(akita::sim::Event* evt) {
    event_scheduler_->Schedule(evt);
}
...
```

其中：
- `MakePlayNextEvent()` 的作用是创建一个 `PlayNextEvent` 类型的事件并返回；
- `ScheduleEvent()` 的作用是通过事件调度器将事件放入调度队列。代码中所有用到的事件调度器实际上都是 `class SerialEngine` 的实例。

后续流程中，调度器会不断从队列中取出最早的事件，并调用其 `Handle()` 函数进行处理，例如：

```c++
int InferenceTracePlayer::Handle(akita::sim::Event* e) {
    if (auto* evt = dynamic_cast<PlayNextEvent*>(e)) {
        PlayNext(evt->gpu_id_);
    } else if (auto* evt = dynamic_cast<LayerCompletionEvent*>(e)) {
        CompleteLayer(evt);
    } else if (auto* evt = dynamic_cast<PlayNextReduceEvent*>(e)) {
        (void)evt;
        PlayNextReduce();
    } else if (auto* evt = dynamic_cast<PlayNextReduceHopEvent*>(e)) {
        PlayNextReduceHop(evt->gpu_id_);
    }
    return 0;
}
```

举例来说，如果从队列中取出的事件是 `PlayNextEvent` 类型，那么 `if (auto* evt = dynamic_cast<PlayNextEvent*>(e))` 条件成立，随后进入 `PlayNext(evt->gpu_id_);` 执行该事件的实际处理逻辑。

---

# Hook 的作用

在 `SerialEngine` 中，`Run()` 每处理一个事件都会触发两次 Hook：

```c++
int SerialEngine::Run() {
    ...
    // 触发 BeforeEvent hook
    HookCtx hook_ctx;
    hook_ctx.domain = this;
    hook_ctx.pos = HookPosBeforeEvent;
    hook_ctx.item = evt;
    hookable_base_.InvokeHook(hook_ctx);

    IHandler* handler = evt->GetHandler();
    int result = handler->Handle(evt);      // 真正处理事件

    // 触发 AfterEvent hook
    hook_ctx.pos = HookPosAfterEvent;
    hookable_base_.InvokeHook(hook_ctx);
    ...
}
```

**Hook 的设计意图是：在不修改主逻辑代码的前提下，插入观测或扩展行为。**

> 注意：目前 TrioSim 模拟器尚未在 `hook_list_` 中注册任何 Hook，因此 `SerialEngine::Run()` 中与 Hook 相关的代码实际上并未生效。

如果想实现一个 Hook，下面是一个简单示例——在 `BeforeEvent` 时打印事件的模拟时间：

```cpp
// simple_time_hook.h
#pragma once
#include "akita/sim/hook.h"
#include "akita/sim/event.h"
#include <iostream>

class SimpleTimeHook : public akita::sim::Hook {
public:
    void Func(akita::sim::HookCtx ctx) override {
        if (ctx.pos != akita::sim::HookPosBeforeEvent) return;

        auto* evt = static_cast<<akita::sim::Event*>(ctx.item);
        if (evt == nullptr) return;

        std::cout << "[Hook] sim_time=" << evt->GetTime() << std::endl;
    }
};
```

然后在 `main.cpp` 中注册：

```c++
auto* engine = new akita::sim::SerialEngine();
auto* timeHook = new SimpleTimeHook();
engine->AcceptHook(timeHook);  // 注册到 hook_list_

auto err = engine->Run();
(void)err;

delete timeHook;  // 引擎不会负责释放
delete engine;
```

---






