---
layout: post.njk
post_id: 2026-05-27-triosim-模拟器-五-消息的发送和接收
archive: triosim模拟器
title: TrioSim 模拟器 （五）：消息的发送和接收
date: 2026-05-27
tags:
  - post
---
消息发送/接收不是靠 `InferenceTracePlayer::Handle()` 完成的。`Handle()` 只处理事件，比如 `PlayNextEvent`、`LayerCompletionEvent`、`PlayNextReduceEvent`。

真正的消息路径是：

```text
TracePlayer 创建 TensorMsg
-> src Port 发送
-> NetworkModel 计算传输时间并调度 transferUpdateEvent
-> Engine 到时间后调用 NetworkModel::Handle()
-> NetworkModel 把消息投递给 dst Port
-> dst Port 调用 TracePlayer::NotifyRecv()
-> TracePlayer 把 tensor 落到目标 MemoryRegion
```

> **本文重点讨论电气网络的消息发送和接收机制。在光学网络中，消息的传输过程与此类似。**

---

# 发送消息

发送入口是 inference.cpp：

```cpp
bool InferenceTracePlayer::MsgPkgToSend(...)
```

它会先找到源端口和目标端口：

```cpp
auto* src = GetPortByName(srcRegion);
auto* dst = GetPortByName(dstRegion);
```

然后计算本次消息的数据量：

```cpp
int64_t totalBytes = 0;
for (const auto& tensor : tensors) {
    totalBytes += static_cast<int64_t>(tensor.Bytes());
}
```

接着创建 `TensorMsg`：

```cpp
auto* msg = new TensorMsg();
msg->tensor_pkg = tensors;
msg->dst_region_name = dstRegion;
msg->gpu_id = gpu_id;
msg->purpose = purpose;
msg->meta.id = akita::sim::GetIDGenerator()->Generate();
msg->meta.src = src;
msg->meta.dst = dst;
msg->meta.send_time = CurrentTime();
msg->meta.traffic_bytes = totalBytes;
```

`TensorMsg` 定义在 trace.hpp。

它里面最重要的是：

```cpp
tensor_pkg       // 要传输的张量
src / dst        // 源端口和目标端口
traffic_bytes    // 数据量
purpose          // fetch / scatter / gather / hop
gpu_id           // 目标 GPU 或相关 GPU ID
```

最后发送：

```cpp
auto* sendError = src->Send(msg);
```

这里的 `src` 是 `LimitNumMsgPort`。

---

# Port::Send 做了什么

代码在 port.cpp：

```cpp
SendError* LimitNumMsgPort::Send(Msg* msg) {
    SendError* err = conn_->Send(msg);
    ...
    return err;
}
```

`Port` 自己不计算网络延迟。它只是把消息交给连接对象 `conn_`。

这个 `conn_` 是谁？

在构建网络时设置的：

```cpp
port->SetConnection(this);
```

所以如果当前是电气网络：

```text
conn_ = PacketSwitchingNetworkModel
```

如果当前是光学网络：

```text
conn_ = OpticalNetworkModel
```

---

# 网络模型负责消息传输

以电气网络为例，进入 packetswitching.cpp：

```cpp
PacketSwitchingNetworkModel::Send(Msg* msg)
```

它不会立刻把消息交给目标端口，而是：

```cpp
Route* route = findRoute(msg);
UpdateProgressNextHappenEvent(route);
scheduleNextHappenEvent();
```

含义是：

```text
找路由
计算链路带宽/剩余传输时间
调度一个 transferUpdateEvent
```

也就是说发送是异步的。消息发出后，并不会马上到达，而是等模拟时间推进到传输完成事件。

光学网络类似，在 optical.cpp：

```cpp
OpticalNetworkModel::Send(Msg* msg)
```

它会计算：

```cpp
evt_time = now + latency + transfer_time;
```

然后调度：

```cpp
transferUpdateEvent* evt = new transferUpdateEvent(evt_time, this, msg);
event_scheduler_->Schedule(evt);
```

---

# 传输完成事件如何触发接收

当模拟时间到达 `transferUpdateEvent`，`SerialEngine` 会调用网络模型的 `Handle()`。

电气网络的处理在 packetswitching.cpp：

```cpp
int PacketSwitchingNetworkModel::handleTransferUpdateEvent(...)
```

关键代码：

```cpp
msg->Meta()->recv_time = time_teller_->CurrentTime();
akita::sim::SendError* err = msg->Meta()->dst->Recv(msg);
```

这一步才是真正“消息到达目标端口”。

也就是说：

```text
NetworkModel::Send() 只是发起传输
NetworkModel::handleTransferUpdateEvent() 才把消息投递到 dst Port
```

---

# Port::Recv 做了什么

代码在 port.cpp：

```cpp
SendError* LimitNumMsgPort::Recv(Msg* msg)
```

它会先检查端口 buffer 是否有空间：

```cpp
if (!buf_->CanPush()) {
    port_busy_ = true;
    return NewSendError();
}
```

如果目标端口 buffer 满了，接收失败，网络模型会把消息放进 `pending_delivery_`，以后端口空了再投递。

如果 buffer 有空间：

```cpp
buf_->Push(msg);
```

然后通知端口所属组件：

```cpp
comp_->NotifyRecv(msg->Meta()->recv_time, this);
```

这里的 `comp_` 就是 `InferenceTracePlayer`。

所以接收回调进入：

```cpp
InferenceTracePlayer::NotifyRecv()
```

---

# TracePlayer 接收消息

代码在 inference.cpp：

```cpp
void InferenceTracePlayer::NotifyRecv(...)
```

它先从端口 buffer 取出消息：

```cpp
akita::sim::Msg* msg = port->Retrieve(now);
```

然后转成 `TensorMsg`：

```cpp
auto* tensorMsg = dynamic_cast<TensorMsg*>(msg);
```

接着真正落地张量：

```cpp
RecvTensorPkg(tensorMsg);
```

`RecvTensorPkg()` 在 inference.cpp：

```cpp
RemoveInflightTransfer(msg);
AddTensorsToMemRegion(msg->tensor_pkg, msg->gpu_id, msg->purpose);
```

含义是：

```text
从 inflight_transfer 中移除这条在途消息
把 tensor_pkg 加入目标 GPU 的 MemoryRegion
```

如果是 Ring AllReduce 的 `scatter/gather`，还会减少未完成发送计数：

```cpp
if (send_to_finish_ > 0) {
    send_to_finish_--;
}
```

最后 `NotifyRecv()` 根据消息用途继续推进状态机：

```cpp
if (purpose == "scatter" || purpose == "gather" || purpose == "hop") {
    ScheduleEvent(MakePlayNextReduceEvent(CurrentTime(), this, gpuID));
} else {
    ScheduleEvent(MakePlayNextEvent(CurrentTime(), this, gpuID));
}
```

