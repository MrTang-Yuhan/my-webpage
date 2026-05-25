---
layout: post.njk
post_id: 2026-05-25-triosim-模拟器-二-trace-的产生和处理
archive: triosim模拟器
title: " TrioSim 模拟器 （二）：Trace 的产生和处理"
date: 2026-05-25
tags:
  - post
---
# 生成 Trace 并转换为 TrioSim 格式

Trace 的采集与格式转换主要由以下两个脚本完成：

- `tracer/datacollect.py`
- `tracer/dataprocess.py`

具体使用方法可参考：[TrioSim 模拟器（一）：事件驱动模拟](https://my-webpage-adu.pages.dev/posts/triosim%E6%A8%A1%E6%8B%9F%E5%99%A8/2026-05-24-triosim-%E6%A8%A1%E6%8B%9F%E5%99%A8-%E4%B8%80-%E4%BA%8B%E4%BB%B6%E9%A9%B1%E5%8A%A8%E6%A8%A1%E6%8B%9F/)。

由于本文重点介绍模拟器本身的运行机制，因此暂不展开脚本的具体实现细节，后续将另行补充说明。

---

# TrioSim 模拟器加载 TrioSim 格式 Trace

TrioSim 模拟器启动时，会在 `triosim/main.cpp` 中通过如下代码加载 Trace 数据：

```cpp
int main()
{
    // ...

    // 加载 TrioSim 格式的 Trace
    triosim::Trace trace = LoadTrace(config.batch_size, config.batch_size_sim);

    // ...
}
```

Trace 加载过程主要包括以下三部分：

1. 对数据规模和执行时间进行缩放；
2. 解析 `tensor.csv`；
3. 解析 `trace.csv`。

---

## 数据规模与执行时间缩放

模拟器会根据运行时参数 `--batch-size` 和 `--batch-size-sim` 计算缩放比例。

其中，`--batch-size` 必须与使用 `tracer/datacollect.py` 采集 Trace 时的 batch size 保持一致；`--batch-size-sim` 则表示模拟时希望使用的 batch size。

缩放比例计算公式为：

$$
\frac{\text{batch size}}{\text{batch size sim}}
$$

该比值可以大于 1，也可以小于 1。代码中在函数：

```cpp
triosim::Trace LoadTrace(int batchSizeTrace, int batchSizeSim)
```

内通过如下语句实现：

```cpp
double bsRatio = static_cast<double>(batchSizeTrace) / static_cast<double>(batchSizeSim);
```

程序假设所有数据大小和算子执行时间都与 batch size 成正比，因此会通过除以 `bsRatio` 的方式，对数据规模和执行时间进行相应缩放。

---

## 解析 `tensor.csv`

`tensor.csv` 由 `trace.cpp` 中的以下函数负责解析：

```cpp
std::map<std::string, Tensor> TraceLoader::ReadTensors()
```

解析得到的张量信息会存储到 `trace.hpp` 中定义的 `Tensor` 类中。

`tensor.csv` 的表头如下：

```csv
Index,TensorID,TensorShape,TensorNumElement,TensorEachByte,TensorType,TensorStorgeid,gpuid
```

以 `sample_trace/trace2-h100-bs128/vgg13/tensor.csv` 为例，前几行内容如下：

```csv
0,4,"[128, 3, 224, 224]",19267584,4,input,5,0
1,6,"[64, 3, 3, 3]",1728,4,weight,7,0
2,8,[64],64,4,bias=None,9,0
3,14,"[128, 64, 224, 224]",411041792,4,output,15,0
4,14,"[128, 64, 224, 224]",411041792,4,input,15,0
```

各字段含义如下：

- `Index`：张量唯一索引，按顺序递增。
- `TensorID`：张量 ID，用于在 tensor map 中索引该张量的信息。
- `TensorShape`：张量形状。
- `TensorNumElement`：张量元素个数。
- `TensorEachByte`：每个张量元素占用的字节数。
- `TensorType`：张量类型。
- `TensorStorgeid`：暂未使用。
- `gpuid`：张量所属的 GPU ID。

---

## 解析 `trace.csv`

`trace.csv` 由 `trace.cpp` 中的以下函数负责解析：

```cpp
Trace TraceLoader::ReadLayers(std::map<std::string, Tensor>& tensors)
```

解析得到的算子信息会存储到 `trace.hpp` 中定义的 `Layer` 类中。

`trace.csv` 的表头如下：

```csv
OperatorID,OperatorName,Operator_input,Operator_output,Operator_cudatime,Operator_cudatimenooverlap,InputSize,OutputSize,gpuid,stage,tpflag
```

以 `sample_trace/trace2-h100-bs128/vgg13/trace.csv` 为例，前几行内容如下：

```csv
1,aten::conv2d,[4; 6; 8],[14],2673,2671,[19267584; 1728; 64],[411041792],0,forward,1
2,aten::relu_,[14],[14],1083,1083,[411041792],[411041792],0,forward,0
3,aten::conv2d,[14; 22; 24],[30],5901,5892,[411041792; 36864; 64],[411041792],0,forward,1
4,aten::relu_,[30],[30],1081,1081,[411041792],[411041792],0,forward,0
```

各字段含义如下：

- `OperatorID`：层唯一索引，按顺序递增。
- `OperatorName`：当前层的算子名称。
- `Operator_input`：输入张量列表。其值对应 `tensor.csv` 中的 `TensorID`，用于在 tensor map 中索引对应张量的信息。
- `Operator_output`：输出张量列表。其值对应 `tensor.csv` 中的 `TensorID`，用于在 tensor map 中索引对应张量的信息。
- `Operator_cudatime`：当前未使用。
- `Operator_cudatimenooverlap`：当前层的实际执行时间，单位为微秒，即 us。
- `InputSize`：输入张量的元素总数。
- `OutputSize`：输出张量的元素总数。
- `gpuid`：当前层所属的 GPU ID。
- `stage`：当前层所属的传播阶段，例如 `"forward"` 或 `"backward"`。
- `tpflag`：当前层是否可以参与张量并行。`1` 表示可以，`0` 表示不可以。

所有解析得到的 layer 会共同组成一条 Trace。代码中定义为：

```cpp
using Trace = std::vector<Layer*>;
```

模拟器后续的执行过程都会基于该 Trace 进行处理。
