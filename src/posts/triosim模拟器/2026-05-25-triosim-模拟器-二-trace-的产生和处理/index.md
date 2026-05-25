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

这两个步骤通过以下两个脚本完成：

- `tracer/datacollect.py`
- `tracer/dataprocess.py`

具体使用方法请参考 [TrioSim 模拟器（一）：事件驱动模拟](https://my-webpage-adu.pages.dev/posts/triosim%E6%A8%A1%E6%8B%9F%E5%99%A8/2026-05-24-triosim-%E6%A8%A1%E6%8B%9F%E5%99%A8-%E4%B8%80-%E4%BA%8B%E4%BB%B6%E9%A9%B1%E5%8A%A8%E6%A8%A1%E6%8B%9F/)。

由于本文重点在于模拟器本身的机制，脚本的具体实现细节暂不展开，后续再另行补充说明。

# TrioSim 模拟器载入 TrioSim 格式 Trace

TrioSim 模拟器运行时，首先通过 `triosim/main.cpp` 中下面的函数载入 Trace：
```
int main()
{
...
// 处理 TrioSim 格式 Trace
triosim::Trace trace = LoadTrace(config.batch_size, config.batch_size_sim);
...
}
```

它的核心功能是三个：




