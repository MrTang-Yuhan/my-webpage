---
layout: post.njk
post_id: 2026-05-27-triosim-模拟器-六-分布式并行策略
archive: triosim模拟器
title: TrioSim 模拟器 （六）：Cases 总结
date: 2026-05-27
tags:
  - post
---
TrioSim 模拟器的 Cases 有:

- Case 0: forward 推理。

- Case 1: forward + 梯度 AllReduce。

- Case 2: 数据并行训练。<br>
参考资料：
[Data-Parallelism数据并行（一）](https://my-webpage-adu.pages.dev/posts/model-parallelism/Data-Parallelism%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%B8%80%EF%BC%89/)；<br>[Data-Parallelism数据并行（二）](https://my-webpage-adu.pages.dev/posts/model-parallelism/Data-Parallelism%E6%95%B0%E6%8D%AE%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%BA%8C%EF%BC%89/)
- Case 3: 张量并行训练。<br>
参考资料：
[Tensor Parallelism张量并行（一）](https://my-webpage-adu.pages.dev/posts/model-parallelism/tensor-parallelism%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%B8%80%EF%BC%89/)<br>
[Tensor Parallelism张量并行（二）](https://my-webpage-adu.pages.dev/posts/model-parallelism/tensor-parallelism%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%BA%8C%EF%BC%89/)<br>
[Tensor Parallelism张量并行（三）](https://my-webpage-adu.pages.dev/posts/model-parallelism/tensor-parallelism%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%B8%89%EF%BC%89/)
- Case 4: 流水线并行训练。<br>
参考资料：
[Pipeline-Parallelism流水线并行（一）](https://my-webpage-adu.pages.dev/posts/model-parallelism/Pipeline-Parallelism%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%B8%80%EF%BC%89/)
- Case 5: 通信/Hop AllReduce 算法。<br>
参考资料：[Hop: Heterogeneityaware decentralized training](https://arxiv.org/abs/1902.01064)

其中，
- **Case 0 和 Case 1 是最简单的实现，建议从这两个案例开始阅读代码。**
- Case 2 到 Case 4 展示了多种分布式并行策略。
- Case 5 我目前也还没看对应的论文，只看了代码，实现也不算很复杂。

> TrioSim 模拟器的代码实现中存在诸多简化，这可能会对模拟效果产生影响。
