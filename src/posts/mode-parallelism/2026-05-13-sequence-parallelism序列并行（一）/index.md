---
layout: post.njk
archive: mode-parallelism
title: Sequence Parallelism序列并行（一）
date: 2026-05-14
tags:
  - post
---
原始论文 [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)

# 序列并行

序列并行是在张量并行的基础上进行的进一步深度优化，旨在减少“中间值”带来的显存占用[^1](“中间值”是反向传播所必需的。如果不保存这些中间值，在反向传播过程中就必须重新执行前向计算来生成它们，这会显著增加训练的时间开销。)。

关于 Transformer 各层的显存占用分析，请参考我的文章：

* [Transformer 模型 GPU 显存分析（三）：反向传播需要保存哪些中间结果？](https://my-webpage-adu.pages.dev/admin/#/collections/posts/entries/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%98%BE%E5%AD%98%E5%92%8Cflops%E5%88%86%E6%9E%90/transformer%E6%A8%A1%E5%9E%8B%E7%9A%84gpu%E6%98%BE%E5%AD%98%E4%BD%BF%E7%94%A8%E5%88%86%E6%9E%90%EF%BC%88%E4%B8%89%EF%BC%89%EF%BC%9A%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%88%B0%E5%BA%95%E9%9C%80%E8%A6%81%E5%93%AA%E4%BA%9B%E4%B8%AD%E9%97%B4%E7%BB%93%E6%9E%9C/index)

关于张量并行，请参考我的文章：

* [Tensor Parallelism张量并行（一）](https://my-webpage-adu.pages.dev/posts/mode-parallelism/tensor-parallelism%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%B8%80%EF%BC%89/)
* [Tensor Parallelism张量并行（二）](https://my-webpage-adu.pages.dev/posts/mode-parallelism/tensor-parallelism%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%BA%8C%EF%BC%89/)
* [Tensor Parallelism张量并行（三）](https://my-webpage-adu.pages.dev/posts/mode-parallelism/tensor-parallelism%E5%BC%A0%E9%87%8F%E5%B9%B6%E8%A1%8C%EF%BC%88%E4%B8%89%EF%BC%89/)

# “中间值”显存占用分析

![](img/transformer-arch.png)

![](img/softmax-attention-block.png)
