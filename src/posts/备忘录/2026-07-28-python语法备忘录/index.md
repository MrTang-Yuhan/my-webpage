---
layout: post.njk
post_id: 2026-07-28-python语法备忘录
archive: 备忘录
title: Python 和 Pytorch 语法备忘录
date: 2026-07-28
tags:
  - post
---
# Python

## 使用
- [Python 魔法方法](https://zhuanlan.zhihu.com/p/436732709)：比如 `__repr__`、`__init__` 等。
- [Python 装饰器](https://zhuanlan.zhihu.com/p/1916230371353269505)：常用如 `@property`（属性化访问）、`@dataclass`（自动生成 `__init__`/`__repr__`）。

## 注释
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)：在指引 AI 进行编程时，可以要求遵循该规范。

> 扩展：其他语言同样有 Goole xx-Language Style Guide。

## 安装
- [可编辑安装](https://pip.pypa.io/en/stable/topics/local-project-installs/)：`pip install -e .`，改源码即时生效，适合开发期。

# PyTorch

## 使用
- [CUDA Graph](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/torch-integration.html)：将一连串 kernel 启动录制为图，replay 一次提交，消除逐 kernel 的 CPU 启动开销；输入需 `copy_` 进静态 buffer，decode 场景收益明显。

## 载入 LLM 模型
- 国内可用 [魔塔 ModelScope](https://www.modelscope.cn/models) 下载模型，平替 HuggingFace Hub；也可作为 transformers 库的平替（如果模型有的话）。
