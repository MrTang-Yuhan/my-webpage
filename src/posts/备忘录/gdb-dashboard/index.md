---
layout: post.njk
title: "gdb dashboard 工具"
date: 2026-05-01
description: "gdb dashboard 工具指南"
tags:
  - post
  - gdb dashboard
---

# 仓库

[gdb dashboard](https://github.com/cyrus-and/gdb-dashboard) 


# 命令备忘录

## 下载

```bash
wget -P ~ https://github.com/cyrus-and/gdb-dashboard/raw/master/.gdbinit
pip install pygments
vim ~/.gdbinit
```

## help

```bash
help dashboard
help dashboard -layout
```

## layout 切换

- 统一切换：

  ```bash
  dashboard -layout !assembly breakpoints expressions !history !memory !registers source stack !threads variables
  ```

- 单独切换某个模块的开关

  ```bash
  dashboard variables
  ```

## 重定向模块到某个终端

- 查看当前终端的编号

  ```bash
  tty
  ```

- 重定向模块到某个终端

  ```bash
  dashboard -output /dev/pts/1
  dashboard assembly -output /dev/pts/3
  dashboard source -output /dev/pts/2

  ```

- 重定向后，建议将对应模块高度设为 0 实现全屏展示:

  ```bash
  dashboard assembly -style height 0
  ```

## 监视变量

  ```bash
  dashboard expressions watch 变量名 
  ```


## 自定义配置快捷键

一般建议在 `~/.gdbinit` 下存放自定义配置。比如你可以为不同场景定义快捷布局：

- 查看当前的自定义配置

  ```bash
  show user
  ```

- 在 GDB 里输入 srcview 就切换到源码调试布局，asmview 切换到汇编级调试布局。对于经常在源码和汇编之间切换的底层开发工作，这种预设能节省大量重复配置时间。

  ```bash
  define srcview
      dashboard -layout source stack variables
      dashboard source -style height 20
  end

  define asmview
      dashboard -layout registers assembly stack
      dashboard assembly -style height 0
  end
  ```

- 在 GDB 输入 de 查看某个特定的表达式

  ```bash
  define de
      dashboard expressions watch $arg0
  end
  ```


