---
layout: post.njk
post_id: 2026-05-28-工具配置
archive: 备忘录
title: 工具配置
date: 2026-05-28
description: vim, tmux, nvcc 配置
tags:
  - post
---
# vim 配置

写入 ` ~/.vimrc`:
```bash

set number
set relativenumber
set smartindent
set autoindent
set smarttab
set tabstop=4
set shiftwidth=4
set expandtab
set nobackup
set cursorline
set showcmd
set noswapfile
set nowritebackup
set noundofile

```

# tmux 配置

安装 tmux:
```bash
apt-get update
apt-get upgrade
apt-get install tmux
```

写入： `~/.tmux.conf`
```bash
unbind C-b
set -g prefix C-a
bind C-a send-prefix
```

# NVCC 配置

写入 ` ~/.bashrc`:

```bash
# cd /usr/local  # 查看已有CUDA库
# nvcc --version     # 查看当前CUDA版本
export CUDA_INSTALL_PATH=/usr/local/cuda
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
```

载入配置：`source ~/.bashrc`

