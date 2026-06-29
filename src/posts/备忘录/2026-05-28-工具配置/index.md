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
set backspace=indent,eol,start "删除自动缩进产生的空白，换行符，删除本次插入开始之前已有的内容
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

syntax on        "启用语法高亮
set showmatch    "匹配括号高亮

set tags=./tags;,tags; "自动在当前目录及上级目录递归查找 tags 文件
set ignorecase         "搜索时忽略大小写（间接改善 ctag 查找体验）
set previewheight=20   "设置预览窗口默认高度为 15 行

nnoremap <F4> :echo expand('%:p')<CR> 按 F4 显示完整路径

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

