---
layout: post.njk
post_id: 2026-08-10-linux-语法
archive: 备忘录
title: Linux 开发环境配置
date: 2026-08-10
updated: 2026-10-05
tags:
  - post
---
## 一、Vim 配置

将以下内容保存到用户主目录下的 `~/.vimrc` 文件中（如果文件不存在则新建）。

```vim
" ============================================================
" 基础设置
" ============================================================
set nocompatible              " 使用 Vim 增强模式，而不是兼容 vi
filetype plugin indent on     " 开启文件类型检测、插件和缩进
syntax on                     " 开启语法高亮

" ============================================================
" 编码设置
" ============================================================
set encoding=utf-8            " Vim 内部使用 UTF-8
set fileencodings=utf-8,gb18030,gbk,latin1 " 打开文件时自动检测编码
set fileencoding=utf-8        " 新建文件默认使用 UTF-8
set fileformats=unix,dos,mac  " 自动识别文件换行格式

if !has('nvim')
  set termencoding=utf-8      " Neovim 不需要此设置，Vim 需要
endif

set ambiwidth=double          " 让中文等宽字符显示更整齐

" ============================================================
" 显示与界面
" ============================================================
set number                    " 显示绝对行号
set relativenumber            " 显示相对行号，方便跳转
set cursorline                " 高亮当前行
set showcmd                   " 显示未完成的命令
set showmode                  " 显示当前模式
set ruler                     " 显示光标位置
set laststatus=2              " 总是显示状态栏
set cmdheight=1               " 命令行高度
set numberwidth=4             " 行号列宽度

set wildmenu                  " 命令行补全菜单
set wildmode=longest:full,full
set wildignorecase            " 命令行补全忽略大小写
set wildignore=*.o,*.obj,*.pyc,*.class,*.DS_Store,__pycache__

set showmatch                 " 显示匹配的括号
set matchtime=2               " 匹配括号高亮时间，单位 0.1 秒
set scrolloff=5               " 光标上下保留 5 行
set sidescrolloff=5           " 光标左右保留 5 列

" 状态栏内容：文件名、修改标记、文件格式、编码、行列、百分比
set statusline=%f%m%r%h%w\ [%{&ff}]\ [%{&fileencoding?&fileencoding:&encoding}]\ [%l,%c]\ [%p%%]

" 关闭错误响铃，使用视觉闪烁
set noerrorbells
set visualbell
set t_vb=

" 如果需要 80 列参考线，取消下面注释
" set colorcolumn=80

" ============================================================
" 缩进与制表符
" ============================================================
set autoindent                " 新行继承上一行缩进
set smartindent               " 智能缩进，适合 C 风格语言
set smarttab                  " 按 Tab 时根据 shiftwidth 插入
set tabstop=4                 " Tab 显示为 4 个空格
set shiftwidth=4              " 自动缩进使用 4 个空格
set softtabstop=4             " 编辑时按 Tab 或退格按 4 个空格处理
set expandtab                 " 将 Tab 转换为空格

augroup filetype_indent
  autocmd!
  " Makefile 必须使用真正的 Tab，否则会出错
  autocmd FileType make setlocal noexpandtab tabstop=4 shiftwidth=4 softtabstop=0
  " Python 使用 4 空格
  autocmd FileType python setlocal tabstop=4 shiftwidth=4 softtabstop=4 expandtab
  " 前端常用 2 空格
  autocmd FileType javascript,typescript,json,html,css,scss,yaml setlocal tabstop=2 shiftwidth=2 softtabstop=2 expandtab
  " Markdown 使用 4 空格，并开启折行
  autocmd FileType markdown setlocal tabstop=4 shiftwidth=4 softtabstop=4 expandtab wrap linebreak
augroup END

" ============================================================
" 搜索设置
" ============================================================
set incsearch                 " 输入搜索时即时高亮
set hlsearch                  " 高亮所有匹配结果
set ignorecase                " 搜索忽略大小写
set smartcase                 " 如果包含大写字母，则区分大小写

" ============================================================
" 剪贴板与粘贴
" ============================================================
if has('clipboard')
  if has('unnamedplus')
    set clipboard=unnamedplus
  else
    set clipboard=unnamed
  endif
endif

set nopaste

if !has('nvim')
  set pastetoggle=<F2>        " 旧版 Vim 用 F2 切换粘贴模式，避免缩进混乱
endif

nnoremap <leader>y "+y
vnoremap <leader>y "+y
nnoremap <leader>p "+p
vnoremap <leader>p "+p

" ============================================================
" 文件、备份与撤销
" ============================================================
set nobackup                  " 不生成备份文件
set nowritebackup             " 写入时不生成备份
set noswapfile                " 不生成 swap 文件
set noundofile                " 不生成持久撤销文件

set autoread                  " 文件在外部被修改时自动重新读取
set hidden                    " 允许隐藏未保存的缓冲区
set confirm                   " 退出时如果有未保存内容则确认

set history=1000
set undolevels=1000

" ============================================================
" 编辑与操作
" ============================================================
set backspace=indent,eol,start " 退格键可删除缩进、行尾、插入前字符
set whichwrap+=<,>,h,l         " 左右键可以跨行移动
set mouse=a                    " 启用鼠标
set timeoutlen=500             " 映射等待时间
set ttimeoutlen=50             " 终端按键超时
set splitright                 " 垂直分屏在右侧
set splitbelow                 " 水平分屏在下方
set updatetime=300             " 更新时间，影响 CursorHold 等
set completeopt=menuone,noinsert,noselect " 补全菜单行为

set formatoptions+=mM          " 允许在中文等多字节字符间换行

" ============================================================
" 快捷键
" ============================================================
let mapleader=","              " 使用逗号作为 leader 键

nnoremap <leader>w :w<CR>
nnoremap <leader>q :q<CR>
nnoremap <leader>x :wq<CR>
nnoremap <leader>h :nohlsearch<CR>

inoremap jk <Esc>              " 插入模式快速退出

nnoremap <leader>d "_d
vnoremap <leader>d "_d
vnoremap <leader>P "_dP

nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" ============================================================
" 自动命令
" ============================================================
augroup custom_autocmds
  autocmd!
  " 打开文件时恢复到上次光标位置
  autocmd BufReadPost *
        \ if line("'\"") > 0 && line("'\"") <= line("$") |
        \   execute "normal! g`\"" |
        \ endif
augroup END
```

---

## 二、tmux 配置

### 1. 安装 tmux

在终端中执行以下命令进行安装（适用于 Debian/Ubuntu 系统）：

```bash
apt-get update
apt-get upgrade
apt-get install tmux
```

### 2. 配置组合键

默认 tmux 的组合键是 `Ctrl+b`。为了提高效率，很多人会将其修改为 `Ctrl+a`（与 `screen` 保持一致）。

编辑或创建 `~/.tmux.conf` 文件：

```bash
vim ~/.tmux.conf
```

在文件中添加以下内容：

```tmux
# 解除默认的 Ctrl+b 绑定
unbind C-b

# 将前缀键设置为 Ctrl+a
set -g prefix C-a

# 允许通过按 Ctrl+a 两次来发送 Ctrl+a 给终端
bind C-a send-prefix
```

### 3. 加载配置

保存退出后，在 tmux 会话中执行以下命令重新加载配置：

```bash
tmux source-file ~/.tmux.conf
```
*(如果在 tmux 外部，直接重新进入 tmux 即可生效)*

---

## 三、NVCC (CUDA) 配置

### 1. 查看环境

在配置之前，可以先检查系统是否已有 CUDA 库以及当前版本：

```bash
cd /usr/local        # 查看已有 CUDA 库
nvcc --version       # 查看当前 CUDA 版本
```

### 2. 配置环境变量

编辑 `~/.bashrc` 文件：

```bash
vim ~/.bashrc
```

在文件末尾添加以下内容：

```bash
# CUDA 安装路径
export CUDA_INSTALL_PATH=/usr/local/cuda
# 将 CUDA 的 bin 目录添加到 PATH 中
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
```

### 3. 加载配置

保存退出后，执行以下命令让配置立即生效：

```bash
source ~/.bashrc
```

### 4. 验证

再次执行 `nvcc --version`，如果能看到版本信息，说明配置成功。

---
*文档结束*

