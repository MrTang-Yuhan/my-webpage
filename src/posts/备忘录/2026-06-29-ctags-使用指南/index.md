---
layout: post.njk
post_id: 2026-06-29-ctags-使用指南
archive: 备忘录
title: ctags 使用指南
date: 2026-06-29
description: ""
tags:
  - post
---
# ctags + Vim 备忘录

> 参考视频：https://www.bilibili.com/video/BV1JE411G7to

---

## 一、生成 Tags 文件（Shell 中执行）

| 命令 | 说明 |
|------|------|
| `ctags -R .` | 递归扫描当前目录及子目录，生成 `tags` 文件 |
| `ctags *.h *.c` | 仅扫描当前目录下的 `.h` 和 `.c` 文件（不递归） |
| `ctags -R --exclude='*.js' .` | 递归扫描当前目录，**排除**所有 `.js` 文件 |
| `ctags -R --exclude='*.js' . ~/workpath` | 同时扫描当前目录和 `~/workpath`，并排除 `.js` |
| `ctags -R --exclude=.git --exclude=node_modules .` | 递归扫描，排除 `.git` 和 `node_modules` 等目录 |
| `ctags -R --languages=C,C++ .` | 仅扫描 C/C++ 语言文件 |

>**提示**：`--exclude` 后的通配符建议加**单引号**，避免 Shell 提前展开。

---

## 二、Vim 内核心跳转操作

| 快捷键 / 命令 | 简写 | 说明 |
|-------------|------|------|
| `Ctrl + ]` | — | 跳转到光标下符号的定义（若有多匹配，默认跳第一个） |
| `g` + `Ctrl + ]` | — | 若存在多个匹配，**列出候选列表**供选择（比 `Ctrl + ]` 更智能） |
| `Ctrl + t` | — | 返回至上一次跳转位置（可多次回退） |
| `:tags` | `:ts` | **显示标签栈（跳转历史）**，而非候选列表 |
| `:tselect &lt;name&gt;` | `:tse` | 显示符号 `&lt;name&gt;` 的所有匹配候选列表 |
| `:tjump &lt;name&gt;` | `:tj` | 若只有一个匹配则直接跳转，多个则列出候选 |
| `:tnext` | `:tn` | 跳转到下一个匹配的标签 |
| `:tprevious` | `:tp` | 跳转到上一个匹配的标签 |
| `:tfirst` | `:tf` | 跳转到第一个匹配 |
| `:tlast` | `:tl` | 跳转到最后一个匹配 |


---

## 三、预览窗口操作（不离开当前文件）

| 快捷键 / 命令 | 说明 |
|-------------|------|
| `Ctrl + W + }` | 在预览窗口中显示光标下符号的定义，**光标留在原处** |
| `:pclose` | 关闭预览窗口（简写 `:pc`） |
| `Ctrl + W + z` | 关闭预览窗口的快捷键 |
| `Ctrl + W + P` | **将光标直接切换到预览窗口**（大写 P，专门用于 preview） |

---

## 四、窗口与光标移动（含预览窗口）

| 快捷键 | 说明 |
|--------|------|
| `Ctrl + W + P` | **直接跳转到预览窗口**（与 `Ctrl + W + w` 的区别） |
| `Ctrl + W + j` / `k` / `h` / `l` | 向下 / 上 / 左 / 右切换窗口 |
| `Ctrl + W + H` / `J` / `K` / `L` | 将当前窗口移动到最左 / 下 / 上 / 右 |

---

## 五、Vim 配置建议（放入 `.vimrc`）

| 配置项 | 说明 |
|--------|------|
| `set tags=./tags;,tags;` | 自动在当前目录及上级目录递归查找 `tags` 文件 |
| `set ignorecase` | 搜索时忽略大小写（间接改善 tag 查找体验） |

---

