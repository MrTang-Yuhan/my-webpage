---
layout: post.njk
title: "网页语法介绍"
date: 2026-05-01
description: "写网页的一些语法介绍"
tags:
  - post
  - web guide
---

总体而言，正文支持 markdown 和 html 语法，而脚注的使用仅支持 html 语法 和 部分 markdown 语法。

## 脚注的使用

而脚注的使用仅支持 html 语法 和 部分 markdown 语法。

下面列出一些 markdown 不支持，只能使用 html 语法的部分：

### 脚注插入和编号

```markdown
<sup class="footnote-ref"><a href="#fn1">[1]</a></sup>

<aside id="fn1" class="footnote">
  <p>xxxxxx</p>
</aside>
```

### 脚注插入图片

```markdown
<img src="图片路径" alt="图片名">
```

### 脚注插入代码块

```markdown
<code> x </code>
```

### 脚注加粗和斜体

```markdown
<strong> xxx </strong>
<em> xxx </em>
```
