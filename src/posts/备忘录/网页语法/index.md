---
layout: post.njk
title: "网页语法"
date: 2026-05-01
description: "写网页的一些语法介绍"
tags:
  - post
  - 网页语法

---

总体而言，正文支持 markdown 和 html 语法，而脚注的使用仅支持 html 语法 和 部分 markdown 语法。

## 脚注的使用

而脚注的使用仅支持 html 语法 和 部分 markdown 语法。

下面列出一些 markdown 不支持，只能使用 html 语法的部分：

### 脚注插入和编号

- 脚注插入
  ```markdown
  [^1]

  ```

- 脚注内容（使用至少两个空格来表示多行）
  ```bash
  [^1]: xxxxxx
    $aasdad$
    $$
    asdasd
    $$
    `sasd`
  ```

### 脚注插入图片

```markdown
<img src="图片路径" alt="图片名">

```
- **已优化，可以直接使用markdown语法**

### 脚注插入代码块

```markdown
<code> x </code>
```

- **已优化，可以直接使用markdown语法**


### 脚注加粗和斜体

```markdown
<strong> xxx </strong>
<em> xxx </em>
```

- **已优化，可以直接使用markdown语法**

