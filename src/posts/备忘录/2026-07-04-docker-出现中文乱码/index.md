---
layout: post.njk
post_id: 2026-07-04-docker-出现中文乱码
archive: 备忘录
title: docker 出现中文乱码
date: 2026-07-04
description: ""
tags:
  - post
---
进入容器后执行：

```bash
# 1. 安装 locales 和中文字体
apt update
apt install -y locales fonts-wqy-zenhei fonts-noto-cjk

# 2. 生成 UTF-8 locale
locale-gen zh_CN.UTF-8 en_US.UTF-8

# 3. 设置环境遍历：vim ~/.bashrc，进行如下配置
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# 4. 刷新字体缓存
fc-cache -fv
```

重启容器即可。
