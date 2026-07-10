---
layout: post.njk
post_id: 2026-07-04-docker-出现中文乱码
archive: 备忘录
title: docker 环境问题
date: 2026-07-04
description: ""
tags:
  - post
---
# docker 出现中文乱码 

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

---

# conda 环境提醒空间不足

在 docker 容器内，使用 conda 加载环境时，**提醒空间不足**。

## 解决方法一：

```bash
du -ah /dev/shm

# 删除不需要的共享内存，比如
rm /dev/shm/mpich_shm_*

# 全部释放
rm /dev/shm/*
```

## 解决方法二（一劳永逸）
在容器创建时，就指定共享内存的大小。

---

# docker 环境配置

```bash
docker run -itd --gpus all --ipc=host --runtime=nvidia  --ulimit memlock=-1 --ulimit stack=67108864   -v /home/tangyuhan/workpath/docker-data/infra:/root --name tangyuhan-infra-tech nvcr.io/nvidia/pytorch:26.01-py3 /bin/bash
```

- `-itd`: 容器在后台运行，但保留交互能力，后续可通过 `docker exec -it tangyuhan-infra-tech /bin/bash` 进入
- `--gpus all`: 通过 NVIDIA Container Toolkit 将主机上所有 GPU 暴露给容器。
- `--runtime=nvidia`: 显式指定容器运行时（runtime）为 nvidia。
- `--ipc=host`: 将容器的 IPC（Inter-Process Communication）命名空间与主机共享。让容器直接使用主机的 /dev/shm（通常为主机内存的 50%，如 256GB 服务器对应 128GB），彻底消除共享内存瓶颈。
- `--ulimit memlock=-1`: 解除容器内进程的 locked memory（锁定内存）限制。
- `--ulimit stack=67108864`: 设置容器内进程的 栈大小上限为 67,108,864 bytes = 64 MiB。
- `-v /home/tangyuhan/workpath/docker-data/infra:/root`: 将主机路径挂载到容器内的 /root 目录。


