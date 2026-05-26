---
layout: post.njk
post_id: 2026-05-26-软件调优-六-checkpoint-精度选择与内存占用优化
archive: 训练优化
title: 软件调优（六）：Checkpoint 文件优化
date: 2026-05-26
tags:
  - post
---
# 对 Checkpoint 进行脚本处理

- [torch-checkpoint-convert-to-bf16](#torch-checkpoint-convert-to-bf16)：这个脚本会将两类 checkpoint 文件： **torch 的 ".bin" 文件和 safetensor 的 ".safetensor" 文件中的权重转换为 bf16** ，并在名为 bf16 的子目录下创建一个新的 checkpoint。

  注意事项：该脚本假设所有权重均为浮点张量，因此仅适用于标准的 fp32/fp16 浮点 checkpoint

- [torch-checkpoint-shrink.py](#torch-checkpoint-shrink):这个脚本用于修复某些 ".pt" 文件后缀的 checkpoint：由于某些原因，这些 checkpoint 在保存时，张量对应的底层 storage 比当时实际使用的 view 更大。**它会克隆当前 view，并重新保存张量，使其只保留当前 view 所需的 storage**。

  注意事项：这个脚本会直接覆盖原 checkpoint 文件，因此使用前最好先备份

# 脚本

## torch-checkpoint-convert-to-bf16

<a id="torch-checkpoint-convert-to-bf16"></a>

`torch-checkpoint-convert-to-bf16`
```bash
#!/bin/bash

# 这个脚本会将两类 checkpoint 文件： torch 的 *.bin 文件和 safetensor 的 *.safetensor 文件
# 中的权重转换为 bf16，并在名为 bf16 的子目录下创建一个新的 checkpoint
#
# 注意事项：该脚本假设所有权重均为浮点张量，因此仅适用于标准的 fp32/fp16 浮点 checkpoint
# 
# 使用方法：
# cd checkpoint
# bash torch-checkpoint-convert-to-bf16

# 设置目标目录
target_dir=bf16

echo "creating a new checkpoint under dir $target_dir"
mkdir -p $target_dir

# 复制 config 和其他文件，可按需调整；也可以执行 `cp * $target_dir`
cp *json *model $target_dir

# 转换 *bin 文件
echo "converting *bin torch files"
python -c "import torch, sys; [torch.save({k:v.to(torch.bfloat16) for k,v in torch.load(f).items()}, f'{sys.argv[1]}/{f}') for f in sys.argv[2:]]" $target_dir *bin

# 转换 *safetensors 文件，来源是原始的 *bin 文件
if compgen -G "*.safetensors" > /dev/null; then
    echo "converting *safetensors files"
    cd $target_dir
    python -c "import re, sys, torch; from safetensors.torch import save_file; [save_file(torch.load(f), re.sub(r'.*?(model.*?)\.bin',r'\1.safetensors',f), metadata={'format': 'pt'}) for f in sys.argv[1:]]" *bin
    if test -e "pytorch_model.bin.index.json"; then
        cp pytorch_model.bin.index.json model.safetensors.index.json
        perl -pi -e 's|pytorch_||; s|\.bin|.safetensors|' model.safetensors.index.json
    fi
    cd - > /dev/null
fi

echo "the dir $target_dir now contains a copy of the original checkpoint with bf16 weights"
```

## torch-checkpoint-shrink

<a id="torch-checkpoint-shrink"></a>

`torch-checkpoint-shrink.py`

```python
#!/usr/bin/env python

# 这个脚本用于修复某些 ".pt" 文件后缀的 checkpoint：由于某些原因，这些 checkpoint 在保存时，
# 张量对应的底层 storage 比当时实际使用的 view 更大。
# 它会克隆当前 view，并重新保存张量，使其只保留当前 view 所需的 storage。
#
# 注意事项：这个脚本会直接覆盖原 checkpoint 文件，因此使用前最好先备份
#
#
# 示例：
#
# 1. 处理 checkpoint 中的所有文件
# ./torch-checkpoint-shrink.py --checkpoint_dir ./checkpoints/global_step10
#
# 2. 只处理 checkpoint 中匹配多个模式的指定文件
# ./torch-checkpoint-shrink.py --checkpoint_dir ./checkpoints/global_step10 --patterns 'layer*pt' 'zero*pt'

import argparse
import torch
import glob
import os
import collections.abc
from fnmatch import fnmatch

debug = 0

# 加载到 CPU
device = torch.device('cpu')

def get_pt_files(checkpoint_dir, patterns):

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Directory '{checkpoint_dir}' doesn't exist")

    pt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt")))

    if len(pt_files) == 0:
        raise FileNotFoundError(
            f"can't find '*.pt' files in directory '{checkpoint_dir}'")

    # 按模式过滤文件，只匹配文件名部分，不包含任何父目录
    pt_files = [f for f in pt_files for p in patterns if fnmatch(os.path.basename(f), p)];

    return pt_files


# 当从检查点（checkpoint）加载模型时，张量可能共享底层存储（storage）。例如：
# """
#   weight = torch.randn(1000, 1000)
#   bias = weight[0]  # 与 weight 共享存储
# """
# 通过 .clone() 为每个张量创建独立的内存副本，丢弃未使用的 storage 部分
def shrink_dict_values(d, prefix=""):
    for k, v in d.items():
        k_full = f"{prefix}.{k}" if len(prefix) else k
        if isinstance(v, collections.abc.Mapping):
            shrink_dict_values(v, k_full)
        else:
            if debug:
                print(f"{k_full}")
            if v is not None and torch.is_tensor(v):
                d[k] = v.clone() # 丢弃任何未使用的 storage

def shrink_pt_file(f):
    print(f"-> {f}")
    size_before = os.path.getsize(f)
    sd = torch.load(f, map_location=device)     # 加载 .pt 文件
    shrink_dict_values(sd)
    torch.save(sd, f)                           # 覆盖原先的 .pt 文件
    size_after = os.path.getsize(f)
    size_delta = size_before - size_after       # 统计节省的内存空间    
    if debug:
        print(f"before {size_before / 2**20:.2f}MB, after {size_after / 2**20:.2f}MB, saved {size_delta / 2**20:.2f}MB")
    return size_before, size_after, size_delta

def checkpoint_shrink(checkpoint_dir, patterns):
    """
    参数：
        - ``ds_checkpoint_dir``：deepspeed checkpoint 文件夹路径，也就是 optimizer 文件所在的位置
    """
    print(f"Processing zero checkpoint '{checkpoint_dir}'")
    pt_files = get_pt_files(checkpoint_dir, patterns)
    before, after, delta = 0, 0, 0
    for f in pt_files:
        size_before, size_after, size_delta = shrink_pt_file(f)
        before += size_before
        after  += size_after
        delta  += size_delta
    print(f"Done. Before {before / 2**20:.2f}MB, after {after / 2**20:.2f}MB, saved {delta / 2**20:.2f}MB")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, help="path to the desired checkpoint folder, e.g., path/checkpoints/global_step10")
    parser.add_argument("--patterns", nargs='+', default="*.pt", required=False, type=str, help="one or more patterns of checkpoint files - make sure to quote those! by default all *.pt files")
    parser.add_argument("-d", "--debug", action='store_true', help="enable debug")
    args = parser.parse_args()

    debug = args.debug

    checkpoint_shrink(args.checkpoint_dir, args.patterns)
```
