import os
import time
import torch
import torchvision
import sys
import numpy as np
import torchvision.models as models
from torchvision import transforms
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import ExecutionTraceObserver

bsinput=int(sys.argv[1])
listmodel=[]
num_iters = 43
# 预置常见 CNN 模型；后续会逐个模型采集 profiler 与 graph trace。
vgg11 = models.vgg11(weights=None)
vgg13 = models.vgg13(weights=None)
vgg16 = models.vgg16(weights=None)
vgg19 = models.vgg19(weights=None)

resnet18 = models.resnet18(weights=None)
resnet34 = models.resnet34(weights=None)
resnet50 = models.resnet50(weights=None)
resnet101 = models.resnet101(weights=None)
resnet152 = models.resnet152(weights=None)

densenet161 = models.densenet161(weights=None)
densenet169 = models.densenet169(weights=None)
densenet121 = models.densenet121(weights=None)
densenet201 = models.densenet201(weights=None)

bslist=[bsinput]

data_transforms = {
    'predict': transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])}

device = torch.device("cuda:0")
# 打印设备信息，便于确认采集环境与 GPU 一致性。
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
   
for bs in bslist:
    
    BATCH_SIZE = bs
    EPOCHS = 1
    WORKERS = 8
    IMG_DIMS = (336, 336)
    CLASSES = 10

    # 采集阶段只需稳定、可复现的小样本输入；这里固定取前 BATCH_SIZE 张图像。
    dataset = {'predict' : torchvision.datasets.ImageFolder("./ILSVRC2012_img_val", data_transforms['predict'])}
    dataset_subset=dataset['predict']
    dataset_subset = torch.utils.data.Subset(dataset['predict'],range(BATCH_SIZE))
    data_loader = {'predict': torch.utils.data.DataLoader(dataset_subset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False,
                                            num_workers=WORKERS)}

    listmodel=[vgg11,vgg13,vgg16,vgg19,resnet18,resnet34,resnet50,resnet101,resnet152,
              densenet161,densenet169,densenet121,densenet201]
    namelist = ['vgg11','vgg13','vgg16','vgg19','resnet18','resnet34','resnet50','resnet101','resnet152',
              'densenet161','densenet169','densenet121','densenet201']
    j = 0
    for model in listmodel:
        name = namelist[j]
        print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}, Workers: {WORKERS}, Network: {name}")
        j=j+1
        try: 
            model = model.to(device)
            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(EPOCHS): # 仅跑 1 个 epoch 用于采样
                    model.train()
                    for batch in tqdm(data_loader['predict'], total=len(data_loader['predict'])):
                        features, labels = batch[0].to(device), batch[1].to(device)
                        print( "Batchsize:", features.size())
                        total_time1 = 0
                        total_time2 = 0
                        # 迭代分三段：
                        # 1) 前 0~30 轮：纯预热，稳定 cudnn/caching/编译状态
                        # 2) 31~40 轮：只计时，不导出文件，用于估算稳定耗时
                        # 3) 41~42 轮：计时 + 导出 profiler/graph，作为离线处理输入
                        for i in range(num_iters):
                            if 40<i<=num_iters-1:
                                os.makedirs("./data/graph", exist_ok=True)
                                # ExecutionTraceObserver 导出算子图依赖信息（graph_*.json）。
                                eg = ExecutionTraceObserver()
                                eg.register_callback("./data/graph/graph_"+name+"-iter"+str(i)+".json")
                                eg.start()
                                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True,with_stack=True,profile_memory=True) as prof: # 如需接 tensorboard 可启用 on_trace_ready
                                    # 用 CUDA Event 包裹一步训练，测量端到端 GPU 时间（毫秒）。
                                    torch.cuda.synchronize()
                                    starter = torch.cuda.Event(enable_timing=True)
                                    ender = torch.cuda.Event(enable_timing=True)
                                    starter.record()
                    
                                    optimizer.zero_grad()

                                    preds = model(features)
                                    loss = loss_fn(preds, labels)

                                    loss.backward()
                                    optimizer.step()
                                    
                                    ender.record()
                                    torch.cuda.synchronize()
                                    curr_time = starter.elapsed_time(ender)
                                    total_time1 += curr_time                           
                                os.makedirs("./data/profiler", exist_ok=True)
                                # 导出 Chrome Trace（profiler_*.json），供 dataprocess.py 解析。
                                prof.export_chrome_trace("./data/profiler/profiler_"+name+"-iter"+str(i)+".json")
                                eg.stop()
                                eg.unregister_callback()
                                print(name, "gpu time", curr_time)
                            elif 30<i<=40:
                                # 稳态计时段：用于估计稳定训练步时延，不导出 trace 文件。
                                torch.cuda.synchronize()
                                starter = torch.cuda.Event(enable_timing=True)
                                ender = torch.cuda.Event(enable_timing=True)
                                starter.record()
                                
                                optimizer.zero_grad()
                                preds = model(features)
                                loss = loss_fn(preds, labels)
                                loss.backward()
                                optimizer.step()
                                
                                ender.record()
                                torch.cuda.synchronize()
                                curr_time = starter.elapsed_time(ender)
                                total_time2 += curr_time
                                print(name, "gpu time 2", curr_time)
                            else:
                                # 预热段：仅执行训练步，避免把冷启动噪声带入后续统计。
                                optimizer.zero_grad()
                                preds = model(features)
                                loss = loss_fn(preds, labels)
                                loss.backward()
                                optimizer.step()
                        print("avg profiler Total time1:", total_time1/2)
                        print("avg Total time2:", total_time2/10)
                        
        except Exception as e:
            print(e)
            pass
