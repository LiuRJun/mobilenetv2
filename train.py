from model import MobileNet_v2
import torch
import argparse
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
import torch.optim as optim
import torch.nn as nn
from utils import *


def train(opt):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('using {} to train'.format(device))

    #数据预处理
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])

    #下载数据集
    train_datasets=torchvision.datasets.CIFAR10('data',train=True,transform=transform,download=True)
    val_datasets=torchvision.datasets.CIFAR10('data',train=False,transform=transform,download=True)

    #加载数据集
    train_dataloader=DataLoader(train_datasets,batch_size=opt.batch,shuffle=True,num_workers=opt.numworkers,pin_memory=True)
    val_dataloader=DataLoader(val_datasets,batch_size=opt.batch,shuffle=False,num_workers=opt.numworkers,pin_memory=True)


    # 保存路径如果不存在，新建一个
    if not os.path.exists(opt.savepath):
        os.mkdir(opt.savepath)

    #实例化网络
    net=MobileNet_v2(theta=1,num_classes=opt.classNum).to(device)

    #是否冻结权重
    if opt.freeze:
        for name,params in net.named_parameters():
            if  'follow_Conv' not in name and 'linear' not in name:
                params.requires_grad_(False)
            else:
                params.requires_grad_(True)

    #定义优化器和损失函数
    optimizer=optim.SGD([p for p in net.parameters() if p.requires_grad],lr=0.01,momentum=0.9,weight_decay=5e-4,nesterov=True)
    loss=nn.CrossEntropyLoss()
    # 学习率调整策略
    lr_schedule=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=200,min_lr=1e-6)


    start_epoch=0
    #加载权重
    if opt.weights.endswith('.pt') or opt.weights.endswith('.pth'):
        ckpt=torch.load(opt.weights)

        if opt.weights=='weights/mobilenet_v2-b0353104.pth':
            weights={}
            #官方预训练权重
            module_lst = [i for i in net.state_dict()]
            for idx, (k, v) in enumerate(ckpt.items()):
                if net.state_dict()[module_lst[idx]].numel() == v.numel():
                    weights[module_lst[idx]] = v
            net.load_state_dict(weights, strict=False)

        else:
            #我们自己训练的权重
            net.load_state_dict(ckpt['model'])  #加载权重
            start_epoch=ckpt['epoch']+1
            optim_pares=ckpt['optim']
            optimizer.load_state_dict(optim_pares)



    #开始训练
    for epoch in range(start_epoch,opt.epoches):
        #训练
        mean_loss=train_per_epoch(net,optimizer,loss,lr_schedule,epoch,train_dataloader,device,opt.printf,opt.batch)
        writer.add_scalar('train_loss',mean_loss,epoch)

        #验证
        val_accuracy=val(val_dataloader,net,device,epoch)
        writer.add_scalar('val_acc',val_accuracy,epoch)

        #保存模型
        par_save_path=os.path.join(opt.savepath,'mobilenet_v2_{}.pth'.format(epoch))
        save_params={
            'model':net.state_dict(),
            'epoch':epoch,
            'optim':optimizer.state_dict()
        }
        torch.save(save_params,par_save_path)


if __name__ == '__main__':
    parse=argparse.ArgumentParser()  # 参数解释器
    parse.add_argument('--epoches',type=int,default=30,help='train  epoches')
    parse.add_argument('--batch',type=int,default=64,help='batch size')
    parse.add_argument('--freeze',type=bool,default=False,help='freeze some weights')
    parse.add_argument('--weights',type=str,default='weights/mobilenet_v2-b0353104.pth',help='last weight path')
    parse.add_argument('--numworkers', type=int, default=8)
    parse.add_argument('--savepath',type=str,default='weights',help='model savepath')
    parse.add_argument('--printf',type=int,default=100,help='print training info after 50 batch')
    parse.add_argument('--classNum',type=int,default=10,help='classes num')

    opt=parse.parse_args()
    print(opt)
    writer=SummaryWriter('runs')
    train(opt)