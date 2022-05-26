from dataclasses import replace
from turtle import Turtle
import torch.nn as nn
from collections import OrderedDict
import math

#把channel变为8的整数倍
from torchsummary import summary


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 确保四舍五入的降幅不超过10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

# coding 2020-5-12
def autopad(k, p=None):
    if p is None:
        p = k//2 if isinstance(k, int) else [x//2 for x in k]
    return p

# 深度卷积
def DS_Conv(in_ch, out_ch, k, s, act = True):
    return Conv(in_ch, out_ch, k, s,  g = math.gcd(in_ch, out_ch), actFun ="ReLU6")

class Conv(nn.Module):
    # 标准卷积操作
    # 输入通道数，输出通道数，卷积核, 步长，分组，边界填充，激活函数
    def __init__(self, ch_in, ch_out, k, s, g, actFun, p=None, act = True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.9, eps = 1e-5)
        # self.act = nn.ReLU6(inplace=True) if actFun =="ReLU6" else nn.Linear(ch_in, ch_out)
        self.act = nn.ReLU6(inplace=True) if actFun == "ReLU6" else self.bn
    
    def forward(self, x):
        # conv_out = self.conv(x)
        # print(f"x_shape:{x.size()}, out_shape:{conv_out.size()}")
        # bn_out = self.bn(conv_out)
        # out = self.act(bn_out)  
        
        # return out
        return self.act(self.bn(self.conv(x)))

class inverstBottleneck(nn.Module):
    # 输入通道数，输出通道数，步长，通道扩充倍数，分组数，shortcut连接
    def __init__(self, ch_in, ch_out, s, e, g = 1, shortcut=True):
        super(inverstBottleneck, self).__init__()
        out_ch_ = int(ch_in * e)
        self.cv1 = Conv(ch_in, out_ch_, k=1, s=1, actFun="ReLU6",g=1)
        self.cv2 = DS_Conv(out_ch_, out_ch_, k=3, s=s)
        self.cv3 = Conv(out_ch_, ch_out, k=1, s=1, actFun="ReLu6", g=1)
        self.add = True if ch_in == ch_out else False

    def forward(self, x):
        # out_cv1 = self.cv1(x)
        # out_cv2 = self.cv2(out_cv1)
        # out_cv3 = self.cv3(out_cv2)
        # out = x + out_cv3 if self.add else out_cv3
        # print(f"bottleneck int size: {x.size}, '\n',bottleneck out size: {out.size()}")
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
        # return out


#定义mobilenetv2
class MobileNet_v2(nn.Module):
    def __init__(self,theta=1,num_classes=10,init_weight=True):
        super(MobileNet_v2, self).__init__()
        #bottleneck网络配置
        #           [in_c, exp_ratio, out_c, stride]
        net_config=[[32,1,16,1],
                    [16,6,24,2],
                    [24,6,32,2],
                    [32,6,64,2],
                    [64,6,96,1],
                    [96,6,160,2],
                    [160,6,320,1]]
        repeat_num=[1,2,3,4,3,3,1] # 重叠bottleNeck数

        modules = nn.Sequential()
        conv = Conv(3, 32, 3, 2, 1,actFun ="ReLU6")#构建mobilenetv2的第一层
        modules.add_module("conv2d", conv)
        is_cennect = 0
         
        for idx,copy_num in enumerate(repeat_num):
            parse=net_config[idx]
            if is_cennect == 0:         
                current_input_c = parse[0]
            current_expand_ratio = parse[1]
            current_output_c = parse[2] 
            current_stride = parse[3]
                
            for i in range(copy_num):
                bottleneck = inverstBottleneck(current_input_c, current_output_c, current_stride, e=current_expand_ratio)
                modules.add_module(f"bottleneck_{idx+1}_{i+1}", bottleneck)
                current_stride = 1
                current_input_c = current_output_c
                is_cennect = 1
            idx +=1
            is_cennect = 0        
        
     
        conv = Conv(320, 1280, 1, 1, 1, actFun="ReLU6", p=None, act = True)
        modules.add_module("conv2d_1_1",conv)
        modules.add_module('avg_pool',nn.AdaptiveAvgPool2d(1))
        self.module = modules
        self.linear=nn.Sequential( nn.Dropout(p=0.5),nn.Linear(_make_divisible(theta*1280),num_classes))
        #初始化权重
        if init_weight:
            self.init_weight()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
            elif isinstance(w, nn.Linear):
                nn.init.normal_(w.weight, 0, 0.01)
                nn.init.zeros_(w.bias)


    def forward(self,x):
        # print("formward_module:", self.module)
        out=self.module(x)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out


if __name__ == '__main__':
    device='cuda'
    # net=MobileNet_v2(theta=0.75).to(device)
    net = MobileNet_v2(theta=1).to(device)
    summary(net,(3,224,224))