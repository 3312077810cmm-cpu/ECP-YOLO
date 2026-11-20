import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k, C3k2


class ContextAggregation(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        # 用标准卷积层替换ConvModule
        self.a = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.k = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.v = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.m = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        # 用PyTorch原生初始化替换mmcv的初始化
        for layer in [self.a, self.k, self.v]:
            init.xavier_normal_(layer.weight)  # 对应caffe2_xavier_init
            if layer.bias is not None:
                init.constant_(layer.bias, 0)

        init.constant_(self.m.weight, 0)  # 对应constant_init
        if self.m.bias is not None:
            init.constant_(self.m.bias, 0)

    def forward(self, x):
        n = x.size(0)
        c = self.inter_channels

        a = torch.sigmoid(self.a(x))  # 手动添加sigmoid激活

        k = self.k(x).view(n, 1, -1, 1).softmax(dim=2)
        v = self.v(x).view(n, 1, c, -1)

        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y


class PConv_att(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=p[g]) for g in range(4)]

        # 用标准卷积层替换Conv
        self.cw = nn.Conv2d(c1, c2 // 4, kernel_size=(1, k), stride=s, padding=0)
        self.ch = nn.Conv2d(c1, c2 // 4, kernel_size=(k, 1), stride=s, padding=0)
        self.cat = nn.Conv2d(c2, c2, kernel_size=2, stride=1, padding=0)

        self.att = ContextAggregation(c1)

    def forward(self, x):
        x = self.att(x)
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))