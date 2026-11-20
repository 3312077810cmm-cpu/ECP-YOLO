######################################## APConv AAAI2025   by AI Little monster start  ########################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k, C3k2


from mmengine.model import caffe2_xavier_init, constant_init
from ultralytics.nn.modules.conv import Conv

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.norm = None  # 可添加 BatchNorm2d 如果需要
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


def constant_init(module, val):
    if isinstance(module, nn.Conv2d):
        nn.init.constant_(module.weight, val)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def caffe2_xavier_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class ContextAggregation(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        self.a = ConvModule(in_channels, 1, kernel_size=1, act=False)
        self.k = ConvModule(in_channels, 1, kernel_size=1, act=False)
        self.v = ConvModule(in_channels, self.inter_channels, kernel_size=1, act=False)
        self.m = ConvModule(self.inter_channels, in_channels, kernel_size=1, act=False)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        n = x.size(0)
        c = self.inter_channels

        a = self.a(x).sigmoid()  # [N, 1, H, W]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)  # [N, 1, HW, 1]
        v = self.v(x).view(n, 1, c, -1)             # [N, 1, C, HW]

        y = torch.matmul(v, k).view(n, c, 1, 1)     # [N, C, 1, 1]
        y = self.m(y) * a                           # broadcast 和 attention 相乘

        return x + y

class ContextAggregation(nn.Module):

    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        # n, c = x.size(0)
        n = x.size(0)
        c = self.inter_channels
        # n, nH, nW, c = x.shape

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y


class PConv_att(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)
        self.att = ContextAggregation(c1)

    def forward(self, x):
        x = self.att(x)
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

