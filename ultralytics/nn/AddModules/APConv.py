import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k, C3k2
from torchvision.ops import deform_conv2d

class PConv(nn.Module):
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''

    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


# class CoordAtt(nn.Module):
#     ''' Coordinate Attention Module (CVPR 2021) '''
#
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         # 通道缩减比例
#         self.reduction = reduction
#
#         # 水平方向和垂直方向的池化
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         # 中间通道数
#         mid_channels = max(8, inp // reduction)
#
#         # 1x1卷积用于特征变换
#         self.conv1 = nn.Conv2d(inp, mid_channels, 1, 1, 0)
#         self.bn1 = nn.BatchNorm2d(mid_channels)
#         self.act = nn.ReLU(inplace=True)
#
#         # 水平卷积分支
#         self.conv_h = nn.Conv2d(mid_channels, oup, 1, 1, 0)
#         # 垂直卷积分支
#         self.conv_w = nn.Conv2d(mid_channels, oup, 1, 1, 0)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         identity = x
#         n, c, h, w = x.size()
#
#         # 水平方向池化 [n,c,h,1]
#         x_h = self.pool_h(x)
#         # 垂直方向池化 [n,c,1,w]
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         # 拼接两个方向的特征
#         y = torch.cat([x_h, x_w], dim=2)
#         # 特征变换
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         # 分割特征图
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         # 调整垂直特征的维度
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         # 生成注意力权重
#         a_h = self.sigmoid(self.conv_h(x_h))
#         a_w = self.sigmoid(self.conv_w(x_w))
#
#         # 应用注意力权重
#         out = identity * a_w * a_h
#         return out


# class PConv(nn.Module):
#     ''' 集成坐标注意力的Pinwheel卷积 '''
#
#     def __init__(self, c1, c2, k, s):
#         super().__init__()
#
#         # 非对称填充层
#         p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
#         self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
#
#         # 四个方向的分离卷积
#         self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
#         self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
#
#         # 特征融合卷积
#         self.cat = Conv(c2, c2, 2, s=1, p=0)
#
#         # 坐标注意力模块
#         self.ca = CoordAtt(c2, c2)
#
#     def forward(self, x):
#         # 四个方向的特征提取
#         yw0 = self.cw(self.pad[0](x))
#         yw1 = self.cw(self.pad[1](x))
#         yh0 = self.ch(self.pad[2](x))
#         yh1 = self.ch(self.pad[3](x))
#
#         # 特征拼接和融合
#         y = torch.cat([yw0, yw1, yh0, yh1], dim=1)
#         y = self.cat(y)
#
#         # 应用坐标注意力
#         y = self.ca(y)
#         return y

# class DeformableConv(nn.Module):
#     def __init__(self, c1, c2, k, s=1, p=0, d=1, bias=False):
#         super().__init__()
#         if isinstance(k, int):
#             k = (k, k)
#         self.k = k
#         self.p = p
#         self.s = s
#         self.d = d
#         self.offset = nn.Conv2d(c1, 2 * k[0] * k[1], kernel_size=k, stride=s, padding=p, dilation=d, bias=True)
#         self.weight = nn.Parameter(torch.randn(c2, c1, *k)) # 注意：形状为 (out_channels, in_channels, kh, kw)
#         self.bias = nn.Parameter(torch.zeros(c2)) if bias else None
#         self.bn = nn.BatchNorm2d(c2)
#         self.act = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         offset = self.offset(x)
#         x = deform_conv2d(x, offset, self.weight, self.bias, stride=self.s, padding=self.p, dilation=self.d)
#         x = self.bn(x)
#         return self.act(x)
#
# # 假设已有的 CoordAtt 和 Conv
# # from .coord_attention import CoordAtt
# # from .conv_module import Conv
#
# class PConv(nn.Module):
#     ''' 集成可变形卷积与坐标注意力的 Pinwheel 卷积 '''
#
#     def __init__(self, c1, c2, k, s):
#         super().__init__()
#
#         # 非对称填充层（保持不变）
#         p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
#         self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
#
#         # 使用可变形卷积替代普通卷积
#         self.cw = DeformableConv(c1, c2 // 4, (1, k), s=s, p=0)
#         self.ch = DeformableConv(c1, c2 // 4, (k, 1), s=s, p=0)
#
#         # 融合卷积与注意力（保持不变）
#         self.cat = Conv(c2, c2, 2, s=1, p=0)
#         self.ca = CoordAtt(c2, c2)
#
#     def forward(self, x):
#         # 四方向特征提取（保持逻辑）
#         yw0 = self.cw(self.pad[0](x))
#         yw1 = self.cw(self.pad[1](x))
#         yh0 = self.ch(self.pad[2](x))
#         yh1 = self.ch(self.pad[3](x))
#
#         # 融合与注意力
#         y = torch.cat([yw0, yw1, yh0, yh1], dim=1)
#         y = self.cat(y)
#         return self.ca(y)







class APC2f(nn.Module):
    """Faster Implementation of APCSP Bottleneck with Asymmetric Padding convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, P=True, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        if P:
            self.m = nn.ModuleList(
                APBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        else:
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))


class APBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        p = [(2, 0, 2, 0), (0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cv1 = Conv(c1, c_ // 4, k[0], 1, p=0)
        # self.cv1 = nn.ModuleList([nn.Conv2d(c1, c_, k[0], stride=1, padding= p[g], bias=False) for g in range(4)])
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        # y = self.pad[g](x) for g in range(4)
        return x + self.cv2((torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1))) if self.add else self.cv2(
            (torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1)))


class C3k_AP(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(APBottleneck(c_, c_) for _ in range(n)))


class C3k2_AP(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_AP(self.c, self.c, 2, shortcut, g) if c3k else APBottleneck(self.c, self.c) for _ in range(n))