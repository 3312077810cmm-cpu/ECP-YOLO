import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.layers import trunc_normal_
import math

from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck, C2f, C3k, ABlock, C3k2, A2C2f


class ChannelBranch(nn.Module):
    # Channel Branch
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        channel_att_raw = self.fc(avg_pool)
        channel_att = torch.sigmoid(channel_att_raw).unsqueeze(-1).unsqueeze(-1)
        return x * channel_att


class SpatialBranch(nn.Module):
    # Spatial Branch（手动 reflect padding）
    def __init__(self, in_channels):
        super(SpatialBranch, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.sigmoid(self.conv(F.pad(x, (3, 3, 3, 3), mode='reflect')))
        return x * scale


# Channel Spatial Attention Module
class ChannelSpatialAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelSpatialAttentionModule, self).__init__()
        self.channel_attention = ChannelBranch(in_channels)
        self.spatial_attention = SpatialBranch(in_channels)

    def forward(self, x):
        out = self.channel_attention(x) + self.spatial_attention(x)
        return out


class LocalChannelAttention(nn.Module):
    def __init__(self, dim):
        super(LocalChannelAttention, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3)  # 手动 pad
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, groups=dim)  # 手动 pad
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.shape
        # 1D 注意力分支：先反射填充再卷积
        att = self.GAP(x).reshape(N, 1, C)
        att = self.conv(F.pad(att, (1, 1), mode='reflect')).sigmoid()
        att = att.reshape(N, C, 1, 1)
        # 局部深度卷积分支：先反射填充再卷积
        local = self.sigmoid(self.local_conv(F.pad(x, (1, 1, 1, 1), mode='reflect')))
        out = ((x * att) + x) + (local * x)
        return out


class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.Mish(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class DualAttentionBlock(nn.Module):
    def __init__(self, dim, network_depth=1):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.dim = dim

        # shallow feature extraction layer
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)  # main
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, groups=dim)  # main（手动 pad）

        self.attn = ChannelSpatialAttentionModule(dim)

        # Local Channel Attention
        self.gp = LocalChannelAttention(dim)

        # Global Channel Attention
        self.cam = GlobalChannelAttention(dim)

        # Spatial Attention
        self.pam = SpatialAttention(dim)

        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * 4.), out_features=dim)
        self.mlp2 = Mlp(network_depth, dim * 3, hidden_features=int(dim * 4.), out_features=dim)

    def forward(self, x):
        # Channel Spatial Attention Module
        identity = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.mlp(x)
        x = identity + x

        # Parallel Attention Module
        identity = x
        x = self.norm2(x)
        x = self.conv1(x)
        x = self.conv2(F.pad(x, (2, 2, 2, 2), mode='reflect'))  # 先 pad 再 depthwise 5x5
        x = torch.cat([self.gp(x), self.cam(x), self.pam(x)], dim=1)
        x = self.mlp2(x)
        x = identity + x

        return x


# Global Channel Attention
class GlobalChannelAttention(nn.Module):
    def __init__(self, dim, bias=True):
        super(GlobalChannelAttention, self).__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.ca(x) * x


# Spatial Attention（手动 reflect padding）
class SpatialAttention(nn.Module):
    def __init__(self, dim, bias=True):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(dim, 1, kernel_size=7)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(F.pad(x, (3, 3, 3, 3), mode='reflect'))) * x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, network_depth):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList(
            [DualAttentionBlock(dim=dim, network_depth=network_depth) for _ in range(depth)]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class C3k_DAB(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(DualAttentionBlock(c_) for _ in range(n)))


class C3k2_DAB(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            C3k_DAB(self.c, self.c, 2, shortcut, g) if c3k else DualAttentionBlock(self.c) for _ in range(n)
        )


class A2C2f_DAB(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, )
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k_DAB(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

