import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    if p is None:
        p = k // 2
    return p


class RepDWConv(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, act=True):
        super().__init__()
        self.act = nn.SiLU() if act else nn.Identity()

        self.dw_3x3 = nn.Conv2d(
            channels, channels, kernel_size, stride,
            autopad(kernel_size), groups=channels, bias=False
        )
        self.dw_1x1 = nn.Conv2d(
            channels, channels, 1, stride,
            0, groups=channels, bias=False
        )

        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.dw_3x3(x) + self.dw_1x1(x)
        out = self.bn(out)
        return self.act(out)


class DySample(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(
            x, scale_factor=self.scale,
            mode='bilinear', align_corners=False
        )


class CoordAttention(nn.Module):
    """
    完整 Coordinate Attention
    """
    def __init__(self, channels, reduction=32):
        super().__init__()

        mid_channels = max(8, channels // reduction)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(
            channels, mid_channels, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(
            mid_channels, channels, kernel_size=1, bias=False
        )
        self.conv_w = nn.Conv2d(
            mid_channels, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        identity = x
        b, c, h, w = x.size()

        x_h = self.pool_h(x)                # (b,c,h,1)
        x_w = self.pool_w(x).permute(0,1,3,2)  # (b,c,w,1)

        y = torch.cat([x_h, x_w], dim=2)    # (b,c,h+w,1)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)

        x_w = x_w.permute(0,1,3,2)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w

        return out


class Rep_DEUCB(nn.Module):
    """
    Rep Depthwise + Upsample + Coordinate Attention Block
    """
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        self.up = DySample(scale=2)
        self.rep_dwc = RepDWConv(c1, kernel_size=k, stride=s)

        self.ca = CoordAttention(c1)

        self.pwc = nn.Conv2d(
            c1, c2, kernel_size=1,
            stride=1, padding=0, bias=True
        )

    def forward(self, x):
        x = self.up(x)
        x = self.rep_dwc(x)

        x = self.ca(x)

        x = self.pwc(x)

        return x
