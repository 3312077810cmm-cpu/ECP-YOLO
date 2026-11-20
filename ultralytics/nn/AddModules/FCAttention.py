import torch
import torch.nn as nn
import math


def conv_block(in_features, out_features, kernel_size=(3, 3), padding=1, dilation=1,
               norm_type='bn', activation=True, groups=1):
    layers = []
    layers.append(nn.Conv2d(in_features, out_features, kernel_size=kernel_size,
                            stride=1, padding=padding, dilation=dilation, groups=groups, bias=False))
    if norm_type == 'bn':
        layers.append(nn.BatchNorm2d(out_features))
    if activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class MDCR(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 2, 4, 8]):
        super().__init__()

        self.block1 = conv_block(in_features // 4, out_features // 4,
                                 padding=rate[0], dilation=rate[0],
                                 norm_type=norm_type, activation=activation, groups=8)
        self.block2 = conv_block(in_features // 4, out_features // 4,
                                 padding=rate[1], dilation=rate[1],
                                 norm_type=norm_type, activation=activation, groups=8)
        self.block3 = conv_block(in_features // 4, out_features // 4,
                                 padding=rate[2], dilation=rate[2],
                                 norm_type=norm_type, activation=activation, groups=8)
        self.block4 = conv_block(in_features // 4, out_features // 4,
                                 padding=rate[3], dilation=rate[3],
                                 norm_type=norm_type, activation=activation, groups=8)

        self.out_s = conv_block(4, 4, kernel_size=(1, 1), padding=(0, 0),
                                norm_type=norm_type, activation=activation)
        self.out = conv_block(out_features, out_features, kernel_size=(1, 1),
                              padding=(0, 0), norm_type=norm_type, activation=activation)

    def forward(self, x):
        split_tensors = []
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x


class Mix(nn.Module):
    def forward(self, x1, x2):
        return (x1 + x2) / 2  # 简单加权平均，你可以换成 learnable fusion

class FCAttentionC(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(FCAttentionC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, bias=True)

        self.highpass = nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False)
        hp_kernel = torch.tensor([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.highpass.weight.copy_(hp_kernel.repeat(channel, 1, 1, 1))

        self.brightness_fc = nn.Sequential(
            nn.Linear(1, channel),
            nn.Sigmoid()
        )

        self.mix = Mix()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        avg = self.avg_pool(x)
        x1 = self.conv1(avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x2 = self.fc(avg)

        att1 = self.sigmoid(x1)
        att2 = self.sigmoid(x2)
        att = self.mix(att1, att2)

        high_freq = self.highpass(x)

        brightness = avg.mean(dim=1, keepdim=True).view(B, 1)
        brightness_weight = self.brightness_fc(1 - brightness).view(B, C, 1, 1)

        enhanced = x + brightness_weight * high_freq
        out = enhanced * att
        return out


class FCAttentionM(nn.Module):
    def __init__(self, in_ch, out_ch, rate=[1, 2, 4, 8], norm_type='bn', activation=True):
        super().__init__()
        self.mdcr = MDCR(in_ch, out_ch, norm_type=norm_type, activation=activation, rate=rate)
        self.fcatt = FCAttentionC(out_ch)

    def forward(self, x):
        x = self.mdcr(x)
        x = self.fcatt(x)
        return x


class ContextAggregation(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        self.a = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.k = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.v = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.m = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        a = torch.sigmoid(self.a(x))  # [N,1,H,W]
        k = torch.softmax(self.k(x).view(n, 1, -1, 1), dim=2)  # [N,1,HW,1]
        v = self.v(x).view(n, 1, self.inter_channels, -1)      # [N,1,C',HW]
        y = torch.matmul(v, k).view(n, self.inter_channels, 1, 1)
        y = self.m(y) * a
        return x + y


class FCAttention(nn.Module):
    def __init__(self, in_ch, out_ch, rate=[1, 2, 4, 8]):
        super().__init__()
        self.fca = FCAttentionM(in_ch, out_ch, rate=rate)
        self.context = ContextAggregation(in_ch, reduction=4)
        # 融合两个分支
        self.fuse = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.fca(x)        # 局部增强 + 多尺度
        out2 = self.context(x)    # 全局上下文
        out = torch.cat([out1, out2], dim=1)
        out = self.fuse(out)      # 通道融合
        return out


