import torch
from torch import nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

# class SDI(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.convs = nn.ModuleList(
#             [nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels]
#         )
#
#     def forward(self, xs):
#         ans = torch.ones_like(xs[0])
#         target_size = xs[0].shape[2:]
#         for i, x in enumerate(xs):
#             current_size = x.shape[2:]
#             if current_size != target_size:
#                 # 计算池化尺寸以匹配目标尺寸
#                 kernel_size = (
#                     int(current_size[0] / target_size[0]),
#                     int(current_size[1] / target_size[1])
#                 )
#                 # 确保 kernel_size 是整数且至少为 1
#                 kernel_size = tuple(max(1, k) for k in kernel_size)
#                 x = F.avg_pool2d(x, kernel_size=kernel_size)
#             ans = ans * self.convs[i](x)
#         return ans


class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 创建多个卷积层，每个分支对应一个卷积
        # 输入通道数取自channels列表，输出通道数统一为channels[0]
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=channel,
                      out_channels=channels[0],
                      kernel_size=3,
                      stride=1,
                      padding=1)
            for channel in channels
        ])

    def forward(self, xs):
        # 初始化输出为全1张量（作为乘法初始值）
        ans = torch.ones_like(xs[0], device=xs[0].device)

        # 获取目标尺寸（以第一个输入特征图的空间尺寸为基准）
        target_size = xs[0].shape[2:]  # (H, W)

        for i, x in enumerate(xs):
            current_size = x.shape[2:]  # 当前特征图的空间尺寸

            # 尺寸调整：当空间尺寸不匹配时进行平均池化
            if current_size != target_size:
                # 计算池化核大小（自动向下取整）
                kernel_size = (
                    int(current_size[0] / target_size[0]),
                    int(current_size[1] / target_size[1])
                )
                # 确保核大小至少为1x1
                kernel_size = tuple(max(1, k) for k in kernel_size)
                x = F.avg_pool2d(x, kernel_size=kernel_size)

            # 应用对应的卷积层并逐元素相乘
            ans = ans * self.convs[i](x)

        return ans