import torch
import torch.nn as nn

# GhostConv: 轻量化卷积，用较少的真实卷积生成特征，再通过 cheap operation 生成冗余特征
class GhostConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        c1: 输入通道数
        c2: 输出通道数
        k:  卷积核大小
        s:  步幅
        g:  分组卷积参数（一般为1）
        act: 是否使用激活函数（默认True）
        """
        super().__init__()
        # GhostNet原理：先用普通卷积生成少量特征，再用便宜的线性变换生成剩余特征
        c_ = c2 // 2  # 一半特征来自主卷积，一半来自cheap conv

        # 主分支：标准卷积
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, c_, k, s, k // 2, groups=g, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU() if act else nn.Identity()
        )

        # 辅助分支：cheap operation（使用 depthwise 3x3）
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(c_, c2 - c_, 3, 1, 1, groups=c_, bias=False),
            nn.BatchNorm2d(c2 - c_),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x):
        # 主分支特征
        y = self.primary_conv(x)
        # 辅助分支生成“ghost”特征
        y_ghost = self.cheap_operation(y)
        # 拼接输出
        return torch.cat([y, y_ghost], 1)

    def fuseforward(self, x):
        # 用于模型推理阶段（与 Conv 类似的接口）
        return self.forward(x)
