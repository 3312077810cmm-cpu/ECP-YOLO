import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedAdd(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        # 两个可学习的权重参数
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = eps
        self.channels = channels

    def forward(self, x):
        # x 是 list: [feat_low, feat_high]
        w = F.relu(self.w)  # 保证非负
        w = w / (w.sum() + self.eps)  # 归一化
        return w[0] * x[0] + w[1] * x[1]
