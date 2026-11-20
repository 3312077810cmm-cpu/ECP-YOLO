import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma=2, b=1, local_weight=0.5):
        super(MLCA, self).__init__()

        # -----------------------------
        # ECA参数计算
        # -----------------------------
        self.local_size = local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)
        k = t if t % 2 else t + 1  # 保证卷积核为奇数

        # -----------------------------
        # 局部与全局注意力
        # -----------------------------
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight = local_weight

        self.local_avg_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # -----------------------------
        # 局部与全局平均池化
        # -----------------------------
        local_avg = self.local_avg_pool(x)
        global_avg = self.global_avg_pool(x)

        b, c, m, n = x.shape
        b_local, c_local, m_local, n_local = local_avg.shape

        # -----------------------------
        # ECA变换：通道维卷积
        # -----------------------------
        temp_local = local_avg.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_avg.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)

        # -----------------------------
        # 维度还原
        # -----------------------------
        y_local_transpose = y_local.reshape(b, self.local_size * self.local_size, c).transpose(-1, -2).view(
            b, c, self.local_size, self.local_size
        )
        y_global_transpose = y_global.transpose(-1, -2).unsqueeze(-1)  # 修正了你的错误用法

        # -----------------------------
        # 注意力融合
        # -----------------------------
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(), [self.local_size, self.local_size])

        att_all = F.adaptive_avg_pool2d(
            att_global * (1 - self.local_weight) + att_local * self.local_weight, [m, n]
        )

        # -----------------------------
        # 加权输出
        # -----------------------------
        x = x * att_all
        return x
