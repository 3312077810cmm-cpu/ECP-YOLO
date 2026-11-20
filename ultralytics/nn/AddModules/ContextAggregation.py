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










# import torch
# from torch import nn
#
# from mmengine.model import caffe2_xavier_init, constant_init
# from ultralytics.nn.modules.conv import Conv
#
# def build_activation(act_cfg):
#     """根据 act_cfg 构建激活层。
#     act_cfg 可以是:
#       - None: 不使用激活
#       - 字符串: 如 'ReLU'、'SiLU'
#       - 字典: 例如 {'type': 'LeakyReLU', 'negative_slope': 0.1, 'inplace': True}
#     """
#     if act_cfg is None:
#         return None
#     if isinstance(act_cfg, str):
#         act_type = act_cfg
#         kwargs = {}
#     elif isinstance(act_cfg, dict):
#         act_cfg = act_cfg.copy()
#         act_type = act_cfg.pop('type', 'ReLU')
#         kwargs = act_cfg
#     else:
#         raise TypeError(f'Unsupported act_cfg type: {type(act_cfg)}')
#
#     mapping = {
#         'ReLU': nn.ReLU,
#         'LeakyReLU': nn.LeakyReLU,
#         'SiLU': nn.SiLU,
#         'GELU': nn.GELU,
#         'PReLU': nn.PReLU,
#         'ELU': nn.ELU,
#         'Tanh': nn.Tanh,
#         'Sigmoid': nn.Sigmoid,
#         'Identity': nn.Identity,
#     }
#     if act_type not in mapping:
#         raise KeyError(f'Unsupported activation type: {act_type}')
#     return mapping[act_type](**kwargs)
#
# class ConvModule(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size=1,
#                  stride=1,
#                  padding=None,
#                  dilation=1,
#                  groups=1,
#                  bias=True,
#                  # 兼容旧参数
#                  act=True,
#                  # 新参数：与 mmcv 风格一致
#                  act_cfg=None,
#                  norm_cfg=None):
#         """
#         - act_cfg 优先级高于 act。
#         - norm_cfg 预留，当前不构建归一化层（需要的话可按需扩展）。
#         """
#         super().__init__()
#
#         # auto padding（同感受野），除非手动传入
#         if padding is None:
#             if isinstance(kernel_size, int):
#                 padding = (kernel_size - 1) // 2
#             else:
#                 # tuple/list 时由调用者自行控制
#                 padding = 0
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
#                               stride=stride, padding=padding, dilation=dilation,
#                               groups=groups, bias=bias)
#
#         # 归一化占位（如需可扩展 build_norm）
#         self.norm = None
#         if norm_cfg is not None:
#             # 简单示例：只支持 BatchNorm2d
#             # norm_cfg 可为 {'type':'BN', 'eps':1e-5, 'momentum':0.1}
#             _eps = norm_cfg.get('eps', 1e-5) if isinstance(norm_cfg, dict) else 1e-5
#             _momentum = norm_cfg.get('momentum', 0.1) if isinstance(norm_cfg, dict) else 0.1
#             self.norm = nn.BatchNorm2d(out_channels, eps=_eps, momentum=_momentum)
#
#         # 激活层构建：优先使用 act_cfg，否则回退到 act=True -> ReLU
#         if act_cfg is None:
#             act_cfg = {'type': 'ReLU', 'inplace': True} if act else None
#         self.act = build_activation(act_cfg)
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.act is not None:
#             x = self.act(x)
#         return x
# class ContextAggregation(nn.Module):
#     def __init__(self, in_channels, reduction=1):
#         super(ContextAggregation, self).__init__()
#         self.in_channels = in_channels
#         self.reduction = reduction
#         self.inter_channels = max(in_channels // reduction, 1)
#         conv_params = dict(kernel_size=1, act_cfg=None)
#         self.a = ConvModule(in_channels, 1, **conv_params)
#         self.k = ConvModule(in_channels, 1, **conv_params)
#         self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
#         self.m = ConvModule(self.inter_channels, in_channels, **conv_params)
#         self.init_weights()
#
#     def init_weights(self):
#         for m in (self.a, self.k, self.v):
#             caffe2_xavier_init(m.conv)
#         constant_init(self.m.conv, 0)
#
#     def forward(self, x):
#         n = x.size(0)
#         c = self.inter_channels
#         a = self.a(x).sigmoid()
#         k = self.k(x).view(n, 1, -1, 1).softmax(2)#
#         v = self.v(x).view(n, 1, c, -1)
#         y = torch.matmul(v, k).view(n, c, 1, 1)
#         y = self.m(y) * a
#         return x + y
#
#
# # class ContextAggregationV2(nn.Module):
# #     def __init__(self, in_channels, reduction=4, num_heads=1):
# #         super().__init__()
# #         self.in_channels = in_channels
# #         self.inter_channels = max(in_channels // reduction, 1)
# #         self.num_heads = num_heads
# #
# #         self.a = ConvModule(in_channels, 1, kernel_size=1, act_cfg=None)   # 空间门控
# #         self.k = ConvModule(in_channels, num_heads, kernel_size=1, act_cfg=None)  # 多头空间注意力
# #         self.v = ConvModule(in_channels, self.inter_channels, kernel_size=1, act_cfg=None)  # 值投影
# #         self.m = ConvModule(self.inter_channels, in_channels, kernel_size=1, act_cfg=None)
# #
# #         self.gamma = nn.Parameter(torch.zeros(1))  # 控制残差比例
# #
# #     def forward(self, x):
# #         n, _, h, w = x.shape
# #         c = self.inter_channels
# #         s = h * w
# #
# #         a = self.a(x).sigmoid()                   # (N,1,H,W)
# #         k = self.k(x).view(n, self.num_heads, -1, 1).softmax(2)  # (N,Hs,S,1)
# #         v = self.v(x).view(n, 1, c, s)            # (N,1,C,S)
# #
# #         y = torch.matmul(v, k).view(n, c, 1, 1)   # (N,C,1,1)
# #         y = self.m(y) * a                         # (N,C,H,W)
# #         return x + self.gamma * y
