# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from ultralytics.nn.modules.block import C3, C2f
#
# class CBAM(nn.Module):
#     def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
#         super(CBAM, self).__init__()
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
#             nn.Sigmoid()
#         )
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         ca = self.channel_attention(x)
#         x = x * ca
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
#         x = x * sa
#         return x
#
# class conv_block(nn.Module):
#     def __init__(self,
#                  in_features,
#                  out_features,
#                  kernel_size=(3, 3),
#                  stride=(1, 1),
#                  padding=(1, 1),
#                  dilation=(1, 1),
#                  norm_type='bn',
#                  activation=True,
#                  use_bias=True,
#                  groups=1
#                  ):
#         super().__init__()
#
#         self.conv = nn.Conv2d(in_channels=in_features,
#                               out_channels=out_features,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding,
#                               dilation=dilation,
#                               bias=use_bias,
#                               groups=groups)
#
#         self.norm_type = norm_type
#         self.act = activation
#
#         if self.norm_type == 'gn':
#             self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
#         if self.norm_type == 'bn':
#             self.norm = nn.BatchNorm2d(out_features)
#         if self.act:
#             # self.relu = nn.GELU()
#             self.relu = nn.ReLU(inplace=False)
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.norm_type is not None:
#             x = self.norm(x)
#
#         if self.act:
#             x = self.relu(x)
#         return x
#
#
# class LocalGlobalAttention(nn.Module):
#     def __init__(self, output_dim, patch_size):
#         super().__init__()
#         self.output_dim = output_dim
#         self.patch_size = patch_size
#         self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
#         self.norm = nn.LayerNorm(output_dim // 2)
#         self.mlp2 = nn.Linear(output_dim // 2, output_dim)
#         self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
#         self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
#         self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)
#
#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)
#         B, H, W, C = x.shape
#         P = self.patch_size
#
#         # Local branch
#         local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
#         local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
#         local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)
#
#         local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
#         local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
#         local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)
#
#         local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
#         local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)
#
#         cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
#         mask = cos_sim.clamp(0, 1)
#         local_out = local_out * mask
#         local_out = local_out @ self.top_down_transform
#
#         # Restore shapes
#         local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
#         local_out = local_out.permute(0, 3, 1, 2)
#         local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
#         output = self.conv(local_out)
#
#         return output
#
#
# class Bag(nn.Module):
#     def __init__(self):
#         super(Bag, self).__init__()
#
#     def forward(self, p, i, d):
#         edge_att = torch.sigmoid(d)
#         return edge_att * p + (1 - edge_att) * i
#
#
# class ECA(nn.Module):
#     def __init__(self, in_channel, gamma=2, b=1):
#         super(ECA, self).__init__()
#         k = int(abs((math.log(in_channel, 2) + b) / gamma))
#         kernel_size = k if k % 2 else k + 1
#         padding = kernel_size // 2
#         self.pool = nn.AdaptiveAvgPool2d(output_size=1)
#         self.conv = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         out = self.pool(x)
#         out = out.view(x.size(0), 1, x.size(1))
#         out = self.conv(out)
#         out = out.view(x.size(0), x.size(1), 1, 1)
#         return out * x
#
#
# class SpatialAttentionModule(nn.Module):
#     def __init__(self):
#         super(SpatialAttentionModule, self).__init__()
#         self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         return out * x
#
#
# class GatingModule(nn.Module):
#     def __init__(self, channels, branches=3, reduction=8):
#         super(GatingModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(channels, channels // reduction, 1, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channels // reduction, branches, 1, bias=True)
#         )
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, feats):  # feats: List of [B, C, H, W]
#         batch, c, h, w = feats[0].size()
#         feats_stack = torch.stack(feats, dim=1)  # shape: [B, branches, C, H, W]
#         context = self.avg_pool(torch.sum(feats_stack, dim=1))  # [B, C, 1, 1]
#         weights = self.mlp(context).view(batch, -1, 1, 1, 1)  # [B, branches, 1, 1, 1]
#         weights = self.softmax(weights)  # Softmax over branches
#         fused = torch.sum(weights * feats_stack, dim=1)  # Weighted sum
#         return fused
#
#
#
# class PPA(nn.Module):
#     def __init__(self, c1, c2, *args, **kwargs):  # 兼容 Ultralytics 会多传一个 c2 的情况
#         super().__init__()
#         # 跳连（对齐到 c2）
#         self.skip = conv_block(c1, c2, kernel_size=(1, 1), padding=(0, 0),
#                                norm_type='bn', activation=False)
#
#         # 基本卷积路径（全部输出 c2）
#         self.c1 = conv_block(c1, c2, kernel_size=(3, 3), padding=(1, 1),
#                              norm_type='bn', activation=True)
#         self.c2_ = conv_block(c2, c2, kernel_size=(3, 3), padding=(1, 1),
#                               norm_type='bn', activation=True)
#         self.c3 = conv_block(c2, c2, kernel_size=(3, 3), padding=(1, 1),
#                              norm_type='bn', activation=True)
#
#         # 多尺度聚合（ASPP-lite）
#         self.msf = nn.ModuleList([
#             nn.Conv2d(c2, c2, kernel_size=3, padding=1, dilation=1, bias=False),
#             nn.Conv2d(c2, c2, kernel_size=3, padding=2, dilation=2, bias=False),
#             nn.Conv2d(c2, c2, kernel_size=3, padding=3, dilation=3, bias=False),
#         ])
#         self.fuse = nn.Conv2d(c2 * 3, c2, kernel_size=1)
#
#         # 注意力
#         self.cbam = CBAM(c2)
#         self.lga2 = LocalGlobalAttention(c2, 2)
#         self.lga4 = LocalGlobalAttention(c2, 4)
#
#         # 非可学习 Sobel 边缘引导（深度可分离：groups=c2）
#         self.edge_conv = nn.Conv2d(c2, c2, kernel_size=3, padding=1,
#                                    bias=False, groups=c2)
#         sobel_kernel = torch.tensor([[-1, -2, -1],
#                                      [ 0,  0,  0],
#                                      [ 1,  2,  1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#         sobel_kernel = sobel_kernel.repeat(c2, 1, 1, 1)  # [c2, 1, 3, 3]
#         with torch.no_grad():
#             self.edge_conv.weight.copy_(sobel_kernel)
#         self.edge_conv.weight.requires_grad = False  # 冻结
#
#         # 可学习边缘增强（逐通道）
#         self.learnable_edge = nn.Sequential(
#             nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.Sigmoid()
#         )
#
#         # 正则与归一化
#         self.drop = nn.Dropout2d(0.1)
#         self.bn1 = nn.BatchNorm2d(c2)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         # 跳连对齐
#         x_skip = self.skip(x)
#
#         # 边缘分支
#         x_edge_sobel = self.edge_conv(x_skip)
#         x_edge_learned = self.learnable_edge(x_skip)
#         x_edge = x_edge_sobel + x_edge_learned  # 融合边缘
#
#         # 局部-全局注意力
#         x_lga2 = self.lga2(x_skip)
#         x_lga4 = self.lga4(x_skip)
#
#         # 主干卷积
#         x1 = self.c1(x)
#         x2 = self.c2_(x1)
#         x3 = self.c3(x2)
#
#         # 多尺度特征聚合
#         ms1 = self.msf[0](x3)
#         ms2 = self.msf[1](x3)
#         ms3 = self.msf[2](x3)
#         x_msf = torch.cat([ms1, ms2, ms3], dim=1)
#         x_msf = self.fuse(x_msf)
#
#         # 融合
#         x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4 + x_edge + x_msf
#
#         # 注意力 + 正则
#         x = self.cbam(x)
#         x = self.drop(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         return x
# #下面是最初的版本
# # class DASI(nn.Module):
# #     def __init__(self, in_features, out_features):
# #         super().__init__()
# #         self.bag = Bag()
# #
# #         # 原始 shortcut
# #         self.skip1 = conv_block(in_features, out_features, kernel_size=(1, 1), padding=(0, 0), norm_type='bn', activation=False)
# #
# #         # 加入 PPA 特征增强模块
# #         self.ppa = PPA(in_features, out_features)
# #
# #         # 边缘特征提取，用于 Bag 模块
# #         self.edge_extract = nn.Sequential(
# #             conv_block(in_features, out_features, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True),
# #             conv_block(out_features, out_features, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True),
# #             nn.Conv2d(out_features, out_features, kernel_size=1)
# #         )
# #
# #         self.tail_conv = conv_block(out_features, out_features, kernel_size=(1, 1), padding=(0, 0), norm_type='bn', activation=False)
# #         self.relu = nn.ReLU()
# #         self.bn = nn.BatchNorm2d(out_features)
#
# class ECABlock(nn.Module):
#     def __init__(self, in_ch, k_size=3):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         # B,C,H,W
#         y = self.avg_pool(x)                     # B,C,1,1
#         y = self.conv(y.squeeze(-1).transpose(-1, -2))   # B,C,1
#         y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
#         return x * y.expand_as(x)
#
# class DASI(nn.Module):
#     def __init__(self, in_features, out_features):
#         super().__init__()
#         self.bag = Bag()
#
#         # shortcut 分支
#         self.skip1 = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type='bn', activation=False)
#
#         # ECA 分支 + 通道对齐
#         self.eca = ECABlock(in_features)
#         self.eca_proj = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type='bn', activation=False)
#
#         # 边缘分支
#         self.edge_extract = nn.Sequential(
#             conv_block(in_features, out_features, kernel_size=3, padding=1, norm_type='bn', activation=True),
#             conv_block(out_features, out_features, kernel_size=3, padding=1, norm_type='bn', activation=True),
#             nn.Conv2d(out_features, out_features, kernel_size=1)
#         )
#
#         # 输出调整
#         self.tail_conv = conv_block(out_features, out_features, kernel_size=1, padding=0, norm_type='bn', activation=False)
#         self.relu = nn.ReLU()
#         self.bn = nn.BatchNorm2d(out_features)
#
#     def forward(self, x):
#         # ECA增强 + 通道对齐
#         x_eca = self.eca(x)
#         x_eca = self.eca_proj(x_eca)
#
#         # 边缘分支
#         x_edge = self.edge_extract(x)
#
#         # shortcut
#         x_skip = self.skip1(x)
#
#         # Bag 融合
#         out = self.bag(x_eca, x_skip, x_edge)
#
#         # 后处理
#         out = self.tail_conv(out)
#         out = self.bn(out)
#         out = self.relu(out)
#         return out
#
#
#
#
# class MDCR(nn.Module):
#     def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 2, 4, 8]):
#         super().__init__()
#
#         self.block1 = conv_block(
#             in_features=in_features // 4,
#             out_features=out_features // 4,
#             padding=rate[0],
#             dilation=rate[0],
#             norm_type=norm_type,
#             activation=activation,
#             groups=8
#         )
#         self.block2 = conv_block(
#             in_features=in_features // 4,
#             out_features=out_features // 4,
#             padding=rate[1],
#             dilation=rate[1],
#             norm_type=norm_type,
#             activation=activation,
#             groups=8
#         )
#         self.block3 = conv_block(
#             in_features=in_features // 4,
#             out_features=out_features // 4,
#             padding=rate[2],
#             dilation=rate[2],
#             norm_type=norm_type,
#             activation=activation,
#             groups=8
#         )
#         self.block4 = conv_block(
#             in_features=in_features // 4,
#             out_features=out_features // 4,
#             padding=rate[3],
#             dilation=rate[3],
#             norm_type=norm_type,
#             activation=activation,
#             groups=8
#         )
#         self.out_s = conv_block(
#             in_features=4,
#             out_features=4,
#             kernel_size=(1, 1),
#             padding=(0, 0),
#             norm_type=norm_type,
#             activation=activation,
#         )
#         self.out = conv_block(
#             in_features=out_features,
#             out_features=out_features,
#             kernel_size=(1, 1),
#             padding=(0, 0),
#             norm_type=norm_type,
#             activation=activation,
#         )
#
#     def forward(self, x):
#         split_tensors = []
#         x = torch.chunk(x, 4, dim=1)
#         x1 = self.block1(x[0])
#         x2 = self.block2(x[1])
#         x3 = self.block3(x[2])
#         x4 = self.block4(x[3])
#         for channel in range(x1.size(1)):
#             channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
#             concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
#             split_tensors.append(concatenated_channel)
#         x = torch.cat(split_tensors, dim=1)
#         x = self.out(x)
#         return x

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C3, C2f

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)

        if self.act:
            x = self.relu(x)
        return x


class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size * patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P * P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention  # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output


class Bag(nn.Module):
    def __init__(self):
        super(Bag, self).__init__()

    def forward(self, p, i, d):
        edge_att = torch.sigmoid(d)
        return edge_att * p + (1 - edge_att) * i


class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class GatingModule(nn.Module):
    def __init__(self, channels, branches=3, reduction=8):
        super(GatingModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, branches, 1, bias=True)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats):  # feats: List of [B, C, H, W]
        batch, c, h, w = feats[0].size()
        feats_stack = torch.stack(feats, dim=1)  # shape: [B, branches, C, H, W]
        context = self.avg_pool(torch.sum(feats_stack, dim=1))  # [B, C, 1, 1]
        weights = self.mlp(context).view(batch, -1, 1, 1, 1)  # [B, branches, 1, 1, 1]
        weights = self.softmax(weights)  # Softmax over branches
        fused = torch.sum(weights * feats_stack, dim=1)  # Weighted sum
        return fused



class PPA(nn.Module):
    def __init__(self, in_features, filters):
        super().__init__()

        # 跳连
        self.skip = conv_block(in_features=in_features, out_features=filters,
                               kernel_size=(1, 1), padding=(0, 0), norm_type='bn', activation=False)

        # 基本卷积路径
        self.c1 = conv_block(in_features=in_features, out_features=filters,
                             kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)
        self.c2 = conv_block(in_features=filters, out_features=filters,
                             kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)
        self.c3 = conv_block(in_features=filters, out_features=filters,
                             kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True)

        # ✅ 多尺度聚合模块（ASPP-lite）
        self.msf = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.Conv2d(filters, filters, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.Conv2d(filters, filters, kernel_size=3, padding=3, dilation=3, bias=False),
        )
        self.fuse = nn.Conv2d(filters * 3, filters, kernel_size=1)

        # ✅ CBAM注意力
        self.cbam = CBAM(filters)

        # ✅ Local-Global 注意力
        self.lga2 = LocalGlobalAttention(filters, 2)
        self.lga4 = LocalGlobalAttention(filters, 4)

        # ✅ 非可学习 Sobel 边缘引导
        self.edge_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1,
                                   bias=False, groups=filters)
        sobel_kernel = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel = sobel_kernel.repeat(filters, 1, 1, 1)
        self.edge_conv.weight = nn.Parameter(sobel_kernel, requires_grad=False)

        # ✅ 可学习边缘增强分支（简单版本）
        self.learnable_edge = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters, bias=False),
            nn.BatchNorm2d(filters),
            nn.Sigmoid()
        )

        self.drop = nn.Dropout2d(0.1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_skip = self.skip(x)

        # Sobel 边缘图
        x_edge_sobel = self.edge_conv(x_skip)

        # 可学习边缘图
        x_edge_learned = self.learnable_edge(x_skip)
        x_edge = x_edge_sobel + x_edge_learned  # 融合边缘

        # Local-Global Attention
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)

        # 主干卷积流程
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)

        # 多尺度特征聚合
        ms1 = self.msf[0](x3)
        ms2 = self.msf[1](x3)
        ms3 = self.msf[2](x3)
        x_msf = torch.cat([ms1, ms2, ms3], dim=1)
        x_msf = self.fuse(x_msf)

        # 特征融合
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4 + x_edge + x_msf

        # 注意力 + 正则
        x = self.cbam(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


class DASI(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bag = Bag()

        # 原始 shortcut
        self.skip1 = conv_block(in_features, out_features, kernel_size=(1, 1), padding=(0, 0), norm_type='bn', activation=False)

        # 加入 PPA 特征增强模块
        self.ppa = PPA(in_features, out_features)

        # 边缘特征提取，用于 Bag 模块
        self.edge_extract = nn.Sequential(
            conv_block(in_features, out_features, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True),
            conv_block(out_features, out_features, kernel_size=(3, 3), padding=(1, 1), norm_type='bn', activation=True),
            nn.Conv2d(out_features, out_features, kernel_size=1)
        )

        self.tail_conv = conv_block(out_features, out_features, kernel_size=(1, 1), padding=(0, 0), norm_type='bn', activation=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        # 主干分支：PPA增强（多尺度注意）
        x_ppa = self.ppa(x)

        # 边缘感知分支
        x_edge = self.edge_extract(x)

        # 残差连接
        x_skip = self.skip1(x)

        # 使用 Bag 结构融合增强分支、残差和边缘信息
        out = self.bag(x_ppa, x_skip, x_edge)

        # 后处理
        out = self.tail_conv(out)
        out = self.bn(out)
        out = self.relu(out)

        return out




class MDCR(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 2, 4, 8]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features // 4,
            out_features=out_features // 4,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups=8
        )
        self.block2 = conv_block(
            in_features=in_features // 4,
            out_features=out_features // 4,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=8
        )
        self.block3 = conv_block(
            in_features=in_features // 4,
            out_features=out_features // 4,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups=8
        )
        self.block4 = conv_block(
            in_features=in_features // 4,
            out_features=out_features // 4,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups=8
        )
        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )

    def forward(self, x):
        split_tensors = []
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x



class ECABlock(nn.Module):
    def __init__(self, in_ch, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # B,C,H,W
        y = self.avg_pool(x)                     # B,C,1,1
        y = self.conv(y.squeeze(-1).transpose(-1, -2))   # B,C,1
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)

class DASI_Lite(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bag = Bag()

        # shortcut 分支
        self.skip1 = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type='bn', activation=False)

        # ECA 分支 + 通道对齐
        self.eca = ECABlock(in_features)
        self.eca_proj = conv_block(in_features, out_features, kernel_size=1, padding=0, norm_type='bn', activation=False)

        # 边缘分支
        self.edge_extract = nn.Sequential(
            conv_block(in_features, out_features, kernel_size=3, padding=1, norm_type='bn', activation=True),
            conv_block(out_features, out_features, kernel_size=3, padding=1, norm_type='bn', activation=True),
            nn.Conv2d(out_features, out_features, kernel_size=1)
        )

        # 输出调整
        self.tail_conv = conv_block(out_features, out_features, kernel_size=1, padding=0, norm_type='bn', activation=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        # ECA增强 + 通道对齐
        x_eca = self.eca(x)
        x_eca = self.eca_proj(x_eca)

        # 边缘分支
        x_edge = self.edge_extract(x)

        # shortcut
        x_skip = self.skip1(x)

        # Bag 融合
        out = self.bag(x_eca, x_skip, x_edge)

        # 后处理
        out = self.tail_conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out

