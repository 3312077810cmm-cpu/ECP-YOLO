import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv, autopad

class CondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, routing_weights):
        out = 0
        for i in range(self.num_experts):
            out += routing_weights[:, i].view(-1, 1, 1, 1) * \
                F.conv2d(x, self.weight[i], self.bias[i] if self.bias is not None else None,
                         self.stride, self.padding, self.dilation, self.groups)
        return out


class DynamicConv_Single(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding,
                                    dilation, groups, bias, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.softmax(self.routing(pooled_inputs), dim=1)
        return self.cond_conv(x, routing_weights)
