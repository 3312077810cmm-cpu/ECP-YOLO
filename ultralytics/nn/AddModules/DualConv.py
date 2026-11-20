import torch.nn as nn


class DualConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, g=2):
        super(DualConv, self).__init__()
        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        return self.gc(input_data) + self.pwc(input_data)