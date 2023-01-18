from torch import nn
import torch
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class DWSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWSConv, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, k=3, s=1, g=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, k=1, s=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.swish = Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.swish(x)
        return x