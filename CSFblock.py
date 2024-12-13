import math
import torch.nn as nn
# import basicblock as B
import torch


# import math
# import torch.nn.functional as F
# import torchvision
# import SKnet as SK
# import profile

class oneConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings, dilation=dilations,
                      bias=False),  ###, bias=False
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CSFblock(nn.Module):
    def __init__(self, in_channels, channels_1, strides):
        super().__init__()
        # self.layer = nn.Conv1d(in_channels, 512, kernel_size = 1, padding = 0, dilation = 1)
        self.Up = nn.Sequential(
            # nn.MaxPool2d(kernel_size = int(strides*2+1), stride = strides, padding = strides),
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=strides, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.Fgp = nn.AdaptiveAvgPool2d(1)
        self.layer1 = nn.Sequential(
            oneConv(in_channels, channels_1, 1, 0, 1),
            oneConv(channels_1, in_channels, 1, 0, 1),

        )
        self.SE1 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.SE2 = oneConv(in_channels, in_channels, 1, 0, 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_h, x_l):
        x1 = x_h
        x2 = self.Up(x_l)

        x_f = x1 + x2
        Fgp = self.Fgp(x_f)
        x_se = self.layer1(Fgp)
        x_se1 = self.SE1(x_se)
        x_se2 = self.SE2(x_se)
        x_se = torch.cat([x_se1, x_se2], 2)
        x_se = self.softmax(x_se)
        att_3 = torch.unsqueeze(x_se[:, :, 0], 2)
        att_5 = torch.unsqueeze(x_se[:, :, 1], 2)
        x1 = att_3 * x1
        x2 = att_5 * x2
        x_all = x1 + x2
        return x_all


if __name__ == '__main__':
    x1 = torch.randn(2, 32, 16, 16)
    x2 = torch.randn(2, 32, 8, 8)
    model = CSFblock(32, 32, 2)
    print(model(x1, x2).shape)
