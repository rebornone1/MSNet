import os
import sys
import torch
import torch.nn as nn
import math

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

__all__ = ['resnet18', 'resnet50', 'resnet101']

model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
}


# 三乘三卷积 卷积核大小为3
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 这段代码定义了一个名为Bottleneck的类，它是一个继承自nn.Module的模块。Bottleneck模块通常用于构建深度残差网络（ResNet）中的残差块。
# 在__init__方法中，Bottleneck类定义了一系列卷积层和批归一化层。具体来说，它包括三个卷积层：conv1、conv2和conv3，以及对应的批归一化层：bn1、bn2和bn3。这些卷积层用于进行特征提取和变换。
# 在forward方法中，输入数据x通过卷积层和批归一化层进行前向传播。其中，residual变量用于保存输入数据x的残差连接。
# 首先，输入数据通过conv1、bn1和ReLU激活函数进行处理。然后，通过conv2、bn2和ReLU激活函数进行处理。
# 最后，通过conv3、bn3进行处理。如果存在下采样（downsample）操作，输入数据x也会通过下采样进行处理。最终，将残差连接和处理后的特征相加，并通过ReLU激活函数输出结果。
# Bottleneck模块的expansion属性表示输出通道数相对于输入通道数的倍数，通常为4。这是因为在ResNet中，Bottleneck模块通过使用1x1、3x3和1x1的卷积核来减少计算量，同时增加了特征通道的维度，以提高网络的表达能力。
# 总而言之，Bottleneck模块通过残差连接和卷积操作，实现了深度残差网络中的特征提取和变换。这有助于提高网络的性能和表示能力。
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Convkxk(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0):
        super(Convkxk, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResNet(nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.inplanes = 128
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # 这段代码定义了一个名为_make_layer的方法，用于构建残差块的层。
    # 首先，创建了一个变量downsample，并将其初始化为None。然后，通过判断条件stride != 1 or self.inplanes != planes * block.expansion来确定是否需要进行下采样操作。
    # 当步长不为1或输入通道数与输出通道数不匹配时，需要进行下采样。
    # 下采样操作通过一个包含卷积层和批归一化层的序列实现。具体来说，使用了一个1x1的卷积层nn.Conv2d，将输入通道数self.inplanes调整为输出通道数planes * block.expansion，并设置步长为stride。
    # 然后，接着使用了一个批归一化层nn.BatchNorm2d，用于归一化输出特征图。
    # 接下来，创建了一个空列表layers，用于存储残差块的层。首先，将第一个残差块添加到列表中。这个残差块的输入通道数为self.inplanes，输出通道数为planes，步长为stride，下采样操作为前面创建的downsample。
    # 然后，更新self.inplanes的值为planes * block.expansion，以便下一个残差块使用正确的输入通道数。
    # 接下来，通过循环构建剩余的残差块。循环从1开始，因为第一个残差块已经添加到列表中了。在每次循环中，将一个新的残差块添加到列表中，其输入通道数为self.inplanes，输出通道数为planes。
    # 最后，使用nn.Sequential将列表layers转化为一个序列模块，并作为方法的返回值。
    # 总而言之，_make_layer方法用于构建多个残差块的层。根据步长和输入通道数的变化情况，可能需要进行下采样操作。通过循环构建多个残差块，并将它们存储在一个序列模块中返回。这样可以方便地在ResNet模块中使用这些残差块
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        f = []
        x = self.layer1(x)
        f.append(x)
        x = self.layer2(x)
        f.append(x)
        x = self.layer3(x)
        f.append(x)
        x = self.layer4(x)
        f.append(x)

        return tuple(f)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        # return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet18']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on Places
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)
