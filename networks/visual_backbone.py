
import torch.nn as nn
from collections import OrderedDict


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, shortcut=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = shortcut
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, shortcut=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.shortcut = shortcut
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.shortcut is not None:
            identity = self.shortcut(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3, num_filters=None):
        super(ResNet, self).__init__()
        self.in_channels = 64
        if num_filters is None:
            num_filters = [64, 128, 256, 512]

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnet = nn.Sequential(OrderedDict([]))
        for block in range(len(layer_list)):
            name = f"layer{block}"
            self.resnet.add_module(name, self._make_layer(ResBlock, layer_list[block], planes=num_filters[block],
                                                          stride=1 if block == 0 else 2))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.resnet(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        shortcut = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes, shortcut=shortcut, stride=stride))
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], channels)


def ResNet101(channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], channels)


def ResNet152(channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], channels)


def ResNetCustomized(layers, channels=3):
    supported_layers = [18, 34, 50, 101, 152]
    assert layers in supported_layers, \
        "supported layers are {} but input layer is {}".format(
            supported_layers, layers)
    if layers == 18:
        depth = [2, 2, 2, 2]
    elif layers == 34 or layers == 50:
        depth = [3, 4, 6, 3]
    elif layers == 101:
        depth = [3]
    else:  # layers == 152:
        depth = [3, 8, 36, 3]
    num_filters = [64, 128, 256, 512]
    return ResNet(Bottleneck, depth, channels, num_filters)


if __name__ == '__main__':
    # state_dict = torch.load("/home/ysocr/data/cache/tmp.pth", map_location=torch.device("cpu"))
    # new_state_dict = {}
    # for key, value in state_dict.named_parameters():
    #     new_state_dict[key] = value
    model = ResNetCustomized(layers=101)
    print('debug')
