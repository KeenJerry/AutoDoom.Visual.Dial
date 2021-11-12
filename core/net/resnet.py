from typing import Optional
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck


class ResNetBackBone(nn.Module):
    def __init__(self, block: Optional[BasicBlock, Bottleneck], block_num_per_layer: []):
        super(ResNetBackBone, self).__init__()
        self.in_channel: int = 64
        # layer tools
        self.convd1: nn.Conv2d = nn.Conv2d(in_channels=3, out_channels=self.in_channel, kernel_size=(7, 7),
                                           stride=(2, 2), padding=3, bias=False)
        self.bn1: nn.BatchNorm2d = nn.BatchNorm2d(self.in_channel)
        self.relu: nn.ReLU = nn.ReLU(inplace=True)
        self.max_pool1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=3, stride=(2, 2), padding=1)

        self.layer1: nn.Sequential = self._make_layer(block, 64, block_num_per_layer[0], stride=(1, 1))
        self.layer2: nn.Sequential = self._make_layer(block, 128, block_num_per_layer[1], stride=(2, 2))
        self.layer3: nn.Sequential = self._make_layer(block, 256, block_num_per_layer[2], stride=(2, 2))
        self.layer4: nn.Sequential = self._make_layer(block, 512, block_num_per_layer[3], stride=(2, 2))

        # layer weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block: Optional[BasicBlock, Bottleneck], channel: int,
                    block_num: int, stride: tuple = (1, 1)) -> nn.Sequential:
        down_sample: Optional[nn.Sequential, None] = None
        if stride[0] != 1 or self.in_channel != channel * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layers = [block(self.in_channel, channel, downsample=down_sample, stride=stride)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.convd1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet_backbone_18():
    return ResNetBackBone(block=BasicBlock, block_num_per_layer=[2, 2, 2, 2])


def resnet_backbone_34():
    return ResNetBackBone(block=BasicBlock, block_num_per_layer=[3, 4, 6, 3])


def resnet_backbone_50():
    return ResNetBackBone(block=Bottleneck, block_num_per_layer=[3, 4, 6, 3])


def resnet_backbone_101():
    return ResNetBackBone(block=Bottleneck, block_num_per_layer=[3, 4, 23, 3])


def resnet_backbone_152():
    return ResNetBackBone(block=Bottleneck, block_num_per_layer=[3, 8, 36, 3])
