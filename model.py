from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DMR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv_large = nn.Conv2d(dim, dim, 9, stride=1, padding=20, groups=dim, dilation=5)

        self.routing = nn.Sequential(
            nn.Conv2d(3 * dim, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn3 = self.conv_large(attn1)

        attn_cat = torch.cat([attn1, attn2, attn3], dim=1)

        route = self.routing(attn_cat)
        weight1 = route[:, 0:1, :, :]
        weight2 = route[:, 1:2, :, :]
        weight3 = route[:, 2:3, :, :]

        attn = attn1 * weight1 + attn2 * weight2 + attn3 * weight3

        return x * attn


class LRF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv(x)


class _DenseLayer(nn.Module):
    def __init__(self, input_c: int, growth_rate: int, bn_size: int, drop_rate: float):
        super(_DenseLayer, self).__init__()

        self.add_module("norm1", nn.BatchNorm2d(input_c))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1",
                        nn.Conv2d(in_channels=input_c, out_channels=bn_size * growth_rate, kernel_size=1, stride=1,
                                  bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))

        self.add_module("conv3x3",
                        nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module("conv5x5",
                        nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=5, stride=1, padding=2, bias=False))

        self.drop_rate = drop_rate

    def forward(self, inputs: Tensor):
        concat_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concat_features)))
        bottleneck_output = self.relu2(self.norm2(bottleneck_output))

        new_features = self.conv3x3(bottleneck_output) + self.conv5x5(bottleneck_output)

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features


class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers: int, input_c: int, bn_size: int, growth_rate: int, drop_rate: float):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_c + i * growth_rate, growth_rate=growth_rate, bn_size=bn_size,
                                drop_rate=drop_rate)
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self,
                 input_c: int,
                 output_c: int):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(input_c))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(input_c, output_c, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self,
                 growth_rate: int = 32,
                 block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0,
                 num_classes: int = 1000):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Sequential(
                nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(num_init_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
            )),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                input_c=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                refine = LRF(num_features)
                self.features.add_module("refine%d" % (i + 1), refine)

                trans = _Transition(input_c=num_features,
                                    output_c=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        self.lsk_att = DMR(num_features)

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor):
        features = self.features(x)
        features = self.lsk_att(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121(num_classes):
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 24, 16),
                    num_init_features=64,
                    num_classes=num_classes)


def densenet169(num_classes):
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 32, 32),
                    num_init_features=64,
                    num_classes=num_classes)


def densenet201(num_classes):
    return DenseNet(growth_rate=32,
                    block_config=(6, 12, 48, 32),
                    num_init_features=64,
                    num_classes=num_classes)


def densenet161(num_classes):
    return DenseNet(growth_rate=48,
                    block_config=(6, 12, 36, 24),
                    num_init_features=96,
                    num_classes=num_classes)
