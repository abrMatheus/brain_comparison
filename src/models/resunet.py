
# Reference https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# Modified by Jordao Bragantini

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Callable, Optional

from functools import partial

from .unet import UNet, decoder, encoder


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 2

    def __init__(
        self,
        inplanes: int,
        out_planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        width = int(out_planes / self.expansion * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_planes)
        self.bn3 = norm_layer(out_planes)
        self.relu = nn.ReLU(inplace=True)
        if downsample is not None:
            self.downsample = downsample(inplanes, out_planes)
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResUNet(UNet):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, depth=32):
        super(UNet, self).__init__()

        down_conv = partial(Bottleneck, downsample=partial(conv1x1, stride=2))
        up_conv = partial(Bottleneck, downsample=partial(conv1x1, stride=1))
 
        self.start_conv = down_conv(in_channels, depth, stride=2)
        self.down1 = encoder(depth, depth * 2, down_conv, False, 2)
        self.down2 = encoder(depth * 2, depth * 4, down_conv, False, 2)
        self.down3 = encoder(depth * 4, depth * 8, down_conv, False, 2)
        self.down4 = encoder(depth * 8, depth * 16, down_conv, False, 2)

        self.middle_conv = up_conv(depth * 16, depth * 16)

        self.up1 = decoder(depth * 24, depth * 8, up_conv, False)
        self.up2 = decoder(depth * 12, depth * 4, up_conv, False)
        self.up3 = decoder(depth * 6, depth * 2, up_conv, False)
        self.up4 = decoder(depth * 3, depth, up_conv, False)
        self.final_conv = nn.Conv3d(depth, num_classes, kernel_size=1)

        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(module.weight, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
