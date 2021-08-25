# Initial reference https://github.com/yassouali/pytorch-segmentation/blob/master/models/unet.py
# Modified by Jordao Bragantini

import torch
import torch.nn as nn
import torch.nn.functional as F


def x2conv(in_channels, out_channels, inner_channels=None, stride=1):
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv3d(in_channels, inner_channels, kernel_size=3,
                  stride=stride, padding=1, bias=False),
        nn.BatchNorm3d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv, do_pooling, stride):
        super(encoder, self).__init__()
        self.down_conv = conv(in_channels, out_channels, stride=stride)
        if do_pooling:
            self.pool = nn.MaxPool3d(kernel_size=2, ceil_mode=True)
        else:
            self.pool = None

    def forward(self, x):
        x = self.down_conv(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class decoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv, do_convT):
        super(decoder, self).__init__()
        if do_convT:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = None
        self.up_conv = conv(in_channels, out_channels)

    def forward(self, x_copy, x, interpolate=True):
        if self.up is not None:
            x = self.up(x)

        if x.shape[-3:] != x_copy.shape[-3:]:
            if interpolate:
                # Iterpolating instead of padding
                x = F.interpolate(x, size=x_copy.shape[-3:],
                                  mode="trilinear", align_corners=True)
            else:
                # Padding in case the incomping volumes are of different sizes
                diff = x_copy.shape[-3:] - x.size[-3:]
                rpad = diff // 2
                lpad = diff - rpad
                x = F.pad(x, (lpad[2], rpad[2], lpad[1], rpad[1], lpad[0], rpad[0]))

        # Concatenate
        x = torch.cat([x_copy, x], dim=1)
        x = self.up_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, freeze_bn=False, depth=32, conv=x2conv,
                 do_pooling=True, do_convT=True, down_stride=1):
        super().__init__()

        self.start_conv = conv(in_channels, depth, stride=down_stride)
        self.down1 = encoder(depth, depth * 2, conv, do_pooling, down_stride)
        self.down2 = encoder(depth * 2, depth * 4, conv, do_pooling, down_stride)
        self.down3 = encoder(depth * 4, depth * 8, conv, do_pooling, down_stride)
        self.down4 = encoder(depth * 8, depth * 16, conv, do_pooling, down_stride)

        self.middle_conv = conv(depth * 16, depth * 16)

        self.up1 = decoder(depth * 16, depth * 8, conv, do_convT)
        self.up2 = decoder(depth * 8, depth * 4, conv, do_convT)
        self.up3 = decoder(depth * 4, depth * 2, conv, do_convT)
        self.up4 = decoder(depth * 2, depth, conv, do_convT)
        self.final_conv = nn.Conv3d(depth, num_classes, kernel_size=1)

        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm3d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, xf, xt1):
        x = torch.cat([xf, xt1], dim=1).float()
        x1 = self.start_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.middle_conv(self.down4(x4))

        x = self.up1(x4, x)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)

        x = self.final_conv(x)
        return x

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm3d):
                module.eval()

