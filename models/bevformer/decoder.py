import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3x3(nn.Module):
    def __init__(self, in_ch, out_ch, norm='gn', dropout=0.0, groups_gn=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=(norm is None))
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_ch)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(num_groups=min(groups_gn, out_ch), num_channels=out_ch)
        else:
            self.norm = nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.drop(self.act(self.norm(self.conv(x))))
    
    
class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor, size):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(size=size, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, sizes, residual=True, factor=2):
        super().__init__()

        layers = list()
        channels = dim

        for i, out_channels in enumerate(blocks):
            layer = DecoderBlock(channels, out_channels, dim, residual, factor, sizes[i])
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x):
        y = x

        for layer in self.layers:
            y = layer(y, x)

        return y