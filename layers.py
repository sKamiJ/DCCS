# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Norm2dLayer(nn.Module):
    def __init__(self, channels, norm_type='bn', **kwargs):
        super(Norm2dLayer, self).__init__()

        assert norm_type in ['none', 'bn', 'ln', 'in', 'gn']

        if norm_type == 'bn':
            momentum = kwargs.get('momentum', 0.1)
            self.norm = nn.BatchNorm2d(channels, momentum=momentum)
        elif norm_type == 'ln':
            self.norm = nn.GroupNorm(1, channels)
        elif norm_type == 'in':
            self.norm = nn.GroupNorm(channels, channels)
        elif norm_type == 'gn':
            num_groups = kwargs.get('num_groups', 8)
            self.norm = nn.GroupNorm(num_groups, channels)
        else:
            self.norm = EmptyLayer()

        self.reset_parameters()

    def reset_parameters(self):
        if not isinstance(self.norm, EmptyLayer):
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

    def forward(self, x):
        return self.norm(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sample_type='none', norm_type='none', **kwargs):
        super(ResBlock, self).__init__()

        assert sample_type in ['none', 'up', 'down']

        bias = norm_type != 'bn'

        self.block = nn.Sequential(
            Norm2dLayer(in_channels, norm_type=norm_type, **kwargs),
            nn.ReLU(inplace=True),
            nn.UpsamplingNearest2d(scale_factor=2) if sample_type == 'up' else EmptyLayer(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            Norm2dLayer(out_channels, norm_type=norm_type, **kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.AvgPool2d(kernel_size=2, stride=2) if sample_type == 'down' else EmptyLayer()
        )

        if sample_type != 'none' or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2) if sample_type == 'up' else EmptyLayer(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
                nn.AvgPool2d(kernel_size=2, stride=2) if sample_type == 'down' else EmptyLayer(),
            )
        else:
            self.shortcut = None

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.block(x)
        if self.shortcut is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return out + shortcut


class OptimizedResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type='none', **kwargs):
        super(OptimizedResBlockDown, self).__init__()

        bias = norm_type != 'bn'

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            Norm2dLayer(out_channels, norm_type=norm_type, **kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.block(x)
        shortcut = self.shortcut(x)

        return out + shortcut


class SobelLayer(nn.Module):
    def __init__(self, normalize=False):
        super(SobelLayer, self).__init__()

        self.sobel = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)

        self.sobel.weight.requires_grad_(False)
        self.sobel.weight.copy_(torch.tensor([
            [[
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ]],
            [[
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ]]], dtype=torch.float32))
        if normalize:
            self.sobel.weight /= 8

    def forward(self, x):
        return self.sobel(x)
