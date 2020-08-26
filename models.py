# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import layers


def get_encoder(dataset_type, img_type, dim_zs, dim_zc):
    if img_type == 'rgb':
        img_dim = 3
    elif img_type == 'grayscale':
        img_dim = 1
    elif img_type == 'sobel':
        img_dim = 2
    else:
        raise ValueError

    if dataset_type in ['MNIST', 'FashionMNIST']:
        return Encoder28(img_dim, dim_zs, dim_zc, return_act=True)
    elif dataset_type in ['CIFAR10']:
        return Encoder32(img_dim, 128, dim_zs, dim_zc, return_act=True, norm_type='bn', momentum=0.1)
    elif dataset_type in ['STL10', 'ImageNet10']:
        return Encoder96(img_dim, 64, dim_zs, dim_zc, return_act=True, norm_type='bn', momentum=0.1)
    else:
        raise NotImplementedError


def get_critic(dim_zs, dim_zc):
    return Critic(dim_zs, dim_zc)


def get_discriminator(dataset_type, dim_zs, dim_zc):
    if dataset_type in ['MNIST', 'FashionMNIST']:
        return Discriminator(5 * 5 * 128, dim_zs + dim_zc)
    elif dataset_type in ['CIFAR10']:
        return Discriminator(4 * 4 * 128 * 4, dim_zs + dim_zc)
    elif dataset_type in ['STL10', 'ImageNet10']:
        return Discriminator(6 * 6 * 64 * 8, dim_zs + dim_zc)
    else:
        raise NotImplementedError


class Encoder28(nn.Module):
    def __init__(self, img_dim, dim_zs=30, dim_zc=10, return_act=False):
        super(Encoder28, self).__init__()

        self.dim_zs = dim_zs
        self.return_act = return_act

        self.conv = nn.Sequential(
            nn.Conv2d(img_dim, 64, 4, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 128, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, dim_zs + dim_zc, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [b, img_dim, 28, 28]
        b = x.size(0)

        out = self.conv(x)
        out = out.view(b, -1)
        if self.return_act:
            # [b, 128 * 5 * 5]
            act = out

        out = self.fc(out)
        zs, zc_logit = out[:, :self.dim_zs], out[:, self.dim_zs:]
        if self.return_act:
            return zs, zc_logit, act
        else:
            return zs, zc_logit


class Encoder32(nn.Module):
    def __init__(self, img_dim, num_channels, dim_zs=30, dim_zc=10, return_act=False, norm_type='bn', **kwargs):
        super(Encoder32, self).__init__()

        self.dim_zs = dim_zs
        self.return_act = return_act

        # [32, 32] -> [16, 16]
        self.block1 = layers.OptimizedResBlockDown(img_dim, num_channels, norm_type=norm_type, **kwargs)
        # [16, 16] -> [8, 8]
        self.block2 = layers.ResBlock(num_channels, num_channels * 2, sample_type='down', norm_type=norm_type, **kwargs)
        # [8, 8] -> [4, 4]
        self.block3 = layers.ResBlock(num_channels * 2, num_channels * 4, sample_type='down', norm_type=norm_type,
                                      **kwargs)
        self.block4 = layers.ResBlock(num_channels * 4, num_channels * 4, sample_type='none', norm_type=norm_type,
                                      **kwargs)
        # [4, 4] -> [1, 1]
        self.conv5 = nn.Sequential(
            layers.Norm2dLayer(num_channels * 4, norm_type=norm_type, **kwargs),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=4),
            nn.Conv2d(num_channels * 4, dim_zs + dim_zc, kernel_size=1, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [b, img_dim, 32, 32]
        b = x.size(0)

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        if self.return_act:
            # [b, num_channels * 4 * 4 * 4]
            act = out.view(b, -1)
        out = self.block4(out)
        out = self.conv5(out)

        c = out.size(1)

        out = out.view(b, c)
        zs, zc_logit = out[:, :self.dim_zs], out[:, self.dim_zs:]
        if self.return_act:
            return zs, zc_logit, act
        else:
            return zs, zc_logit


class Encoder96(nn.Module):
    def __init__(self, img_dim, num_channels, dim_zs=30, dim_zc=10, return_act=False, norm_type='bn', **kwargs):
        super(Encoder96, self).__init__()

        self.dim_zs = dim_zs
        self.return_act = return_act

        # [96, 96] -> [48, 48]
        self.block1 = layers.OptimizedResBlockDown(img_dim, num_channels, norm_type=norm_type, **kwargs)
        # [48, 48] -> [24, 24]
        self.block2 = layers.ResBlock(num_channels, num_channels * 2, sample_type='down', norm_type=norm_type, **kwargs)
        # [24, 24] -> [12, 12]
        self.block3 = layers.ResBlock(num_channels * 2, num_channels * 4, sample_type='down', norm_type=norm_type,
                                      **kwargs)
        # [12, 12] -> [6, 6]
        self.block4 = layers.ResBlock(num_channels * 4, num_channels * 8, sample_type='down', norm_type=norm_type,
                                      **kwargs)
        self.block5 = layers.ResBlock(num_channels * 8, num_channels * 8, sample_type='none', norm_type=norm_type,
                                      **kwargs)
        # [6, 6] -> [1, 1]
        self.conv6 = nn.Sequential(
            layers.Norm2dLayer(num_channels * 8, norm_type=norm_type, **kwargs),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=6),
            nn.Conv2d(num_channels * 8, dim_zs + dim_zc, kernel_size=1, bias=True)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [b, img_dim, 96, 96]
        b = x.size(0)

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        if self.return_act:
            # [b, num_channels * 8 * 6 * 6]
            act = out.view(b, -1)
        out = self.block5(out)
        out = self.conv6(out)

        c = out.size(1)

        out = out.view(b, c)
        zs, zc_logit = out[:, :self.dim_zs], out[:, self.dim_zs:]
        if self.return_act:
            return zs, zc_logit, act
        else:
            return zs, zc_logit


class Critic(nn.Module):
    def __init__(self, dim_zs=30, dim_zc=10):
        super(Critic, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim_zs + dim_zc, 1024, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z):
        # z: [b, dim_zs + dim_zc]
        out = self.fc(z)
        return out


class Discriminator(nn.Module):
    def __init__(self, x_channels, z_channels):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(x_channels + z_channels, 1024, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1, bias=True),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, z):
        # x: [b, x_channels]
        # z: [b, z_channels]
        logit = self.net(torch.cat([x, z], dim=1))
        return logit
