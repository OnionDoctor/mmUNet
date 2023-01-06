# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     layers
 Version:       
 Description:
 Date:          2022/8/5
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# 双层卷积层
class DualConvLayer(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 residual=False):
        super(DualConvLayer, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_ch),
        )

        self.residual = residual
        if residual:
            self.res_layer = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=False)

        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.residual:
            res = self.res_layer(x)
            x = self.layer(x)
            x = x + res
            x = self.act_layer(x)
        else:
            x = self.layer(x)
            x = self.act_layer(x)
        return x


# 下采样: Size /2
class DownLayer(nn.Module):
    def __init__(self,
                 down_mode: str = 'pool',
                 in_ch: int = 64,
                 out_ch: int = 64):
        super(DownLayer, self).__init__()

        if down_mode == 'pool':
            if in_ch == out_ch:
                self.layer = nn.MaxPool2d(kernel_size=2)
            else:
                self.layer = nn.Sequential(
                    nn.MaxPool2d(kernel_size=2),
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
                )
        else:
            raise ValueError("Please input feasible down mode!")

    def forward(self, x):  # 输入为 NCHW
        x = self.layer(x)
        return x


# 上采样: Size *2
class UpLayer(nn.Module):
    def __init__(self,
                 up_mode: str = 'deconv',
                 in_ch: int = 64,   # in_ch 等于 out_ch 时不包含卷积
                 out_ch: int = 64):
        super(UpLayer, self).__init__()

        if up_mode == 'bilinear':
            if in_ch == out_ch:
                self.layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            else:
                self.layer = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)
                )
        elif up_mode == 'deconv':
            self.layer = nn.ConvTranspose2d(in_channels=in_ch, out_channels=out_ch, kernel_size=2, stride=2)
        else:
            raise ValueError("Please input feasible up mode!")

    def forward(self, x):  # 输入为 NCHW
        x = self.layer(x)
        return x

# 输入层 - 全卷积层
class InLayer(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch):
        super(InLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


# 输出层 - 全卷积层
class OutLayer(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch):
        super(OutLayer, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
