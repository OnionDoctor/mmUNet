# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     EarlyMMUNet
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

from model.layers import DualConvLayer, DownLayer, UpLayer, OutLayer, InLayer


class EarlyMMUNet(nn.Module):
    def __init__(self,
                 in_ch1,
                 in_ch2,
                 num_cls,
                 residual,
                 down_mode: str = 'pool',
                 up_mode: str = 'deconv',
                 base: int = 64,
                 depth: int = 4):
        super(EarlyMMUNet, self).__init__()

        in_ch = in_ch1 + in_ch2
        self.in_ch = in_ch
        self.num_cls = num_cls
        self.residual = residual
        self.down_mode = down_mode
        self.up_mode = up_mode

        self.base = base
        self.depth = depth

        self.in_layer = InLayer(in_ch=in_ch, out_ch=base)

        self.enconv_layers = nn.ModuleList([DualConvLayer(in_ch=base * (2 ** (i - 1)),
                                                          out_ch=base * (2 ** i),
                                                          residual=residual) for i in range(1, depth + 1)])
        self.down_layers = nn.ModuleList([DownLayer(down_mode='pool') for i in range(1, depth + 1)])

        self.deconv_layers = nn.ModuleList([DualConvLayer(in_ch=base * (2 ** i),
                                                          out_ch=base * (2 ** (i - 1)),
                                                          residual=residual) for i in range(depth, 0, -1)])
        self.up_layers = nn.ModuleList([UpLayer(up_mode='deconv',
                                                in_ch=base * (2 ** i),
                                                out_ch=base * (2 ** (i - 1))) for i in range(depth, 0, -1)])

        self.out_layer = OutLayer(in_ch=base, out_ch=num_cls)

    def forward(self, rgb_x, t_x):

        x = torch.cat([rgb_x, t_x], dim=1)       # early fusion 在元数据上进行联结，增广 channels

        x = self.in_layer(x)
        xs = [x]
        for i in range(self.depth):
            x = self.down_layers[i](x)
            x = self.enconv_layers[i](x)
            xs.append(x)

        for i in range(self.depth):
            x = self.up_layers[i](x)
            x = torch.cat([xs[-i - 2], x], dim=1)
            x = self.deconv_layers[i](x)

        x = self.out_layer(x)

        return x

    # 初始化权重
    def initialize_weights(self):
        def init_func(layer):
            if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0.01)
            else:
                pass

        self.apply(init_func)

    # 载入权重
    def load_weights(self, weights):
        if type(weights) == str:
            weights = torch.load(weights)
        self.load_state_dict(weights['model'])


if __name__ == '__main__':
    model = EarlyMMUNet(in_ch1=3,
                        in_ch2=1,
                        num_cls=3,
                        residual=False,
                        down_mode='pool',
                        up_mode='deconv',
                        base=64,
                        depth=4)
    # 模型检查
    # print(model)

    # 模型运行检查
    # rgb_x = torch.randn(size=(2, 3, 480, 480))
    # t_x = torch.randn(size=(2, 1, 480, 480))
    # y = model(rgb_x, t_x)

    # 计算模型参数量
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), [(3, 480, 480), (1, 480, 480)])
