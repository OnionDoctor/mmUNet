# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     LateMMUNet
 Version:       
 Description:
 Date:          2022/8/9
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


class InterMMUNet(nn.Module):
    def __init__(self,
                 in_ch1,
                 in_ch2,
                 num_cls,
                 residual,
                 down_mode: str = 'pool',
                 up_mode: str = 'deconv',
                 base: int = 32,
                 depth: int = 4):
        super(InterMMUNet, self).__init__()

        self.in_ch1 = in_ch1
        self.in_ch2 = in_ch2
        self.num_cls = num_cls
        self.residual = residual
        self.down_mode = down_mode
        self.up_mode = up_mode

        self.base = base
        self.depth = depth

        self.in_layer1 = InLayer(in_ch=in_ch1, out_ch=base)
        self.in_layer2 = InLayer(in_ch=in_ch2, out_ch=base)

        # 两个encoder
        self.enconv_layers1 = nn.ModuleList([DualConvLayer(in_ch=base * (2 ** i),
                                                           out_ch=base * (2 ** i),
                                                           residual=residual) for i in range(1, depth + 1)])
        self.down_layers1 = nn.ModuleList([DownLayer(down_mode='pool') for i in range(1, depth + 1)])

        self.enconv_layers2 = nn.ModuleList([DualConvLayer(in_ch=base * (2 ** (i - 1)),
                                                           out_ch=base * (2 ** i),
                                                           residual=residual) for i in range(1, depth + 1)])
        self.down_layers2 = nn.ModuleList([DownLayer(down_mode='pool') for i in range(1, depth + 1)])

        # 一个decoder 以模态一为主
        self.deconv_layers1 = nn.ModuleList([DualConvLayer(in_ch=base * (2 ** (i + 1)),
                                                           out_ch=base * (2 ** i),
                                                           residual=residual) for i in range(depth, 0, -1)])
        self.up_layers1 = nn.ModuleList([UpLayer(up_mode='deconv',
                                                 in_ch=base * (2 ** (i + 1)),
                                                 out_ch=base * (2 ** i)) for i in range(depth, 0, -1)])

        self.out_layer = OutLayer(in_ch=base * 2, out_ch=num_cls)

    def forward(self, rgb_x, t_x):

        x1 = self.in_layer1(rgb_x)
        x2 = self.in_layer2(t_x)

        x1 = torch.cat([x1, x2], dim=1)
        xs1 = [x1]
        for i in range(self.depth):
            x1 = self.down_layers1[i](x1)
            x1 = self.enconv_layers1[i](x1)

            x2 = self.down_layers2[i](x2)
            x2 = self.enconv_layers2[i](x2)

            x1 = torch.cat([x1, x2], dim=1)  # 在每一步都融合
            xs1.append(x1)

        for i in range(self.depth):
            x1 = self.up_layers1[i](x1)
            x1 = torch.cat([xs1[-i - 2], x1], dim=1)
            x1 = self.deconv_layers1[i](x1)

        x = self.out_layer(x1)

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
    model = InterMMUNet(in_ch1=3,
                        in_ch2=1,
                        num_cls=3,
                        residual=False,
                        down_mode='pool',
                        up_mode='deconv',
                        base=32,
                        depth=4)
    # 模型检查
    # print(model)

    # 模型运行检查
    # rgb_x = torch.randn(size=(2, 3, 480, 480))
    # t_x = torch.randn(size=(2, 1, 480, 480))
    # y = model(rgb_x, t_x)
    # print(y.shape)

    # 计算模型参数量
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), [(3, 480, 480), (1, 480, 480)])

