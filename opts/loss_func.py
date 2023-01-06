# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     loss_func
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


# Dice 损失
class DiceLoss(nn.Module):
    def __init__(self, num_class=1, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.num_cls = num_class
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1).float()  # b x n x h x w
        targets = F.one_hot(targets, num_classes=self.num_cls)  # b x h x w x n 在最后一个维度扩展
        targets = targets.permute(0, 3, 1, 2).contiguous().float()  # b x n x h x w

        assert outputs.shape == targets.shape

        outputs = outputs.view(-1)
        targets = targets.view(-1)

        inter = (outputs * targets).sum()
        union = outputs.sum() + targets.sum()
        dice_loss = 1 - (2. * inter + self.epsilon) / (union + self.epsilon)

        return dice_loss


# IoU Loss
class IoULoss(nn.Module):
    def __init__(self, num_class=1, epsilon=1e-6):
        super(IoULoss, self).__init__()
        self.num_cls = num_class
        self.epsilon = epsilon

    def forward(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1).float()  # b x n x h x w
        targets = F.one_hot(targets, num_classes=self.num_cls)  # b x h x w x n 在最后一个维度扩展
        targets = targets.permute(0, 3, 1, 2).contiguous().float()  # b x n x h x w

        assert outputs.shape == targets.shape

        outputs = outputs.view(-1)
        targets = targets.view(-1)

        inter = (outputs * targets).sum()
        union = outputs.sum() + targets.sum() - inter
        iou_loss = 1 - (inter + self.epsilon) / (union + self.epsilon)

        return iou_loss

if __name__ == '__main__':
    outputs = torch.randn((2, 2, 256, 256))
    targets = torch.randint(0, 2, (2, 256, 256)).long()
    loss_func = DiceLoss(num_class=2)
    print(loss_func(outputs, targets))