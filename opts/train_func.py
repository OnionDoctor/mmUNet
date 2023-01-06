# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     train_func
 Version:       
 Description:
 Date:          2022/8/9
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""
import os
import torch.optim as optim

from model.EarlyMMUNet import EarlyMMUNet
from model.LateMMUNet import LateMMUNet
from model.InterMMUNet import InterMMUNet

from opts.dataloader import create_dataloader, ExWallDefects
from opts.loss_func import DiceLoss, IoULoss
from opts.log_func import get_lastest_save_dir
from opts.run_func import train_model


def train_MMUNet(fusion_mode: str = 'early',
                 in_ch1: int = 3,
                 in_ch2: int = 1,
                 num_cls: int = 3,
                 data_src: str = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset',
                 input_size=(512, 512),
                 augmentation: bool = True,
                 batch_size: int = 4,
                 shuffle: bool = True,
                 split: float = 0.2,
                 loss_fc='Dice',
                 optim_fc='RMSProp',
                 init_lr: float = 1e-3,
                 epochs: int = 1,
                 **kwargs):
    # dataset
    rgb_ext = kwargs.get('rgb_ext', 'jpg')
    t_ext = kwargs.get('t_ext', 'npy')
    msk_ext = kwargs.get('msk_ext', 'png')
    dataset = ExWallDefects(rgb_src=os.path.join(data_src, 'images_rgb'),
                            t_src=os.path.join(data_src, 'images_t'),
                            msk_src=os.path.join(data_src, 'labels'),
                            label_desc=os.path.join(data_src, 'label_desc.yaml'),
                            size=input_size,
                            augmentation=augmentation,
                            rgb_ext=rgb_ext,
                            t_ext=t_ext,
                            msk_ext=msk_ext)

    # dataloader
    num_worker = kwargs.get('num_worker', 8)
    reuse = kwargs.get('reuse', True)
    train_loader, valid_loader = create_dataloader(dataset,
                                                   batch_size=batch_size,
                                                   num_workers=num_worker,
                                                   shuffle=shuffle,
                                                   split=split,
                                                   reuse=reuse)

    # model
    residual = kwargs.get('residual', False)
    down_mode = kwargs.get('down_mode', 'pool')
    up_mode = kwargs.get('up_mode', 'deconv')
    base = kwargs.get('base', 64)
    depth = kwargs.get('depth', 4)
    if fusion_mode == 'early':
        model = EarlyMMUNet(in_ch1=in_ch1,
                            in_ch2=in_ch2,
                            num_cls=num_cls,
                            residual=residual,
                            down_mode=down_mode,
                            up_mode=up_mode,
                            base=base,
                            depth=depth)

    elif fusion_mode == 'late':
        model = LateMMUNet(in_ch1=in_ch1,
                           in_ch2=in_ch2,
                           num_cls=num_cls,
                           residual=residual,
                           down_mode=down_mode,
                           up_mode=up_mode,
                           base=base,
                           depth=depth)

    elif fusion_mode == 'inter':
        model = InterMMUNet(in_ch1=in_ch1,
                            in_ch2=in_ch2,
                            num_cls=num_cls,
                            residual=residual,
                            down_mode=down_mode,
                            up_mode=up_mode,
                            base=base,
                            depth=depth)

    # finetune
    weights = kwargs.get('weights', None)
    if weights:
        model.load_weights(weights)

    # criterion
    epsilon = kwargs.get('epsilon', 1.0e-6)
    if loss_fc == 'Dice':
        criterion = DiceLoss(num_class=num_cls, epsilon=epsilon)
    elif loss_fc == 'IoU':
        criterion = IoULoss(num_class=num_cls, epsilon=epsilon)
    else:
        raise ValueError("Please input a feasible loss function!")

    # optimizer
    weight_decay = kwargs.get('weight_decay', 1.0e-8)
    momentum = kwargs.get('momentum', 0.9)
    if optim_fc == 'RMSProp':
        optimizer = optim.RMSprop(model.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError("Please input a feasible optimizer function!")

    # save dir
    train_dir = os.path.join('runs', 'train')
    save_dir = get_lastest_save_dir(train_dir)

    # train
    train_model(model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                epochs=epochs,
                save_dir=save_dir)
