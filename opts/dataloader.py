# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     dataloader
 Version:       
 Description:
 Date:          2022/8/4
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""

from typing import Tuple
import os

import numpy as np
from glob import glob
from PIL import Image, ImageOps
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import torchvision.transforms.functional as TF

torch.manual_seed(1)  # 测试用固定的随机参数

# -------------------------------------------------------------
# 数据集
# -------------------------------------------------------------
class ExWallDefects(Dataset):
    def __init__(self,
                 rgb_src,
                 t_src,
                 msk_src,
                 label_desc: str = None,
                 size: Tuple = (480, 480),
                 augmentation=False,
                 rgb_ext: str = 'jpg',
                 t_ext: str = 'npy',
                 msk_ext: str = 'png',
                 **kwargs):
        super(ExWallDefects, self).__init__()

        self.rgb_src = rgb_src
        self.rgb_ext = rgb_ext
        self.t_src = t_src
        self.t_ext = t_ext
        self.msk_src = msk_src
        self.msk_ext = msk_ext

        if len(size) == 1:
            self.size = (size, size)
        else:
            self.size = size

        with open(label_desc, 'r') as f:
            label_desc = yaml.safe_load(f)
        self.label_desc = label_desc

        t_fs = glob(os.path.join(self.t_src, '*.' + self.t_ext))        # t 图片容易存在问题
        basenames = [os.path.basename(t_f) for t_f in t_fs]
        rgb_fs = [os.path.join(self.rgb_src, basename.replace(self.t_ext, self.rgb_ext)) for basename in basenames]
        msk_fs = [os.path.join(self.msk_src, basename.replace(self.t_ext, self.msk_ext)) for basename in basenames]

        self.num = len(rgb_fs)
        self.rgb_fs = rgb_fs
        self.t_fs = t_fs
        self.msk_fs = msk_fs
        self.augmentation = augmentation

        if self.augmentation:
            self.random_crop = kwargs.get('RandomCrop', True)
            self.random_flip = kwargs.get('RandomFlip', True)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        # 读取 rgb 数据 -> Image
        rgb = Image.open(self.rgb_fs[index])
        rgb = ImageOps.exif_transpose(rgb)

        # 读取 t 数据 -> Image
        t = np.load(self.t_fs[index]).transpose()
        t = np.array((t - t.min()) / (t.max() - t.min()) * 255, dtype=np.uint8)
        t = Image.fromarray(t.transpose(), 'L')

        # 读取 掩码 数据 -> Image
        msk = Image.open(self.msk_fs[index])

        if self.augmentation:
            # 随机剪裁
            if self.random_crop:
                x, y, h, w = T.RandomCrop.get_params(rgb, output_size=self.size)
                rgb, t, msk = map(lambda img: TF.crop(img, x, y, w, h), (rgb, t, msk))

            # 随机翻转
            if self.random_flip:
                flip = np.random.randint(3)
                if flip == 1:
                    rgb, t, msk = map(TF.hflip, (rgb, t, msk))
                elif flip == 2:
                    rgb, t, msk = map(TF.vflip, (rgb, t, msk))
                else:
                    pass

        else:
            rgb, t, msk = map(T.CenterCrop(size=self.size), (rgb, t, msk))

        # rgb.show()
        # t.show()
        # msk.show()

        rgb, t = map(TF.to_tensor, (rgb, t))

        msk = self.color2mask(msk)
        msk = torch.squeeze(torch.tensor(msk))        # 1 x h x w -> h x w

        return rgb, t, msk

    # 彩色标注图到掩码
    def color2mask(self, color) -> np.ndarray:
        w, h = color.size
        mask = np.zeros((h, w), dtype=np.uint8)
        color = np.array(color)  # PIL Image w x h -> array h x w

        for item in self.label_desc:
            r, g, b = item['color']['r'], item['color']['g'], item['color']['b']
            id = item['id']
            mask_ = np.logical_and(np.logical_and(color[:, :, 0] == r, color[:, :, 1] == g), color[:, :, 2] == b)
            mask[mask_] = id

        return mask


# -------------------------------------------------------------
# 训练集和测试集
# -------------------------------------------------------------
def create_dataloader(dataset: Dataset,
                      batch_size: int = 2,
                      num_workers: int = 8,
                      shuffle: bool = True,
                      split: float = 0.2,
                      reuse: bool = True):
    total = len(dataset)

    batch_size = min(batch_size, total)
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, num_workers])

    idx = list(range(total))

    if shuffle:
        np.random.shuffle(idx)

    if 0 < split < 1:
        valid_size = int(split * total)
        train_idx, valid_idx = idx[: -valid_size], idx[-valid_size:]

        if reuse:
            train_idx = train_idx + valid_idx

        if shuffle:
            train_sampler = SubsetRandomSampler(indices=train_idx)
            valid_sampler = SubsetRandomSampler(indices=valid_idx)
        else:
            train_sampler = SequentialSampler(data_source=train_idx)
            valid_sampler = SequentialSampler(data_source=valid_idx)

        train_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,
                                  num_workers=num_workers)
        valid_loader = DataLoader(dataset,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  num_workers=num_workers)

        return train_loader, valid_loader

    else:
        loader = DataLoader(dataset,
                            num_workers=num_workers,
                            batch_size=batch_size,
                            shuffle=shuffle)
        return loader


if __name__ == '__main__':
    dataset = ExWallDefects(
        rgb_src=r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\images_rgb',
        t_src=r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\images_t',
        msk_src=r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\labels',
        label_desc=r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\label_desc.yaml',
        size=(480, 480),
        augmentation=False,
        rgb_ext='jpg',
        t_ext='npy',
        msk_ext='png')

    for i, (rgb, t, msk) in enumerate(dataset):
        print(rgb.shape, t.shape, msk.shape)
        print(torch.max(t), torch.max(msk))

    # train_loader, valid_loader = create_dataloader(dataset,
    #                                                batch_size=2,
    #                                                num_workers=8,
    #                                                shuffle=False,
    #                                                split=0.2)