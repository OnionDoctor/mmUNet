# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     proprocess
 Version:       
 Description:
 Date:          2022/8/4
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""
import os.path
import csv

import PIL.Image
import numpy as np

from PIL import Image
from glob import glob
from tqdm import tqdm


# 批量裁剪图片
def crop_images(src: str,
                dst: str,
                x: int,
                y: int,
                w: int,
                h: int,
                src_ext: str = 'jpg',
                dst_ext: str = 'jpg'):
    files_src = glob(os.path.join(src, '*.' + src_ext))
    for file_src in tqdm(files_src):
        img = Image.open(file_src)
        crop = img.crop((x, y, x + w, y + h))

        basename = os.path.basename(file_src)
        if src_ext != dst_ext:
            file_dst = os.path.join(dst, basename.replace(src_ext, dst_ext))
        else:
            file_dst = os.path.join(dst, basename)

        crop.save(file_dst)


# 合并标注图像
def fuse_images(image01_src: str,
                image02_src: str,
                w: int = 480,
                h: int = 480,
                save_dst: str = ''):
    img01 = Image.open(image01_src)  # 小尺寸
    img02 = Image.open(image02_src)  # 大尺寸

    if img01.size != (w, h):
        img01 = img01.resize((w, h), resample=PIL.Image.NEAREST)

    if img02.size != (w, h):
        img02 = img02.resize((w, h), resample=PIL.Image.NEAREST)

    bg_img = np.array(img01)
    mk_img = np.array(img02)
    mask = np.sum(mk_img, axis=2) != 0

    for i in range(3):
        bg_img[:, :, i][mask] = mk_img[:, :, i][mask]

    img = Image.fromarray(bg_img)

    if save_dst == '':
        return img
    else:
        img.save(save_dst)


# 读取 csv 热数据
def read_t_csv(csv_f):
    data = []
    with open(csv_f, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i >= 2 and row != []:
                line = [float(j) for j in row[1:] if j != '']
                data.append(np.array(line))
    data = np.array(data)
    return data




if __name__ == '__main__':
    # 1. 合并标签
    # dst = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\labels'
    # j = 0
    # for i in tqdm(range(1810, 1846, 2)):
    #     src01 = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\label\FLIR{}.png'.format(i)
    #     src02 = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\label\FLIR{}- 鐓х墖.png'.format(i)
    #
    #     save_dst = os.path.join(dst, '{:03}.png'.format(j))
    #
    #     if os.path.exists(src01) and os.path.exists(src02):
    #         img = fuse_images(src01, src02, 640, 480, save_dst)
    #         j += 1
    #
    #     elif os.path.exists(src01):
    #         img = Image.open(src01)
    #         img.save(save_dst)
    #         j += 1
    #
    #     elif os.path.exists(src02):
    #         img = Image.open(src02)
    #         img.save(save_dst)
    #         j += 1
    #
    #     else:
    #         print('aaa')

    # 6. 重命名原图
    # j = 0
    # for i in tqdm(range(1810, 1846, 2)):
    #     src = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\images_t\FLIR1810- 照片.jpg'.format(i)
    #     dst = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\images_t\{:03}.jpg'.format(j)
    #     os.rename(src, dst)
    #     j += 1
    #
    # j = 0
    # for i in tqdm(range(1810, 1846, 2)):
    #     src = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\images_rgb\FLIR{}- 照片.jpg'.format(i)
    #     dst = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\images_rgb\{:03}.jpg'.format(j)
    #     os.rename(src, dst)
    #     j += 1

    # 2. 调整 csv 到 numpy 加快数据读取速率 480 * 640
    # src = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\images_t'
    # for csv_f in glob(os.path.join(src, '*.csv')):
    #     data = read_t_csv(csv_f)
    #     basename = os.path.basename(csv_f)
    #     dst = os.path.join(src, basename.replace('.csv', '.npy'))
    #     np.save(dst, data)

    src = r'C:\Users\HP\Pictures\RGBT-HITSZ-G\dataset\images_t\001.csv'
    data = read_t_csv(src)
    dst = src.replace('.csv', '.npy')
    np.save(dst, data)