# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     json2png
 Version:       
 Description:
 Date:          2022/8/5
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""

import os
import json
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import yaml

# 颜色基础
COLOR_DICT = {'r': np.array([[0.0, 1.0],
                             [0.1, 1.0],
                             [0.2, 0.0],
                             [0.4, 0.2],
                             [0.6, 0.0],
                             [0.8, 0.5],
                             [1.0, 0.0]]),
              'g': np.array([[0.0, 0.0],
                             [0.1, 1.0],
                             [0.2, 1.0],
                             [0.4, 1.0],
                             [0.6, 0.0],
                             [0.8, 0.0],
                             [1.0, 0.0]]),
              'b': np.array([[0.0, 0.0],
                             [0.1, 0.0],
                             [0.2, 0.0],
                             [0.4, 1.0],
                             [0.6, 1.0],
                             [0.8, 0.5],
                             [1.0, 0.0]])}


# 获取颜色
def get_color(x,
              mode='bgr',
              norm=False):  # 0 <= x < 1
    color = []
    for ch in mode:
        i = np.digitize(x, COLOR_DICT[ch][:, 0]) - 1
        x1, y1 = COLOR_DICT[ch][i][0], COLOR_DICT[ch][i][1]
        x2, y2 = COLOR_DICT[ch][i + 1][0], COLOR_DICT[ch][i + 1][0]
        c = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
        if norm:
            color.append(c)
        else:
            color.append(int(c * 255))
    return color


# 标签颜色
LABEL_COLOR = [get_color(i, mode='bgr', norm=False) for i in np.linspace(0, 1, num=10, endpoint=False)]

# json格式转换
def json2png(json_src,
             dst_dir,
             labels):
    if os.path.isdir(json_src):
        json_files = glob(os.path.join(json_src, '*.json'))

        if len(json_files) > 1:  # 单层目录结构
            print("Processing {}".format(json_src))
            for json_file in tqdm(json_files):
                json2png(json_file, dst_dir, labels)

        else:  # 双层目录结构
            json_dirs = [json_dir for json_dir in os.listdir(json_src)]
            for json_dir in json_dirs:
                json_src_ = os.path.join(json_src, json_dir)
                if os.path.isdir(json_src_):
                    dst_dir_ = os.path.join(dst_dir, json_dir)
                    if not os.path.exists(dst_dir_):
                        os.mkdir(dst_dir_)
                    json2png(json_src_, dst_dir_, labels)

    else:

        with open(json_src, 'r') as f:
            json_file = json.load(f)

        height = json_file['imageHeight']
        width = json_file['imageWidth']
        shapes = json_file['shapes']

        img = np.zeros((height, width, 3), dtype=np.uint8)  # 注意先是高度、后是宽度

        for shape in shapes:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)

            if label in labels:
                i = labels.index(label)
                img = cv2.fillPoly(img, [points], color=LABEL_COLOR[i])

            else:
                raise ValueError('label {} is not included in label dict!'.format(label))

        json_name = os.path.basename(json_src)
        png_name = json_name.replace('.json', '.png')
        png_dst = os.path.join(dst_dir, png_name)

        cv2.imwrite(png_dst, img)


# 颜色说明
def get_label_desc(labels,
                   save_path=None):
    label_desc = []

    for i, label in enumerate(labels):
        id = i + 1
        b, g, r = LABEL_COLOR[i]
        label_desc.append({'id': id, 'name': label,
                           'color': {'r': r, 'g': g, 'b': b}})

    if save_path:
        if not save_path.endswith('.yaml'):
            save_path = save_path + '.yaml'

        with open(save_path, 'w') as f:
            yaml.dump(label_desc, f)

    else:
        print(label_desc)

    return label_desc

    # if save_path:
    #     cv2.imwrite(save_path, img)
    # else:
    #     cv2.imshow('color_description', img)
    #     if cv2.waitKey():
    #         cv2.destroyAllWindows()


if __name__ == '__main__':
    labels = [
        'detached',
        'missing',
    ]

    # 批量图像转换
    json2png(json_src=r'C:\Users\HP\Pictures\RGBT-HITSZ-G\raw',
             dst_dir=r'C:\Users\HP\Pictures\RGBT-HITSZ-G\label',
             labels=labels)

    # 获取标注描述
    label_desc = get_label_desc(labels, save_path=r'C:\Users\HP\Pictures\RGBT-HITSZ-G\label_desc.yaml')

    # 注意标注小的应排序在后
    # print(LABEL_COLOR)
