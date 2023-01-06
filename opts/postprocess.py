# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     postprocess
 Version:       
 Description:
 Date:          2022/8/21
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""

import csv
import os.path

import numpy as np
import matplotlib.pylab as plt
from opts.log_func import parse_logging
from math import factorial


def draw_logging(ax, logging_data, index='loss', **kwargs):
    start = kwargs.get('start', 0)
    end = kwargs.get('end', 0)
    weight = kwargs.get('weight', 0.5)

    for phase in ['train', 'valid']:

        if index in ['accuracy', 'acc']:
            y = logging_data[phase][:, 0]
        elif index in ['precision', 'pre']:
            y = logging_data[phase][:, 1]
        elif index in ['recall', 'rec']:
            y = logging_data[phase][:, 2]
        elif index in ['f1_score', 'f1']:
            y = logging_data[phase][:, 3]
        else:
            y = logging_data[phase][:, 4]

        if end > start and end > 0:
            y = y[start: end]  # 是否考虑截取

        x = np.arange(len(y))
        ax.plot(x, y, )

        order = kwargs.get('order', 1)
        while order > 0:
            y = smooth(y, weight)
            order -= 1



        ax.legend()


# 数据平滑
# https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
# def savitzky_golay(y, window_size, order, deriv=0, rate=1):
#     # window_size必须为奇数，大于1，且不小于order + 2
#     order_range = range(order + 1)
#     half_window = (window_size - 1) // 2
#     b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
#     m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
#     firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
#     lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
#     y = np.concatenate((firstvals, y, lastvals))
#     return np.convolve(m[::-1], y, mode='valid')

def smooth(scalars, weight):
    last = scalars[0]
    smoothed =[]
    for scalar in scalars:
        smoothed_val = last * weight + (1 - weight) * scalar
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


if __name__ == '__main__':
    logging_p = r'C:\Users\HP\PycharmProjects\mmUNet\runs\train\D4L1000\logging.log'
    logging_d = parse_logging(logging_p)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    draw_logging(ax, logging_d, index='loss', order=1)

    plt.show()
