# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     log_func
 Version:       
 Description:
 Date:          2022/8/9
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""
import os
import logging
import re

# 获取最新的保存文件地址，
import numpy as np


def get_lastest_save_dir(src):
    if not os.path.exists(src):
        os.mkdir(src)
    sub_dirs = [os.path.join(src, d) for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    if len(sub_dirs) > 0:
        lastest_sub_dir = max(sub_dirs, key=os.path.getmtime)
        lastest_num = int(lastest_sub_dir[-2:]) + 1
        save_dir = lastest_sub_dir[:-2] + '%02d' % lastest_num
    else:
        save_dir = os.path.join(src, 'exp01')

    os.mkdir(save_dir)

    return save_dir


# 获取 logger
def get_logger(src):
    log_file = os.path.join(src, 'logging.log')

    level = logging.INFO
    format = '%(message)s'
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]

    logging.basicConfig(level=level,
                        format=format,
                        handlers=handlers)
    return logging.getLogger('exp_logging')


# 获取 logging 文件中的训练信息
def parse_logging(logging_file):
    logging_data = {}
    train_data = []
    valid_data = []
    pattern = re.compile('[0-9].[0-9]{4}')
    time_pattern = re.compile('[0-9].[0-9]{3} ')
    with open(logging_file, 'r') as file:
        for line in file.readlines():
            if line.startswith('Train'):
                train_data.append([float(i) for i in re.findall(pattern, line)])
            elif line.startswith('Valid'):
                valid_data.append([float(i) for i in re.findall(pattern, line)])
            elif line.startswith('finish'):
                logging_data['time'] = float(re.findall(time_pattern, line)[0])
    # 顺序为 accuracy, precision, recall, f1, loss
    logging_data['train'] = np.array(train_data)
    logging_data['valid'] = np.array(valid_data)

    return logging_data


if __name__ == '__main__':
    log_file = r'C:\Users\HP\PycharmProjects\mmUNet\runs\train\D4Lh\logging.log'
    data = parse_logging(log_file)
    print(data['time'])
