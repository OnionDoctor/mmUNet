# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     demo_train
 Version:       
 Description:
 Date:          2022/8/9
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""
import argparse
import yaml

from opts.train_func import train_MMUNet


def parse_opt():
    parser = argparse.ArgumentParser(allow_abbrev=True)

    # 全参数
    # parser.add_argument('--hyps', default='hyps/earlymmunet_test.yaml', help='YAML file to all hyper parameters',
    #                     type=str)
    parser.add_argument('--hyps', default='hyps/intermmunet_test.yaml', help='YAML file to all hyper parameters',
                        type=str)
    # parser.add_argument('--hyps', default='hyps/latemmunet_test.yaml', help='YAML file to all hyper parameters',
    #                     type=str)

    opt = parser.parse_args()

    return opt


def main(opt):
    with open(opt.hyps, 'r') as file:
        hyper_parameters = yaml.safe_load(file)

    # print(hyper_parameters)
    train_MMUNet(**hyper_parameters)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
