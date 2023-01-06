# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     eval_func
 Version:       
 Description:
 Date:          2022/8/8
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""
import numpy as np
import torch


# 获取混淆矩阵 Tensor 操作
def get_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_class):
    true = y_true.view(-1)
    pred = y_pred.view(-1)
    M = torch.bincount(num_class * true + pred)
    if len(M) < num_class * num_class:
        m = torch.zeros(num_class * num_class - len(M), dtype=torch.long).to(true.device)
        M = torch.cat((M, m))
    M = M.reshape(num_class, num_class)  # true x pred
    return M


# 非 tensor 操作
def get_fitness(result, average='macro'):
    cf_m = result['cf_m'] if type(result) == dict else result
    tp = np.diag(cf_m)  # True Positive
    tn = np.diag(cf_m).sum() - np.diag(cf_m)  # True Negative
    fp = np.sum(cf_m, axis=0) - tp  # False Positive
    fn = np.sum(cf_m, axis=1) - tp  # False Positive

    if average == 'macro':
        acc, pre, rec, f1 = get_fitness(result, average=None)
        acc, pre, rec, f1 = map(np.mean, [acc, pre, rec, f1])
    elif average == 'micro':
        tp, tn, fp, fn = map(np.sum, [tp, tn, fp, fn])
        acc = np.divide(tp + tn, tp + tn + fp + fn)
        pre = np.divide(tp, fp + tp)
        rec = np.divide(tp, fn + tp)
        f1 = 2 * pre * rec / (pre + rec)
    else:
        acc = np.divide(tp + tn, tp + tn + fp + fn,
                        out=np.zeros_like(tp, dtype=float), where=tp + tn + fp + fn != 0)
        pre = np.divide(tp, fp + tp,
                        out=np.zeros_like(tp, dtype=float), where=fp + tp != 0)
        rec = np.divide(tp, fn + tp,
                        out=np.zeros_like(tp, dtype=float), where=fn + tp != 0)
        f1 = np.divide(2 * pre * rec, pre + rec,
                       out=np.zeros_like(pre, dtype=float), where=pre + rec != 0)

    return acc, pre, rec, f1
