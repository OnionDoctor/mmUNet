# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     plot_func
 Version:       
 Description:
 Date:          2022/8/9
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""

import numpy as np
import os

from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, auc, f1_score

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# 绘制混淆矩阵
def plot_confusion_matrix(cf_m, labels=None, save_path=None):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    norm = np.divide(cf_m, cf_m.sum(axis=0), out=np.zeros_like(cf_m, dtype=float), where=cf_m.sum(axis=0) != 0)

    im = ax.matshow(norm, cmap=plt.get_cmap('Blues'))

    w, h = norm.shape
    for i in range(w):
        for j in range(h):
            element = norm[j, i]
            ax.text(i, j, '{:.2f}'.format(element), va='center', ha='center')

    # ax.set_xlabel('Prediction')
    # ax.set_ylabel('Ground Truth')
    ax.set_title('Confusion Matrix')

    if labels and len(cf_m) == len(labels):
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.tight_layout()

    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'confusion_matrix.png')
            plt.savefig(save_path, dpi=300)
        else:
            plt.savefig(save_path, dpi=300)
    else:
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# 分析曲线
# ----------------------------------------------------------------------------------------------------------------------
FIGSIZE = (8, 6)


def plot_ROC_curve(true, prob, labels=None, save_path=None):
    # 输入 N1HW 和 NCHW
    true = np.squeeze(true)
    C = prob.shape[1]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    ROCs = []
    AUCs = []

    for ch in range(C):
        y_true = np.zeros_like(true)
        y_true[true == ch] = 1
        y_prob = prob[:, ch, :, :]
        fpr, tpr, thres = roc_curve(y_true.reshape(-1), y_prob.reshape(-1))
        roc_auc = auc(fpr, tpr)

        if labels:
            ax.plot(fpr, tpr, label=labels[ch] + ' AUC = {:.2f}'.format(roc_auc))
        else:
            ax.plot(fpr, tpr, label='class ' + str(ch) + ' AUC = {:.2f}'.format(roc_auc))

        ROCs.append([fpr.tolist(), tpr.tolist()])
        AUCs.append(roc_auc)

    ax.plot((0, 1), (0, 1), linestyle='--', label='base')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.set_title('Receiver Operating Characteristic Curve')

    ax.legend()

    plt.tight_layout()

    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'Curve_ROC.png')
            plt.savefig(save_path, dpi=300)
        else:
            plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return AUCs, ROCs


def plot_PR_curve(true, prob, labels=None, save_path=None):
    # 输入 N1HW 和 NCHW
    true = np.squeeze(true)
    C = prob.shape[1]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    total_true = []
    total_prob = []
    APs = []
    # 对每一类来说
    for ch in range(C):
        y_true = np.zeros_like(true)
        y_true[true == ch] = 1
        y_prob = prob[:, ch, :, :]
        pre, rec, thres = precision_recall_curve(y_true.reshape(-1), y_prob.reshape(-1))  # 注意 thresholds 的长度少1
        ap = average_precision_score(y_true.reshape(-1), y_prob.reshape(-1))

        total_true.append(y_true.reshape(-1))
        total_prob.append(y_prob.reshape(-1))
        APs.append(ap)

        if labels:
            ax.plot(rec, pre, label=labels[ch] + ' AP = {:.2f}'.format(ap))
        else:
            ax.plot(rec, pre, label='class ' + str(ch) + ' AP = {:.2f}'.format(ap))

    # 对整体来说，micro
    total_true, total_prob = np.concatenate(total_true), np.concatenate(total_prob)
    pre, rec, thres = precision_recall_curve(total_true, total_prob)
    mAP = average_precision_score(total_true, total_prob)

    ax.plot(rec, pre, label='Micro mAP = {:.2f}'.format(mAP))

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    ax.set_title('Precision Recall Curve')

    ax.legend()

    plt.tight_layout()

    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'Curve_PR.png')
            plt.savefig(save_path, dpi=300)
        else:
            plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return APs, mAP


def plot_F1C_curve(true, prob, n_step=50, labels=None, save_path=None):
    # 输入 N1HW 和 NCHW
    true = np.squeeze(true)  # NHW
    C = prob.shape[1]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)

    f1s = []
    confs = np.linspace(0, 1, num=n_step, endpoint=True)
    Thresholds = []
    # 对每一类来说
    for ch in range(C):
        f1_ = []
        for threshold in confs:
            y_true = np.zeros_like(true)
            y_true[true == ch] = 1

            y_prob = prob[:, ch, :, :]
            y_pred = np.zeros_like(y_prob)
            y_pred[y_prob > threshold] = 1

            f1 = f1_score(y_true.reshape(-1), y_pred.reshape(-1))
            f1_.append(f1)

        max_i = np.argmax(f1_)
        max_f1 = f1_[max_i]
        max_conf = confs[max_i]

        Thresholds.append(max_conf)

        if labels:
            ax.plot(confs, f1_, label=labels[ch] + ' F1 {:.2f} @ {:.2f}'.format(max_f1, max_conf))
        else:
            ax.plot(confs, f1_, label='class ' + str(ch) + ' F1 {:.2f} @ {:.2f}'.format(max_f1, max_conf))

        f1s.append(f1_)

    # 对整体来说，macro
    f1s = np.mean(f1s, axis=0)
    max_i = np.argmax(f1s)
    max_f1 = f1s[max_i]
    max_conf = confs[max_i]

    ax.plot(confs, f1s, label='Macro F1 {:.2f} @ {:.2f}'.format(max_f1, max_conf))

    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1 Score')

    ax.set_title('F1-Confidence Curve')

    ax.legend()

    plt.tight_layout()

    if save_path:
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, 'Curve_F1.png')
            plt.savefig(save_path, dpi=300)
        else:
            plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return Thresholds, max_conf
