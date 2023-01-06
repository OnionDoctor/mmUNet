# -*- coding: utf-8 -*-
"""
----------------------------------------
 File Name:     run_func
 Version:       
 Description:
 Date:          2022/8/8
 Author:        Xincong Yang
 Affilation:    Harbin Institute of Technology (Shenzhen)
 E-mail:        yangxincong@hit.edu.cn
 
----------------------------------------
"""
import os.path
import time

from tqdm import tqdm

import torch
import numpy as np
import torch.nn.functional as F

import json

from opts.eval_func import get_confusion_matrix, get_fitness
from opts.log_func import get_logger
from opts.plot_func import plot_confusion_matrix
from opts.plot_func import plot_PR_curve, plot_F1C_curve, plot_ROC_curve
from torch.utils.tensorboard import SummaryWriter


# 模型运行一次
def run_model(model,
              dataloader,
              phase,
              criterion,
              optimizer,
              epoch,
              epochs,
              writer=None,
              device='cuda',
              return_all=False):
    if phase == 'train':  # 训练时不考虑 confidence 不用保存 logit
        model.train()
    elif phase == 'valid':  # 验证时考虑 confidence 需要保存 logit
        model.eval()
    else:
        raise ValueError('Please input feasible phase.')

    model.to(device)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))

    optimizer.zero_grad()  # 梯度归零

    result = {'loss': 0,
              'cf_m': np.zeros((model.num_cls, model.num_cls)),
              'total': len(dataloader),
              'true': [],
              'pred': [],
              'logit': []}

    for i, (rgb_x, t_x, y) in progress_bar:

        rgb_x = rgb_x.to(device)
        t_x = t_x.to(device)
        y_true = y.to(device).long()

        with torch.set_grad_enabled(phase == 'train'):

            outputs = model(rgb_x.float(), t_x.float())
            loss = criterion(outputs, y_true)  # 计算损失

            logits = F.softmax(outputs, dim=1)  # 正则化预测概率结果
            _, y_pred = torch.max(logits, 1)  # 真值为 y 预测值为 y_
            cf_m = get_confusion_matrix(y_true=y_true, y_pred=y_pred, num_class=model.num_cls)

            if phase == 'train':
                loss.backward()
                optimizer.step()

        # logging
        progress_bar.set_description(('%10s' * 1 + '%12.4g' * 1) % (f'{epoch + 1}/{epochs}', loss))

        result['loss'] += loss.item()
        result['cf_m'] += cf_m.detach().cpu().numpy()

        if return_all:
            result['true'].append(y_true.detach().cpu().numpy())
            result['pred'].append(y_pred.detach().cpu().numpy())
            result['logit'].append(logits.detach().cpu().numpy())

        if writer:
            writer.add_scalar(f'{phase}/loss', loss.item(), epoch * len(dataloader) + i)

            if i == 0:  # 仅在每个batch的开始存储图片
                writer.add_images(f'{phase}/rgb_images', rgb_x[0], epoch * len(dataloader) + i, dataformats='CHW')
                writer.add_images(f'{phase}/t_images', torch.squeeze(t_x[0]), epoch * len(dataloader) + i,
                                  dataformats='HW')

                label_step = int(255 / model.num_cls)  # 方便预测展示
                writer.add_images(f'{phase}/true', y_true[0] * label_step, epoch * len(dataloader) + i,
                                  dataformats='HW')
                writer.add_images(f'{phase}/pred', y_pred[0] * label_step, epoch * len(dataloader) + i,
                                  dataformats='HW')

        result['loss'] /= len(dataloader.dataset)  # 计算平均 loss

    return result


# 模型训练一次
def train_model(model,
                train_loader,
                valid_loader,
                criterion,
                optimizer,
                epochs,
                save_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights_dir = os.path.join(save_dir, 'weights')
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    best_w, last_w = os.path.join(weights_dir, 'best.pt'), os.path.join(weights_dir, 'last.pt')

    since = time.time()

    logger = get_logger(src=save_dir)
    logger.info(f'save results to {save_dir}')
    logger.info(f'start training for {epochs} epochs ...')

    writer = SummaryWriter(log_dir=save_dir)

    best_fi = 0.0
    label_desc = train_loader.dataset.label_desc

    for epoch in range(epochs):

        train_result = run_model(model=model,
                                 dataloader=train_loader,
                                 phase='train',
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 epoch=epoch,
                                 epochs=epochs,
                                 writer=writer,
                                 device=device,
                                 return_all=epoch + 1 == epochs)

        acc, pre, rec, f1 = get_fitness(train_result, average='macro')
        logger.info(f'Train [%4d / %4d] accuracy: %.4f precision: %.4f recall: %.4f f1: %.4f loss: %.4f'
                    % (epoch + 1, epochs, acc, pre, rec, f1, train_result['loss']))

        valid_result = run_model(model=model,
                                 dataloader=valid_loader,
                                 phase='valid',
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 epoch=epoch,
                                 epochs=epochs,
                                 writer=writer,
                                 device=device,
                                 return_all=epoch + 1 == epochs)  # 最后一组返回所有

        acc, pre, rec, f1 = get_fitness(valid_result, average='macro')
        logger.info(f'Valid [%4d / %4d] accuracy: %.4f precision: %.4f recall: %.4f f1: %.4f loss: %.4f'
                    % (epoch + 1, epochs, acc, pre, rec, f1, valid_result['loss']))

        fi = f1  # 在本模型中，根据f1来判断模型好坏
        # 最优模型保存
        best_epoch = fi > best_fi
        if best_epoch:
            best_fi = fi
            ckpt = {'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'label_desc': label_desc}
            torch.save(ckpt, best_w)
            del ckpt

        # 最终模型保存
        last_epoch = epoch + 1 == epochs
        """
        if last_epoch:
            ckpt = {'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'label_desc': label_desc}
            torch.save(ckpt, last_w)
            del ckpt

            # 打印最终结果
            cf_m = train_result['cf_m'] + valid_result['cf_m']

            # 打印准确率等
            acc_m, pre_m, rec_m, f1_m = get_fitness(result=cf_m, average=None)
            logger.info(('\n' + '%15s' * 1 + '%15s' * 4) % ('label', 'accuracy', 'precision', 'recall', 'f1 score'))

            labels = [item['name'] for item in label_desc]
            labels = ['background'] + labels  # 预测的时候要多一个

            for id, label in enumerate(labels):
                logger.info(('%15s' * 1 + '%15.4g' * 4) % (label, acc_m[id], pre_m[id], rec_m[id], f1_m[id]))

            # 绘图
            plot_confusion_matrix(cf_m, labels=labels, save_path=save_dir)

            # 获取所有测试结果及概率
            total_true = train_result['true'] + valid_result['true']
            total_logit = train_result['logit'] + valid_result['logit']
            total_true, total_logit = map(lambda x: np.concatenate(x, axis=0), (total_true, total_logit))  # N1HW, NCHW

            AUCs, ROCs = plot_ROC_curve(total_true, total_logit, labels=labels, save_path=save_dir)
            APs, mAP = plot_PR_curve(total_true, total_logit, labels=labels, save_path=save_dir)
            Thresholds, mThreshold = plot_F1C_curve(total_true, total_logit, labels=labels, save_path=save_dir)

            # 结果保存，一定都是list
            results = {'confusion_matrix': cf_m.tolist(),
                       'labels': labels,
                       'APs': APs,
                       'mAP': mAP,
                       'AUCs': AUCs,
                       'ROCs': ROCs,
                       'Thresholds': Thresholds,
                       'mThreshold': mThreshold,
                       }

            with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as file:
                json.dump(results, file, ensure_ascii=False)
        """

    logger.info(f'\nfinish training {epochs} epochs in {(time.time() - since) / 3600: .3f} hours.')
    logger.info(f'results saved to {save_dir}')

    writer.close()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # 测试 run_model 函数
    from opts.dataloader import *
    from model.EarlyMMUNet import *
    from opts.loss_func import *

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

    train_loader, valid_loader = create_dataloader(dataset,
                                                   batch_size=2,
                                                   num_workers=8,
                                                   shuffle=False,
                                                   split=0.2)

    model = EarlyMMUNet(in_ch1=3,
                        in_ch2=1,
                        num_cls=3,
                        residual=False,
                        down_mode='pool',
                        up_mode='deconv',
                        base=64,
                        depth=4)
    # run_model 测试
    # results = run_model(model=model,
    #                     dataloader=train_loader,
    #                     phase='valid',
    #                     criterion=DiceLoss(num_class=model.num_cls),
    #                     optimizer=torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-08, momentum=0.9),
    #                     epoch=1,
    #                     epochs=2,
    #                     writer=None,
    #                     device='cuda',
    #                     return_all=False)

    train_model(model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                criterion=DiceLoss(num_class=model.num_cls),
                optimizer=torch.optim.RMSprop(model.parameters(), lr=1e-3, weight_decay=1e-08, momentum=0.9),
                epochs=2,
                save_dir=r'C:\Users\HP\PycharmProjects\mmUNet\runs')
