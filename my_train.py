# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger

from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor  # , show_result_pyplot
import numpy as np
import torch
import os
import glob
import pandas as pd
import mmcv
from tqdm import tqdm
import cv2
import os.path as osp


def rle_decode(mask_rle, shape=(512, 512)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def TC_cfg(cfg):
    # cfg.model.decode_head[0].num_classes = 2
    # cfg.model.decode_head[0].loss_decode = [dict(
    #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5, class_weight=[0.16, 0.84]),
    # ]
    cfg.model.decode_head.num_classes = 2
    cfg.model.decode_head.loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8, class_weight=[0.16, 0.84]),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.16, 0.84]),
    ]
    cfg.optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
    cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    # cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    cfg.runner = dict(type='IterBasedRunner', max_iters=200000)
    cfg.checkpoint_config = dict(by_epoch=False, interval=100000)
    cfg.evaluation = dict(interval=100000, metric='mDice', pre_eval=True)
    cfg.log_config = dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
            dict(type='TensorboardLoggerHook')
        ])

    cfg.work_dir = osp.join(log_dir,
                            osp.splitext(osp.basename(config))[0])
    cfg.load_from = './log/ocrnet_hr48_my/iter_360000.pth'
    return cfg


def SD_cfg(cfg):
    distributed = False
    seed_num = 1227
    log_dir = './log'

    cfg.model.decode_head[0].num_classes = 6
    cfg.model.decode_head[0].loss_decode = [dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ]#class_weight=[0.595,0.104,0.005,0.092,0.198,0.005]
    cfg.model.decode_head[1].num_classes = 6
    cfg.model.decode_head[1].loss_decode = [dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ]
    cfg.optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=0.0005)
    cfg.lr_config = dict(policy='poly', power=0.9, min_lr=5e-6, by_epoch=False,warmup='linear', warmup_iters=2000,  warmup_ratio=0.1)
    #,warmup='linear', warmup_iters=2000,  warmup_ratio=0.1
    # cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    cfg.runner = dict(type='IterBasedRunner', max_iters=120000)
    cfg.checkpoint_config = dict(by_epoch=False, interval=10000)
    cfg.evaluation = dict(interval=5000, metric='fwIoU', pre_eval=True)
    cfg.log_config = dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
            dict(type='TensorboardLoggerHook')
        ])


    # print(osp.splitext(osp.basename(config))[0])
    if not osp.exists(cfg.work_dir):
        os.mkdir(cfg.work_dir)
    # cfg.resume_from='./log/ocrnet_hr18_512x1024_40k_cityscapes_my/iter_150000(baseline).pth'
    # cfg.load_from = './log/ocrnet_hr18_512x1024_40k_cityscapes_my/iter_100000.pth'
    # cfg.load_from='./log/ocrnet_hr48_512x512_160k_ade20k/iter_190000（baseline）.pth'
    return cfg


def train():
    config = './configs/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes_my.py'
    # config='./configs/ocrnet/ocrnet_hr48_512x512_160k_ade20k.py'
    distributed = False
    seed_num = 1227
    log_dir = './log'

    cfg = Config.fromfile(config)
    cfg.work_dir = osp.join(log_dir,
                            osp.splitext(osp.basename(config))[0])
    # cfg=TC_cfg(cfg)
    cfg = SD_cfg(cfg)

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level='INFO')

    seed = init_random_seed(seed_num)
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    # logger.info(f'Config:\n{cfg.pretty_text}')

    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)
    #     dataset
    # logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    cfg.gpu_ids = range(1)  # gpu-----0,1
    model.CLASSES = datasets[0].CLASSES

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp)

    pass


if __name__ == '__main__':
    train()

    pass
