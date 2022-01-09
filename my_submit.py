import argparse
import copy
import os
from skimage import io
import os.path as osp
import time
import warnings
import shutil

import mmcv
import matplotlib.pyplot as plt
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
from my_local_demo import rle_encode,rle_decode
import re

def batch_save_result_submit_TC():


    config = './configs/ocrnet/ocrnet_hr48_512x512_160k_ade20k.py'
    distributed = False
    seed_num = 1227
    log_dir = './log'

    cfg = Config.fromfile(config)

    # # cfg.model.decode_head[0].num_classes = 2
    # # cfg.model.decode_head[0].loss_decode = [dict(
    # #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    # #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5, class_weight=[0.16, 0.84]),
    # # ]
    # cfg.model.decode_head.num_classes = 2
    # cfg.model.decode_head.loss_decode = [dict(
    #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8, class_weight=[0.16, 0.84]),
    #     dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.16, 0.84]),
    # ]
    # cfg.optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=0.0005)
    # cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    # # cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    # cfg.runner = dict(type='IterBasedRunner', max_iters=200000)
    # cfg.checkpoint_config = dict(by_epoch=False, interval=100000)
    # cfg.evaluation = dict(interval=100000, metric='mDice', pre_eval=True)
    # cfg.log_config = dict(
    #     interval=100,
    #     hooks=[
    #         dict(type='TextLoggerHook', by_epoch=False),
    #         dict(type='TensorboardLoggerHook')
    #     ])
    #
    # cfg.work_dir = osp.join(log_dir,
    #                         osp.splitext(osp.basename(config))[0])
    # cfg.load_from = './log/ocrnet_hr48_my/iter_360000.pth'

    cfg.model.decode_head[0].num_classes = 2
    cfg.model.decode_head[0].loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5, class_weight=[0.16, 0.84]),
    ]
    cfg.model.decode_head[1].num_classes = 2
    cfg.model.decode_head[1].loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.16, 0.84]),
    ]
    cfg.optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
    cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, warmup='linear', warmup_iters=2000, warmup_ratio=0.1,
                         by_epoch=False)
    # cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    cfg.runner = dict(type='IterBasedRunner', max_iters=320000)
    cfg.evaluation = dict(interval=30000, metric='mDice', pre_eval=True)
    cfg.log_config = dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
            dict(type='TensorboardLoggerHook')
        ])

    cfg.work_dir = osp.join(log_dir,
                            osp.splitext(osp.basename(config))[0])
    cfg.resume_from = './log/ocrnet_hr48_my/iter_460000.pth'

    # -----------------------------build config-------------------------------------------#

    subm=[]
    checkpoint=cfg.resume_from
    device='cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint, device=device)
    # test a single image
    # list_img=glob.glob(args.img)
    test_mask = pd.read_csv('./dataset/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: x)
    for i in tqdm(range(2500)):
        result = inference_segmentor(model, './dataset/test_a/' + test_mask['name'].iloc[i])
        result = np.array(result)
        result = result.astype('uint8').transpose(1, 2, 0)
        # print(type(result),result.shape)
        _, result = cv2.threshold(result, 0.5, 255, cv2.THRESH_BINARY)

        # print(osp.basename(test_mask['name'].iloc[i]))
        # mmcv.imwrite(result,'./tianchi_SegGame/test_result/3.0_R05K5826G4.jpg')
        # exit(0)

        subm.append([osp.basename(test_mask['name'].iloc[i]), rle_encode(result)])
    subm = pd.DataFrame(subm)
    subm.to_csv('./dataset/tmp_HRNet_after_model_2.0.csv', index=None, header=None, sep='\t')

    # mmcv.imshow(result)
    # show the results
    # my_show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     get_palette(args.palette),
    #     opacity=args.opacity)


    pass

def batch_save_result_submit_SD():
    config = './configs/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes_my.py'
    distributed = False
    seed_num = 1227
    log_dir = './log'

    cfg = Config.fromfile(config)

    cfg.model.decode_head[0].num_classes = 6
    cfg.model.decode_head[0].loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
    ]
    cfg.model.decode_head[1].num_classes = 6
    cfg.model.decode_head[1].loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8),
    ]
    cfg.optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
    cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False, warmup='linear', warmup_iters=2000,
                         warmup_ratio=0.1)
    # cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    cfg.runner = dict(type='IterBasedRunner', max_iters=150000)
    cfg.checkpoint_config = dict(by_epoch=False, interval=5000)
    cfg.evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)
    cfg.log_config = dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
            dict(type='TensorboardLoggerHook')
        ])

    cfg.work_dir = osp.join(log_dir,
                            osp.splitext(osp.basename(config))[0])
    cfg.resume_from = './log/ocrnet_hr18_512x1024_40k_cityscapes_my/iter_150000(baseline).pth'
    # -----------------------------build config-------------------------------------------#

    subm=[]
    checkpoint=cfg.resume_from
    device='cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint, device=device)
    img_list=glob.glob('./dataset_sd/初赛A榜_GF/*.png')
    if not os.path.exists('./dataset_sd/result'):
        os.mkdir('./dataset_sd/result')
    for i in tqdm(range(len(img_list))):
        try:
            result = inference_segmentor(model, img_list[i])
            # print(np.unique(result))
            result = np.array(result)
            result+=1
            result = result.astype('uint8').squeeze()
            # print(np.unique(result))
            io.imsave(img_list[i].replace('初赛A榜_GF','result').replace('GF','LT').replace('png','tif'),result)
            # exit(0)
            # _, result = cv2.threshold(result, 0.5, 255, cv2.THRESH_BINARY)
        except AttributeError:
            print(img_list[i])
            continue




    pass


def check_result(img_name='*.png'):
    config = './configs/ocrnet/ocrnet_hr48_512x512_160k_ade20k.py'
    distributed = False
    seed_num = 1227
    log_dir = './log'

    cfg = Config.fromfile(config)

    cfg.model.decode_head[0].num_classes = 2
    cfg.model.decode_head[0].loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5, class_weight=[0.16, 0.84]),
    ]
    cfg.model.decode_head[1].num_classes = 2
    cfg.model.decode_head[1].loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.16, 0.84]),
    ]
    cfg.optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
    cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, warmup='linear', warmup_iters=2000, warmup_ratio=0.1,
                         by_epoch=False)
    # cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    cfg.runner = dict(type='IterBasedRunner', max_iters=320000)
    cfg.evaluation = dict(interval=30000, metric='mDice', pre_eval=True)
    cfg.log_config = dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
            dict(type='TensorboardLoggerHook')
        ])

    cfg.work_dir = osp.join(log_dir,
                            osp.splitext(osp.basename(config))[0])
    cfg.resume_from = './log/ocrnet_hr48_512x512_160k_ade20k/iter_360000.pth'

    # -----------------------------build config-------------------------------------------#

    subm = []
    checkpoint = cfg.resume_from
    device = 'cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint, device=device)
    img_dir=osp.join(cfg.data_root,cfg.data.train.img_dir)
    list_img=glob.glob(img_dir+'/'+img_name)
    # test_mask = pd.read_csv('./dataset/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    # test_mask['name'] = test_mask['name'].apply(lambda x: x)
    for i in range(len(list_img)):
        result = inference_segmentor(model, list_img[i])
        result = np.array(result)
        result = result.astype('uint8').transpose(1, 2, 0)
        # print(type(result),result.shape)
        _, result = cv2.threshold(result, 0.5, 255, cv2.THRESH_BINARY)
        _,label=  cv2.threshold(mmcv.imread(list_img[i].replace('img_dir','ann_dir')), 0.5, 255, cv2.THRESH_BINARY)
        # print(osp.basename(list[i]))

        plt.figure(num=osp.basename(list_img[i]),figsize=(20,16),tight_layout=True)
        plt.subplot(1,3,1)
        plt.imshow(mmcv.imread(list_img[i]))
        plt.subplot(1, 3, 2)
        plt.imshow(label)
        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.show()

        # mmcv.imshow(result,osp.basename(list[i]))
        # mmcv.imwrite(result,'./tianchi_SegGame/test_result/3.0_R05K5826G4.jpg')
        # exit(0)



    pass

def check_result_2model(img_name='*.png'):
    config = './configs/ocrnet/ocrnet_hr48_512x512_160k_ade20k.py'
    distributed = False
    seed_num = 1227
    log_dir = './log'

    cfg = Config.fromfile(config)

    cfg.model.decode_head[0].num_classes = 2
    cfg.model.decode_head[0].loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=0.5, class_weight=[0.16, 0.84]),
    ]
    cfg.model.decode_head[1].num_classes = 2
    cfg.model.decode_head[1].loss_decode = [dict(
        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.8),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0, class_weight=[0.16, 0.84]),
    ]
    cfg.optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0005)
    cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, warmup='linear', warmup_iters=2000, warmup_ratio=0.1,
                         by_epoch=False)
    # cfg.lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
    cfg.runner = dict(type='IterBasedRunner', max_iters=320000)
    cfg.evaluation = dict(interval=30000, metric='mDice', pre_eval=True)
    cfg.log_config = dict(
        interval=100,
        hooks=[
            dict(type='TextLoggerHook', by_epoch=False),
            dict(type='TensorboardLoggerHook')
        ])

    cfg.work_dir = osp.join(log_dir,
                            osp.splitext(osp.basename(config))[0])
    cfg.resume_from = './log/ocrnet_hr48_512x512_160k_ade20k/iter_470000.pth'

    # -----------------------------build config-------------------------------------------#

    subm = []
    checkpoint = cfg.resume_from
    device = 'cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint, device=device)

    cfg.resume_from = './log/ocrnet_hr48_512x512_160k_ade20k_——nodata_clean/iter_440000(mdice_88).pth'
    checkpoint = cfg.resume_from
    model_2 = init_segmentor(cfg, checkpoint, device=device)

    img_dir=osp.join(cfg.data_root,cfg.data.test.img_dir)
    list_img=glob.glob(img_dir+'/'+img_name.replace('png','jpg'))
    # test_mask = pd.read_csv('./dataset/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    # test_mask['name'] = test_mask['name'].apply(lambda x: x)
    for i in range(len(list_img)):
        result = inference_segmentor(model, list_img[i])
        result = np.array(result)
        result = result.astype('uint8').transpose(1, 2, 0)
        # print(type(result),result.shape)
        _, result = cv2.threshold(result, 0.5, 255, cv2.THRESH_BINARY)
        # _,label=  cv2.threshold(mmcv.imread(list_img[i].replace('img_dir','ann_dir')), 0.5, 255, cv2.THRESH_BINARY)

        result_2 = inference_segmentor(model_2, list_img[i])
        result_2 = np.array(result_2)
        result_2 = result_2.astype('uint8').transpose(1, 2, 0)
        # print(type(result_2),result_2.shape)
        _, result_2 = cv2.threshold(result_2, 0.5, 255, cv2.THRESH_BINARY)
        # _, label = cv2.threshold(mmcv.imread(list_img[i].replace('img_dir', 'ann_dir')), 0.5, 255, cv2.THRESH_BINARY)

        # print(osp.basename(list[i]))

        plt.figure(num=osp.basename(list_img[i]),figsize=(20,16),tight_layout=True)
        plt.subplot(1,4,1)
        plt.imshow(mmcv.imread(list_img[i]))
        plt.subplot(1, 4, 2)
        plt.imshow(result_2)
        # plt.subplot(1, 4, 3)
        # plt.imshow(label)
        plt.subplot(1, 4, 4)
        plt.imshow(result)
        plt.show()

        # mmcv.imshow(result,osp.basename(list[i]))
        # mmcv.imwrite(result,'./tianchi_SegGame/test_result/3.0_R05K5826G4.jpg')
        # exit(0)



    pass


def data_clean():
    from mmsegmentation.my_submit import check_result
    f=open('/home/liu1227/Download/Tianchi-AlibabaCloud-Recognition-of-Surface-Structure-Rank-1-solution/code/img.log')
    list=f.readlines()
    img_list=[]
    for i in range(162):
        word = list[i]
        try:
            img_name=re.search(r'[\w]+\.png',word).group(0)
            img_list.append(img_name)
            dir=glob.glob(pathname='./dataset/**/'+img_name,recursive=True)
            print(img_name)
            [os.remove(dir[i]) for i in range(len(dir))]
        except :
            continue

    # for j in range(len(img_list)):
    #     print(img_list[j])
    #     check_result(img_list[j])
    pass

def del_img(img_name):
    for i in range(len(img_name)):
        dir = glob.glob(pathname='./dataset/**/' + img_name[i], recursive=True)
        print(img_name)
        [os.remove(dir[i]) for i in range(len(dir))]
    pass

if __name__ == '__main__':

    # batch_save_result_submit()
    index = 3
    # image = io.imread('/data/r4v74nvbbu/000022_LT.tif')
    # # print(np.unique(image))
    # image = io.imread('./dataset_sd/result_1.0/014736_LT.tif')
    # print(np.unique(image))
    batch_save_result_submit_SD()
    # check_result_2model()
    # check_result()
    # data_clean()

    # img_name=['KZ22OGOP69.png','GZNHUE8G9O.png','JTU6O855E9.png','DW2O69ZXGK.png','U9LMEKK672.png','DQHEVU09AR.png']
    # del_img(img_name)

    pass
