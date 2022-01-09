import cv2
import mmcv

import random
import os
# calculate means and std
from tqdm import tqdm
import numpy as np
import glob
import torch
from skimage import io

import matplotlib.pyplot as plt
import matplotlib
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
import shutil

def 提取平均和方差(img_path):
    img_num = 3000  # select images
    img_h, img_w = 256, 256
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, stdevs = [], []

    img_list = glob.glob(img_path + '/*.png')
    random.shuffle(img_list)  # shuffle images

    for i in tqdm(range(img_num)):
        # img = io.imread(img_list[i])
        img = cv2.imread(img_list[i])
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)

    # imgs = imgs.astype(np.float32) / 255.

    for i in tqdm(range(3)):
        pixels = imgs[:, :, i, :].ravel()  # flatten
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    # cv2 : BGR
    # means.reverse()  # BGR --> RGB
    # stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
#     normMean = [56.859916567524365, 68.51011737542564, 73.80433398014463]
# normStd = [38.059168382666876, 27.285005832693404, 26.106226534171018]


#     normMean = [376.2212412451872, 370.425647469927, 343.1814491808395]
# normStd = [93.1336710999049, 116.8398246012674, 137.46387309520782]
# transforms.Normalize(normMean = [376.2212412451872, 370.425647469927, 343.1814491808395], normStd = [93.1336710999049, 116.8398246012674, 137.46387309520782])

def GF_vision(dst):
    rows, cols, _ = dst.shape

    for i in range(rows):
        for j in range(cols):
            for k in range(_):
                if dst[i, j, k] == 1: dst[i, j, k] = 60
                if dst[i, j, k] == 2: dst[i, j, k] = 160
                if dst[i, j, k] == 3: dst[i, j, k] = 200
    return dst


def LT_vision(dst,is_ann=False,is_result=False,is_ann_vision=False,is_cls_change=None):
    rows, cols = dst.shape
    for i in range(rows):
        for j in range(cols):
            if not is_ann:
                # if all(dst[i, j, :] == [1, 1, 1]): dst[i, j, :] = [128, 64, 128]
                # if all(dst[i, j, :] == [2, 2, 2]): dst[i, j, :] = [244, 35, 232]
                # if all(dst[i, j, :] == [3, 3, 3]): dst[i, j, :] = [70, 70, 70]
                # if all(dst[i, j, :] == [4, 4, 4]): dst[i, j, :] = [102, 102, 156]
                # if all(dst[i, j, :] == [5, 5, 5]): dst[i, j, :] = [220, 20, 60]
                pass
            if is_ann_vision:
                if dst[i, j] == 5: dst[i, j] = 255
                elif dst[i, j] == 4: dst[i, j] =60
                elif dst[i, j] == 3: dst[i, j] = 100
                elif dst[i, j] == 2: dst[i, j] = 125
                elif dst[i, j] == 1: dst[i, j] = 150
                elif dst[i, j] == 0: dst[i, j] = 220
            if is_ann:
                if dst[i, j] == 1: dst[i, j] -= 1
                if dst[i, j] == 2: dst[i, j] -= 1
                if dst[i, j] == 3: dst[i, j] -= 1
                if dst[i, j] == 4: dst[i, j] -= 1
                if dst[i, j] == 5: dst[i, j] -= 1
                if dst[i, j] == 6: dst[i, j] -= 1
            if is_result:
                a = dst[i, j]
                if dst[i, j] == 5: dst[i, j] += 1
                elif dst[i, j] == 4: dst[i, j] += 1
                elif dst[i, j] == 3: dst[i, j] += 1
                elif dst[i, j] == 2: dst[i, j] += 1
                elif dst[i, j] == 1: dst[i, j] += 1
                elif dst[i, j] == 0: dst[i, j] += 1
            if is_cls_change:
                # 0--farmland/1---forestland/2---waterland/3---city
                # if dst[i, j] == 0: dst[i, j] -= 1
                # if dst[i, j] == 1: dst[i, j] -= 1
                if dst[i, j] == 2: dst[i, j] = 255
                if dst[i, j] == 3: dst[i, j] -= 1
                if dst[i, j] == 4: dst[i, j] -= 1
                if dst[i, j] == 5: dst[i, j] = 255
    return dst


def 去重():
    list1 = []
    # dst_LT=list(dst_LT)
    for i in range(256):
        for j in range(256):
            # print(type(dst_LT[i,j,:]))
            if list(dst_LT[i, j, :]) not in list1:
                list1.append(list(dst_LT[i, j, :]))
    # list1=list(set(list1))

    print(list1)


def anasys_cls_pxiel_num(path):
    str = ("gengdi_yellow", "forest_Green", "gress_boheGreen", "water——BLUE", "城乡、工矿、居民用地", "no use ground")

    pixel_num = [0, 0, 0, 0, 0, 0]
    list=glob.glob(path)
    for i in tqdm(range(len(list))):#---------------196608000
        img = mmcv.imread(list[i])
        img=img[:,:,0]
        for j in range(img.shape[0]):
             for k in range(img.shape[1]):
                if img[j,k]==0: pixel_num[0]=pixel_num[0]+1
                elif img[j,k]==1:pixel_num[1]=pixel_num[1]+1
                elif img[j,k]==2:pixel_num[2]=pixel_num[2]+1
                elif img[j, k] ==3:pixel_num[3]=pixel_num[3]+1
                elif img[j, k] ==4:pixel_num[4]=pixel_num[4]+1
                elif img[j, k] ==5:pixel_num[5]=pixel_num[5]+1


    print(pixel_num)
    # pixel_num = [116543838 / 3000, 20597196 / 3000, 951333 / 3000, 18129710 / 3000, 39314420 / 3000, 1071503 / 3000]
    # [116543838, 20597196, 951333, 18129710, 39314420, 1071503]
    # 将全局的字体设置为黑体
    plt.rcParams['font.family'] = ['simhei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    x = np.arange(len(pixel_num))

    # 绘图 x x轴， height 高度, 默认：color="blue", width=0.8
    p1 = plt.bar(x, height=pixel_num, width=0.5, label="类别", tick_label=str)
    # 添加数据标签
    for a, b in zip(x, pixel_num):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    plt.legend()
    # 展示图形
    plt.show()

    pass


def pre_change_tiff_img(path):
    if not os.path.exists(path.replace('_tif', '')):
        os.mkdir(path.replace('_tif', ''))
    path = path + '/*.tif'
    img_list = glob.glob(path)

    for i in tqdm(range(len(img_list))):
        img = io.imread(img_list[i])
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        io.imsave(img_list[i].replace('.tif', '.png').replace('_tif', ''), img)

    pass

def pre_change_ann(path):
    if not os.path.exists(path.replace('_1.0', '')):
        os.mkdir(path.replace('_1.0', ''))
    path = path + '/*.tif'
    img_list = glob.glob(path)

    for i in tqdm(range(len(img_list))):
        mmcv.imwrite(LT_vision(mmcv.imread(img_list[i],0),is_ann=True),img_list[i].replace('_1.0', ''))
    pass

def change_ann_cls_4(path):
    if not os.path.exists(path.replace('_6', '')):
        os.mkdir(path.replace('_6', ''))
    path = path + '/*.png'
    img_list = glob.glob(path)

    for i in tqdm(range(len(img_list))):
        mmcv.imwrite(LT_vision(mmcv.imread(img_list[i],0),is_cls_change=True),img_list[i].replace('_6', ''))

    pass

def tif2png(path):
    path = path + '/*.tif'
    img_list = glob.glob(path)
    for i in tqdm(range(len(img_list))):
        os.rename(img_list[i], img_list[i].replace('tif', 'png'))

def result_check(img_name='*.png'):
    config = './configs/ocrnet/ocrnet_hr48_512x512_160k_ade20k.py'
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
    cfg.resume_from = './log/ocrnet_hr48_512x512_160k_ade20k/iter_150000.pth'
    seed = init_random_seed(seed_num)
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed
    # -----------------------------build config-------------------------------------------#

    subm = []
    checkpoint = cfg.resume_from
    device = 'cuda:0'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(cfg, checkpoint, device=device)
    img_dir = osp.join(cfg.data_root, cfg.data.val.img_dir)
    list_img = glob.glob(img_dir + '/' + img_name)
    # test_mask = pd.read_csv('./dataset/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    # test_mask['name'] = test_mask['name'].apply(lambda x: x)
    for i in range(len(list_img)):
        result = inference_segmentor(model, list_img[i])
        result = np.array(result)
        result = result.astype('uint8').squeeze()
        print(np.unique(result))
        label =LT_vision(cv2.imread(list_img[i].replace('img_dir','ann_dir').replace('GF','LT'),0),is_ann_vision=True)
        result=LT_vision(result,is_ann_vision=True)
        plt.figure(num=osp.basename(list_img[i]), figsize=(20, 16), tight_layout=True)
        plt.subplot(1, 3, 1)
        plt.imshow(io.imread(list_img[i]))
        plt.subplot(1, 3, 2)
        plt.title(label='gengdi_yellow——220|forest_Green——150|grass_boheGreen--125|water_BLUE--100|city_Purpose--60|no_use_BLACK--255')
        plt.imshow(label)
        plt.subplot(1, 3, 3)
        plt.imshow(result)

        plt.show()

        # mmcv.imshow(result,osp.basename(list[i]))
        # mmcv.imwrite(result,'./tianchi_SegGame/test_result/3.0_R05K5826G4.jpg')
        # exit(0)


pass

if __name__ == '__main__':
    path = './dataset_sd/初赛训练集/初赛训练_GF'

    # 提取平均和方差('./dataset_sd/img_dir/train')
    # pre_change_tiff_img(path='./dataset_sd/初赛训练集/初赛训练_GF_tif')
    # pre_change_ann(path='./dataset_sd/初赛训练集/初赛训练_LT（复件）_1.0')
    # tif2png(path='./dataset_sd/初赛训练集/初赛训练_LT（复件）')
    # pre_change_tiff_img(path='./dataset_sd/img_dir/val_tif')
    # pre_change_tiff_img(path='./dataset_sd/初赛A榜_GF_tif')

    # change_ann_cls_4(path='./dataset_sd/ann_dir/val_6')
    # change_ann_cls_4(path='./dataset_sd/ann_dir/train_6')
    # pre_change_ann(path='./dataset_sd/result_1.0')
    # tif2png(path='./dataset_sd/ann_dir/train_1.0')
    # anasys_cls_pxiel_num('./dataset_sd/ann_dir/val/*.png')

    # result_check()

    # plt.figure()
    # plt.subplot(1,2,1)
    img='./dataset_sd/ann_dir/val/000005_LT.png'
    a=cv2.imread(img)
    print(np.unique(a))
    # img='./dataset_sd/初赛A榜_GF/000017_GF.png'
    # plt.imshow(cv2.imread(img))
    # plt.subplot(1,2,2)
    # plt.imshow(LT_vision(cv2.imread('./dataset_sd/result/000017_LT.tif',0),is_ann_vision=True))
    # plt.show()

    pass
