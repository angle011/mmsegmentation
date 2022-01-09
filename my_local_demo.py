import numpy as np
import torch
import os
import glob
import pandas as pd
import mmcv
from tqdm import tqdm
import cv2
import os.path as osp

def demo():
    from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
    from mmseg.core.evaluation import get_palette
    #
    config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    checkpoint_file = '/home/liu1227/Download/mmSeg_demo/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    # test a single image
    img = 'demo/demo.png'
    result = inference_segmentor(model, img)
    # show the results
    show_result_pyplot(model, img, result, get_palette('cityscapes'))



def data_inspect():
    list=glob.glob('/data/tianchi_SegGame/ann_dir/val1/*')
    for i in tqdm(range(len(list))):
        img =cv2.imread(list[i])
        _, dst = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
        _, dst = cv2.threshold(dst, 127, 1, cv2.THRESH_BINARY)
        img=dst[:,:,0]
        # cv2.imwrite(list[i],img)
        cv2.imwrite(list[i].replace('val1','val1').replace('.jpg', '.png'),img)
        # os.rename(list[i],list[i].replace('jpg','png'))
        # cv2.imwrite(list[i].replace('train1', 'train').replace('.jpg', '.png'), cv2.imread(list[i]))
    pass

def net_result_check(pic_name):
    img =cv2.imread(osp.join('./tianchi_SegGame/img_dir/val',pic_name))
    mask=cv2.imread(osp.join('./tianchi_SegGame/ann_dir/val',pic_name))
    _, dst = cv2.threshold(mask, 0.5, 128, cv2.THRESH_BINARY)
    mmcv.imshow(cv2.addWeighted(img,1,dst,1,0))

    pass

def submit():
    from torchvision import transforms as T
    # trfm = T.Compose([
    #     T.ToPILImage(),
    #     T.Resize(256,256),
    #     T.ToTensor(),
    #     T.Normalize([0.625, 0.448, 0.688],
    #                 [0.131, 0.177, 0.101]),
    # ])
    subm=[]
    test_mask = pd.read_csv('./tianchi_SegGame/test_a_samplesubmit.csv', sep='\t', names=['name', 'mask'])
    test_mask['name'] = test_mask['name'].apply(lambda x: '数据集/test_a/' + x)

    for idx, name in enumerate(tqdm(test_mask['name'].iloc[:])):
        image = cv2.imread(name)
        # image = trfm(image)
        with torch.no_grad():
            # image = image.to('cuda')
            score = model(image)['out'][0][0]
            score_sigmoid = score.sigmoid().cpu().np()
            score_sigmoid = (score_sigmoid > 0.5).astype(np.uint8)
            score_sigmoid = cv2.resize(score_sigmoid, (512, 512))

            # break
        subm.append([name.split('/')[-1], rle_encode(score_sigmoid)])
    subm = pd.DataFrame(subm)
    subm.to_csv('./tmp.csv', index=None, header=None, sep='\t')
    pass

def show_result_pyplot(model,
                           img,
                           result,
                           palette=None,
                           fig_size=(15, 10),
                           opacity=0.5,
                           title='',
                           block=True,isshow=False):
        """Visualize the segmentation results on the image.

        Args:
            model (nn.Module): The loaded segmentor.
            img (str or np.ndarray): Image filename or loaded image.
            result (list): The segmentation result.
            palette (list[list[int]]] | None): The palette of segmentation
                map. If None is given, random palette will be generated.
                Default: None
            fig_size (tuple): Figure size of the pyplot figure.
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
            title (str): The title of pyplot figure.
                Default is ''.
            block (bool): Whether to block the pyplot figure.
                Default is True.
        """
        if hasattr(model, 'module'):
            model = model.module
        img = model.show_result(
            img, result, palette=palette, show=False, opacity=opacity)
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.title(title)
        plt.tight_layout()
        plt.show(block=block)

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
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

# 将图片编码为rle格式
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def batch_save_result_submit():
    from argparse import ArgumentParser
    from mmseg.apis import inference_segmentor, init_segmentor  # , show_result_pyplot



    subm=[]
    parser = ArgumentParser()
    parser.add_argument('--img', default='./tianchi_SegGame/test_a/*.jpg', help='Image file')
    parser.add_argument('--config', default='./configs/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes_my.py',
                        help='Config file')
    parser.add_argument('--checkpoint', default='/home/liu1227/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmsegmentation/tools/work_dirs/ocrnet_hr18_512x1024_40k_cityscapes_my/iter_100000.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='mydata_TCDataset',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # test a single image
    list_img=glob.glob(args.img)
    for i in tqdm(range(len(list_img))):
        result = inference_segmentor(model, list_img[i])
        result=np.array(result)
        result=result.astype('uint8').transpose(1,2,0)
        # print(type(result),result.shape)
        _,result=cv2.threshold(result,0.5,255,cv2.THRESH_BINARY)
        subm.append([osp.basename(list_img[i]), rle_encode(result)])
    subm = pd.DataFrame(subm)
    subm.to_csv('./tianchi_SegGame/tmp.csv', index=None, header=None, sep='\t')

    # mmcv.imshow(result)
    # show the results
    # my_show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     get_palette(args.palette),
    #     opacity=args.opacity)


    pass

def rle_load_test(path):
    train_mask = pd.read_csv(path, sep='\t', names=['name', 'mask'])
    # train_mask['name'] = train_mask['name'].apply(lambda x: '数据集/train/' + x)

    mask = rle_decode(train_mask['mask'].iloc[0])
    print(type(mask))
    _,mask=cv2.threshold(mask,0.5,255,cv2.THRESH_BINARY)
    mmcv.imshow(mask)
    pass

def batch_save_result_submit():
    from argparse import ArgumentParser
    from mmseg.apis import inference_segmentor, init_segmentor  # , show_result_pyplot



    subm=[]
    parser = ArgumentParser()
    parser.add_argument('--img', default='./dataset/test_a/*.jpg', help='Image file')
    parser.add_argument('--config', default='./configs/ocrnet/ocrnet_hr18_512x1024_40k_cityscapes_my.py',
                        help='Config file')
    parser.add_argument('--checkpoint', default='/home/liu1227/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmsegmentation/tools/work_dirs/ocrnet_hr18_512x1024_40k_cityscapes_my/iter_100000.pth',
                        help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='mydata_TCDataset',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
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
    subm.to_csv('./dataset/tmp_HRNet.csv', index=None, header=None, sep='\t')

    # mmcv.imshow(result)
    # show the results
    # my_show_result_pyplot(
    #     model,
    #     args.img,
    #     result,
    #     get_palette(args.palette),
    #     opacity=args.opacity)


    pass

def _test_TIMMbackbone():
    from mmseg.models.backbones import TIMMBackbone
    from mmseg.utils import get_root_logger

    model = TIMMBackbone(model_name='efficientnet_b4', pretrained=True)
    logger = get_root_logger(log_file='./log/_test_TIMMbackbone.log')
    logger.info(model)

    pass


def check_pth_file():
    dict=torch.load('/data/ocrnet_hr48_512x512_160k_ade20k_20200615_184705-a073726d.pth')
    # print(type(dict))
    for k,v in dict.items():
        if k !='meta':
            for key,value in v.items():
                print('%s:  %s'%(key,value.shape))
        # exit()
    pass

if __name__ =='__main__':
    # tes()
    # data_inspect()
    # img =cv2.imread('/data/tianchi_SegGame/ann_dir/train2/0ACD1PCJJ1.jpg')
    # print()
    # net_result_check('0BMSWJQU5M.png')
    # submit()

    # batch_save_result_submit()

    # check_pth_file()
    # rle_load_test('./tianchi_SegGame/tmp.csv')
    # _test_TIMMbackbone()
    # anasys_cls_pxiel_num('./tianchi_SegGame/ann_dir/train/*.png')

    os.rename('/home/liu1227/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmsegmentation/000005_GF.tif','/home/liu1227/anaconda3/envs/open-mmlab/lib/python3.7/site-packages/mmsegmentation/000005_GF.png')
    pass