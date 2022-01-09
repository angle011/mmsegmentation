import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class mydata_SDDataset(CustomDataset):
    # CLASSES = ('耕地', '林地','草地','水域','城乡、工矿、居民用地','未利用土地')
    # PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    #            [4, 200, 3], [120, 120, 80],]
    CLASSES = ('耕地', '林地','水域','城乡、工矿、居民用地')
    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               ]
    def __init__(self, **kwargs):
        super(mydata_SDDataset, self).__init__(
            img_suffix='_GF.png',
            seg_map_suffix='_LT.png',
            reduce_zero_label=False,
            weight=[0.595,0.104,0.092,0.198],
            **kwargs)
        assert osp.exists(self.img_dir)