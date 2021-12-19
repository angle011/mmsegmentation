import os.path as osp

import mmcv
import numpy as np
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class mydata_TCDataset(CustomDataset):
    CLASSES = ('background', 'change')
    PALETTE = [[0,0,0], [255,255,255]]
    def __init__(self, **kwargs):
        super(mydata_TCDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)