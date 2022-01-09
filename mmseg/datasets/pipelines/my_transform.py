# Copyright (c) OpenMMLab. All rights reserved.
import cv2
import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random

from ..builder import PIPELINES


@PIPELINES.register_module()
class RandomCutMix(object):
    """CutOut operation.

    Randomly drop some regions of image used in
    `Cutout <https://arxiv.org/abs/1708.04552>`_.
    Args:
        prob (float): cutout probability.
        n_holes (int | tuple[int, int]): Number of regions to be dropped.
            If it is given as a list, number of holes will be randomly
            selected from the closed interval [`n_holes[0]`, `n_holes[1]`].
        cutout_shape (tuple[int, int] | list[tuple[int, int]]): The candidate
            shape of dropped regions. It can be `tuple[int, int]` to use a
            fixed cutout shape, or `list[tuple[int, int]]` to randomly choose
            shape from the list.
        cutout_ratio (tuple[float, float] | list[tuple[float, float]]): The
            candidate ratio of dropped regions. It can be `tuple[float, float]`
            to use a fixed ratio or `list[tuple[float, float]]` to randomly
            choose ratio from the list. Please note that `cutout_shape`
            and `cutout_ratio` cannot be both given at the same time.
        fill_in (tuple[float, float, float] | tuple[int, int, int]): The value
            of pixel to fill in the dropped regions. Default: (0, 0, 0).
        seg_fill_in (int): The labels of pixel to fill in the dropped regions.
            If seg_fill_in is None, skip. Default: None.
    """

    def __init__(self,
                 prob,
                 n_holes,
                 cutout_shape=None,
                 cutout_ratio=None,
                 fill_in=(0, 0, 0),
                 seg_fill_in=None):

        assert 0 <= prob and prob <= 1
        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        if seg_fill_in is not None:
            assert (isinstance(seg_fill_in, int) and 0 <= seg_fill_in
                    and seg_fill_in <= 255)
        self.prob = prob
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.seg_fill_in = seg_fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]

    def get_crop_bbox_(self, img, crop_size):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to drop some regions of image."""
        cutout = True if np.random.rand() < self.prob else False
        if cutout:
            h, w, c = results['img'].shape
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
            for _ in range(n_holes):
                x1 = np.random.randint(0, w)
                y1 = np.random.randint(0, h)
                index = np.random.randint(0, len(self.candidates))
                if not self.with_ratio:
                    cutout_w, cutout_h = self.candidates[index]
                else:
                    cutout_w = int(self.candidates[index][0] * w)
                    cutout_h = int(self.candidates[index][1] * h)

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                # crop---img/label
                # print('(cutout_w,cutout_h)',(cutout_w,cutout_h))
                # print('y1:y2, x1:x2',y1,y2, x1,x2)
                # print(results['img'])
                # print(results['cut_img'].shape)
                img = results['cut_img']
                crop_bbox = self.get_crop_bbox_(img, (y2 - y1, x2 - x1))
                img = self.crop(img, crop_bbox)
                results['img'][y1:y2, x1:x2, :] = img
                ann = results['cut_ann']
                crop_bbox = self.get_crop_bbox_(ann, (y2 - y1, x2 - x1))
                ann = self.crop(ann, crop_bbox)
                # print(ann.shape)
                ann = ann[:, :, 0]
                # print(ann.shape)
                for key in results.get('seg_fields', []):
                    results[key][y1:y2, x1:x2] = ann

                # result
                # mmcv.imshow(results['img'])
                # _, des = cv2.threshold(results['gt_semantic_seg'], 0.5, 255, cv2.THRESH_BINARY)
                # mmcv.imshow(des)

        del  results['cut_img'], results['cut_ann']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'n_holes={self.n_holes}, '
        repr_str += (f'cutout_ratio={self.candidates}, ' if self.with_ratio
                     else f'cutout_shape={self.candidates}, ')
        repr_str += f'fill_in={self.fill_in}, '
        repr_str += f'seg_fill_in={self.seg_fill_in})'
        return repr_str
