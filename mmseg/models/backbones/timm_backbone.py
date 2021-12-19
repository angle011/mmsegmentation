# Copyright (c) OpenMMLab. All rights reserved.
try:
    import timm
except ImportError:
    timm = None

from mmcv.cnn.bricks.registry import NORM_LAYERS
from mmcv.runner import BaseModule
import torch.nn as nn
import torch
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer,build_upsample_layer)
from ..builder import BACKBONES
import torch.nn.functional as F

@BACKBONES.register_module()
class TIMMBackbone(BaseModule):
    """Wrapper to use backbones from timm library. More details can be found in
    `timm <https://github.com/rwightman/pytorch-image-models>`_ .

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name,
        features_only=True,
        pretrained=True,
        checkpoint_path='',
        in_channels=3,
        init_cfg=None,
        channel_list=None,
        skip_channel=None,
        **kwargs,
    ):
        if timm is None:
            raise RuntimeError('timm is not installed')
        super(TIMMBackbone, self).__init__(init_cfg)
        if 'norm_layer' in kwargs:
            kwargs['norm_layer'] = NORM_LAYERS.get(kwargs['norm_layer'])
        self.timm_model = timm.create_model(
            model_name=model_name,
            features_only=features_only,
            pretrained=pretrained,
            in_chans=in_channels,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

        # Make unused parameters None
        self.timm_model.global_pool = None
        self.timm_model.fc = None
        self.timm_model.classifier = None

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True
        self.decoder_EFF = nn.ModuleList()
        self.channel=channel_list #[24,32,56,160,448]
        self.skip_channel=skip_channel
        self.out_channel =[]
        self.out_channel=[ 2**(i+4) for i in reversed(range(len(channel_list)))]
        kwarg = dict(use_batchnorm=True, attention_type=None)
        for in_ch, skip_ch, out_ch in zip(self.channel, self.skip_channel, self.out_channel):
            self.decoder_EFF.append(DecoderBlock(in_ch, skip_ch, out_ch, **kwarg))
            # print(in_ch,skip_ch,out_ch)

    def forward(self, x):
        features = self.timm_model(x)
        # [print(features[i].shape) for i in range(len(features))]
        # 0-----torch.Size([12, 24, 128, 128])
        # 1-----torch.Size([12, 32, 64, 64])
        # 2-----torch.Size([12, 56, 32, 32])
        # 3-----torch.Size([12, 160, 16, 16])
        # 4-----torch.Size([12, 448, 8, 8])
        #
        x = features.pop(-1)
        decode_list=[x]
        for i in range(len(self.decoder_EFF)):
            skip = features[len(features)-(i+1)] if i<len(features) else None
            x = self.decoder_EFF[i](x,skip)  # decode--(x，skip)--x.upsample concat  skip--
            decode_list.append(x)
        return decode_list

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)
