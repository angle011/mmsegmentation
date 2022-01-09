# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .decode_head import BaseDecodeHead
from ..utils import UpConvBlock


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


class ObjectAttentionBlock(_SelfAttentionBlock):
    """Make a OCR used SelfAttentionBlock."""

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@HEADS.register_module()
class my_OCRHead(BaseDecodeHead):
    """Object-Contextual Representations for Semantic Segmentation.

    This head is the implementation of `OCRNet
    <https://arxiv.org/abs/1909.11065>`_.

    Args:
        in_channels (int): The intermediate channels of OCR block.
        scale (int): The scale of probability map in SpatialGatherModule in
            Default: 1.
    """

    def __init__(self, ocr_channels, out_channels, skip_channels, scale=1, **kwargs):
        super(my_OCRHead, self).__init__(**kwargs)
        self.skip_channels = skip_channels
        self.ocr_channels = ocr_channels
        self.out_channels = out_channels
        self.scale = scale
        self.object_context_block = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(1,4):
            self.object_context_block.append(
                ObjectAttentionBlock(
                    self.out_channels[i],
                    self.out_channels[i]//2,
                    self.scale,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.spatial_gather_module = SpatialGatherModule(self.scale)
        self.out_channels.pop(0)
        for in_ch, skip_ch, out_ch in zip(self.ocr_channels, self.skip_channels, self.out_channels):
            self.decoder.append(DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=True))
            # print(in_ch,skip_ch,out_ch)

        self.fcn_conv = nn.ModuleList()
        for i in range(3):
            self.fcn_conv.append(ConvModule(
                self.out_channels[i],
                2,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

    def forward(self, inputs, prev_output=None):
        """Forward function."""
        # print('input','prev_output')
        # prev_output-----torch.Size([5, 2, 64, 64])
        # |------input -----------------|
        # |torch.Size([5, 48, 64, 64])  |
        # |torch.Size([5, 96, 32, 32])  |
        # |torch.Size([5, 192, 16, 16]) |
        # |torch.Size([5, 382, 8, 8])   |
        # |-----------------------------|
        # x----torch.Size([5, 270, 64, 64])
        # feats----torch.Size([5, 512, 64, 64])
        # context----torch.Size([5, 512, 2, 1])
        feats = inputs.pop()
        for i in range(len(self.decoder)):
            skip=inputs[len(inputs)-(i+1)] if i<len(inputs) else None
            feats=self.decoder[i](feats,skip)
            if i <len(self.fcn_conv):
                prev_output=self.fcn_conv[i](feats)
                context = self.spatial_gather_module(feats, prev_output)
                feats = self.object_context_block[i](feats, context)
        output = self.cls_seg(feats)

        return output

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
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

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

