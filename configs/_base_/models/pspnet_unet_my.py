# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='TIMMBackbone',
        model_name='efficientnet_b4',
        pretrained=True,
        init_cfg=dict(),
        channel_list=[448,256,128,64,32],
        skip_channel=[160, 56, 32, 24, 0]),
    decode_head=dict(
        type='FCNHead',
        in_channels=16,
        in_index=5,
        channels=16,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=3.0,
                class_weight=[0.16, 0.84]),
            dict(
                type='DiceLoss',
                loss_name='loss_dice',
                loss_weight=1.0,
                class_weight=[0.16, 0.84])
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=32,
        in_index=4,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256,256), stride=(85, 85)))
