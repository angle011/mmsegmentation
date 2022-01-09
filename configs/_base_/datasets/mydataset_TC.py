# dataset settings
dataset_type = 'mydata_TCDataset'
data_root = '/data/tianchi_SegGame'
img_norm_cfg = dict(
    mean=[62.5, 44.8, 68.8],
    std=[13.1, 17.7, 10.1], to_rgb=True)
img_scale = (512,512)
crop_size = (256,256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # dict(type='RandomCutMix',prob=0.7,n_holes=2,cutout_shape=[(90,90),(100,65)]),
    dict(type='RandomFlip', prob=0.5,direction='horizontal'),
    dict(type='RandomFlip', prob=0.5,direction='vertical'),
    dict(type='RandomRotate',prob=0.5,degree=(90,270)),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[62.5, 44.8, 68.8],
        std=[13.1, 17.7, 10.1],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5,  1.0, 1.25, 1.5,  2.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=10,#batchsize
    workers_per_gpu=9,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test_a',
        ann_dir='ann_dir/val2',
        pipeline=test_pipeline))

