_base_ = [
    '../_base_/models/pspnet_unet_my.py',
    '../_base_/datasets/mydataset_TC.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

# model settings  change BN!
norm_cfg = dict(type='BN', requires_grad=True)
# # UNet
model = dict(
#     backbone=dict(norm_cfg=norm_cfg),
    decode_head=dict(norm_cfg=norm_cfg),
#     auxiliary_head=dict(norm_cfg=norm_cfg),
#     # model training and testing settings
    test_cfg=dict(crop_size=(256,256), stride=(85, 85)))
# efficientNet

# load_from = '/data/open-mmlab_log_dir/demo/log/efficientNet_UNet_mdice_78/iter_15000.pth'
lr_config = dict(policy='poly', power=0.9, min_lr=1e-7, by_epoch=False)
# lr_config = dict(policy='CosineAnnealing',  _delete_=True,min_lr=1e-7, by_epoch=False)
# lr_config = dict(policy='Step', power=0.9, min_lr=1e-6, by_epoch=False)

# log config data
evaluation = dict(interval=5000, metric='mDice')
runner = dict(max_iters=15000)
checkpoint_config = dict(interval=5000)
optimizer = dict(type='SGD', lr=8e-5)
workflow = [('train',1),('val',1)]
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ])
# data config file
data = dict(samples_per_gpu=4,#batchsize
    workers_per_gpu=4,)

# custom_hooks = [
#     dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
# ]
# mean                  std
# ([0.20626388, 0.21765834, 0.20000833], [0.10294722, 0.09351111, 0.08964583])
# [62.5, 44.8, 68.8],
#                         [13.1, 17.7, 10.1]

