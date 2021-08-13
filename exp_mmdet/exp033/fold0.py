exp = "exp033"
model_name = "effdet_d3"
cv = "0"


_base_ = "/workspace/customized_mmdetection/configs/effdet/effdet_base.py"

# =============================================================
# schedule_1x
# =============================================================

# optimizer
optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    gamma=0.5,
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[4, 8, 12, 16])

# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     min_lr_ratio=1e-5)



runner = dict(type='EpochBasedRunner', max_epochs=20)

# # fp16 settings
# fp16 = dict(loss_scale=512.)


# =============================================================
# default_runtime
# =============================================================

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            init_kwargs=dict(project='siim-rsna-covid19-2021-mmdet',
                            tags=[exp, f"cv{cv}", model_name],
                            name=f"{exp}_cv{cv}_{model_name}",
                            entity='inoichan'))
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]


# =============================================================
# Model
# =============================================================

# model = dict(
#     bbox_head=dict(num_classes=1,),
#     # training and testing settings
#     test_cfg=dict(
#         nms_pre=1000,
#         min_bbox_size=0,
#         score_thr=0.001,
#         nms=dict(type='nms', iou_threshold=0.5),
#         max_per_img=100)
# )

# model settings
model = dict(
    type='RetinaNet',
    pretrained="imagenet",
    backbone=dict(
        type='TimmBB',
        name='tf_efficientnet_b3_ns',
        out_indices=(1, 2, 3, 4),
        norm_eval=True,
        drop_rate=0.0,
        drop_path_rate=0.1,
        pad_type='same'),
    neck=dict(
        type='BIFPN',
        in_channels=[32,48,136,384],
        out_channels=160,
        start_level=1,
        stack=6,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=dict(type='BN', requires_grad=False),
        activation=dict(type='ReLU')),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=160,#256->224
        stacked_convs=4,
        feat_channels=160,#256->224
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=1.5, #2->1.5
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0)),

        # training and testing settings
        train_cfg = dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg = dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.001,
            nms=dict(type='nms', iou_thr=0.5),
            max_per_img=100)        
)





# =============================================================
# coco_detection
# =============================================================

# dataset settings
dataset_type = 'CocoDataset'
classes = ('opacity',)
image_root = '/workspace/data/train_640_2/'

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_norm_cfg = dict(
    mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='CutOut', n_holes=2, cutout_ratio=(0.2, 0.2)),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        # img_scale=(1333, 800),
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=12,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f"/workspace/exp_mmdet/config/train_cv{cv}.json",
        img_prefix=image_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f"/workspace/exp_mmdet/config/val_cv{cv}.json",
        img_prefix=image_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=f"/workspace/exp_mmdet/config/val_cv{cv}.json",
        img_prefix=image_root,
        pipeline=test_pipeline))

# evaluation = dict(interval=1, metric='bbox', iou_thrs=[0.5], save_best='bbox_mAP')
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP_50')
