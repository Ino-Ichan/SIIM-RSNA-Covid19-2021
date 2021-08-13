# exp = "exp006"
exp = "exp032"
model_name = "retinanet_r50_fpn_1x_coco"
cv = "0"


# https://www.kaggle.com/sreevishnudamodaran/siim-mmdetection-cascadercnn-weight-bias


_base_ = "/workspace/customized_mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py"

# =============================================================
# schedule_1x
# =============================================================

# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[10, 15])

# ## Learning rate scheduler config used to register LrUpdater hook
# lr_config = dict(
#     policy='CosineAnnealing', # The policy of scheduler, also support CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py#L9.
#     by_epoch=False,
#     warmup='linear', # The warmup policy, also support `exp` and `constant`.
#     warmup_iters=500, # The number of iterations for warmup
#     warmup_ratio=0.001, # The ratio of the starting learning rate used for warmup
#     min_lr=1e-07)

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
load_from = '/workspace/output/mmdet_exp006_cv0/best_bbox_mAP_50_epoch_18.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]


# =============================================================
# Model
# =============================================================


# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
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
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        # score_thr=0.05,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))



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

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.,
        scale_limit=0.2,
        rotate_limit=15,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.0, 0.2],
        contrast_limit=[0.0, 0.2],
        p=0.5),
    dict(
        type='HueSaturationValue',
        hue_shift_limit=20,
        sat_shift_limit=20,
        val_shift_limit=0,
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='CutOut', n_holes=2, cutout_ratio=(0.2, 0.2)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
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
    samples_per_gpu=32,
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
