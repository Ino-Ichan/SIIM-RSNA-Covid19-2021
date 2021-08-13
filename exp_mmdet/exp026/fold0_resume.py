exp = "exp026"
model_name = "cascade_rcnn_r2_101_fpn_20e_coco"
cv = "0"

_base_ = "/workspace/customized_mmdetection/configs/res2net/cascade_rcnn_r2_101_fpn_20e_coco.py"


# =============================================================
# schedule_1x
# =============================================================

# optimizer
optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    gamma=0.5,
    warmup_iters=1,
    warmup_ratio=0.001,
    step=[5, 12, 16])

# lr_config = dict(
#     policy='CosineAnnealing',
#     by_epoch=False,
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     min_lr_ratio=1e-5)



runner = dict(type='EpochBasedRunner', max_epochs=10)

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
load_from = '/workspace/output/mmdet_exp024_cv0/best_bbox_mAP_50_epoch_13.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]


# =============================================================
# Model
# =============================================================


# model settings
model = dict(
    # backbone=dict(
    #     dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
    #     stage_with_dcn=(False, True, True, True)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            # score_thr=0.05,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100))
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
