# # invert white/blackなし, v2 s, freeze bn, from exp032, hard aug adjust epoch, 1 folds, 384
# exp=exp035
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp032, hard aug adjust epoch, 5 folds, 384, bs=32*2
# exp=exp201
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp201, mid aug(-distort) aug adjust epoch, 5 folds, 384, bs=32*2
# exp=exp202
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp201, hard aug adjust epoch, 5 folds, 384, bs=32*2, few cutout holes
# exp=exp203
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 s, freeze bn, from exp030, hard aug adjust epoch
# exp=exp204
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, v2 l, freeze bn, from exp030, middle aug(-distort) adjust epoch, bs=8*8
# exp=exp205
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, b7, freeze bn, from exp030, middle aug(-distort) adjust epoch, bs=8*8
# exp=exp206
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml



# =======================
# RANZCR like model
# =======================


# # swin_base, from exp313, qishen aug, drop random crop, use_amp=True, 5 target, adamw
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp401
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # swin_base w/ dropout 0.5, from exp401, qishen aug, drop random crop, use_amp=True, 5 target, adamw
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp402
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # swin_base, from exp313, qishen aug, drop random crop, use_amp=True, 5 target, adamw
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# # exp=exp401
# exp=exp403
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # swin_base w/ dropout 0.5, from exp401, qishen aug, drop random crop, use_amp=True, 5 target, adamw
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# # exp=exp402
# exp=exp404
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# =======================
# Segmentation
# =======================

# # effb2 segmentation w/ dropout 0.5, from exp401, qishen aug, drop random crop, use_amp=True, 5 target, adamw
# loss_label:loss_seg=0.5:0.5
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp405
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # effb2 segmentation w/ dropout 0.5, from exp401, qishen aug, drop random crop, use_amp=True, 5 target, adamw,
# # loss_label:loss_seg=1:0.5
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp406
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # effb2 segmentation w/ dropout 0.5, from exp401, qishen aug, drop random crop, use_amp=True, 5 target, adamw,
# # loss_label:loss_seg=1:0.2
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp407
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, b7, freeze bn, from exp206, bs=8*8, qishen aug fine tuning
# exp=exp207
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # invert white/blackなし, b7, freeze bn, from exp206, bs=8*8, qishen aug from scratch
# exp=exp208
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # effb7, from exp313, qishen aug, drop random crop, use_amp=True, 5 target, adamw
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp408
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # eff v2 m, from exp319, qishen aug, drop random crop, use_amp=True, 5 target, adamw
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp409
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # eff v2 m, from exp319, qishen aug, drop random crop, use_amp=True, 5 target, adamw, 640 
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp410
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # eff v2 m, from exp319, qishen aug, drop random crop, use_amp=True, 5 target, adamw, 640, softmax
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp411
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # eff v2 m, from exp319, qishen aug, drop random crop, use_amp=True, 5 target, adamw, 512, another arch 
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp412
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # eff v2 m, from exp319, qishen aug, drop random crop, use_amp=True, 5 target, adamw
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp413
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # eff v2 m, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 6, lovaz+bce
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp414
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # eff v2 m, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 7, lovaz+bce
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp415
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # eff v2 m, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 6 & 7 dual, lovaz+bce
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp416
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # eff v2 m, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 5 & 6 & 7 dual, lovaz+bce
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp417
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml




# =======================
# Other models
# =======================

# # seresnet152d, from exp415, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 4, lovaz+bce
# exp=exp601
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # dm_nfnet_f0, from exp415, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 4, lovaz+bce
# exp=exp602
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # swin base p4w7 384, from exp415, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 3, lovaz+bce
# exp=exp603
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # swin base p4w7 384, from exp415, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 4, lovaz+bce
# exp=exp604
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # swin base p4w7 384, from exp603 lr変えてもう一回, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 4, lovaz+bce
# exp=exp605
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # swin base p4w7 384, from exp603 lr変えてもう一回, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 3&4, lovaz+bce
# exp=exp606
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # resnet200d 384, from exp603, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 3&4, lovaz+bce
# exp=exp607
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # resnet200d 384, from exp607 lr変えてもう一回, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 3&4, lovaz+bce
# exp=exp608
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml








# # eff v2 l, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 6, lovaz+bce
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp418
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # eff v2 l, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 7, lovaz+bce
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp419
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# # eff v2 l, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 6&7 dual, lovaz+bce
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp420
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # eff v2 s, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 6&7 dual, lovaz+bce
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp421
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# # eff v2 s, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 6&7 dual, lovaz+bce, 640
# # https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
# exp=exp422
# python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml


# eff v2 s, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 6&7 dual, lovaz+bce, fine tune
# https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
exp=exp421_2
python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml

# eff v2 s, from exp335, qishen aug, drop random crop, use_amp=True, 5 target, adamw, layer 6&7 dual, lovaz+bce, 640, fine tune
# https://www.kaggle.com/haqishen/ranzcr-1st-place-soluiton-cls-model-small-ver
exp=exp422_2
python siim-rsna-2021/exp/$exp/train.py -y siim-rsna-2021/exp/$exp/config.yaml








