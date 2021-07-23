# # baseline, retinanet
# exp=exp002
# sh /workspace/exp_mmdet/$exp/run.sh


# # cascade rcnn
# exp=exp003
# sh /workspace/exp_mmdet/$exp/run.sh

# # retinanet, parameter調整
# exp=exp004
# sh /workspace/exp_mmdet/$exp/run.sh


# # cascade, parameter調整
# exp=exp005
# sh /workspace/exp_mmdet/$exp/run.sh


# # retinanet, parameter調整, 5fold
# exp=exp006
# sh /workspace/exp_mmdet/$exp/run.sh


# # cascade, parameter調整, 5fold
# exp=exp007
# sh /workspace/exp_mmdet/$exp/run.sh


# # cascade, parameter調整, 5fold, step gamma=0.5
# exp=exp008
# sh /workspace/exp_mmdet/$exp/run.sh

# # cascade, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch
# exp=exp009
# sh /workspace/exp_mmdet/$exp/run.sh

# # # cascade, parameter調整, 5fold, step gamma=0.5, img_size=512
# # exp=exp010
# # sh /workspace/exp_mmdet/$exp/run.sh

# # cascade, parameter調整, 5fold, step gamma=0.5, min-max norm, img_size=800
# exp=exp011
# sh /workspace/exp_mmdet/$exp/run.sh

# # cascade, parameter調整, 5fold, step gamma=0.5, min-max norm, aug
# exp=exp012
# sh /workspace/exp_mmdet/$exp/run.sh

# # cascade resnext101, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch
# exp=exp013
# sh /workspace/exp_mmdet/$exp/run.sh

# # cascade resnest101, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch
# exp=exp014
# sh /workspace/exp_mmdet/$exp/run.sh

# # cascade, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch, coco pretrained
# exp=exp015
# sh /workspace/exp_mmdet/$exp/run.sh

# # cascade resnext101, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch, other fold
# exp=exp013
# sh /workspace/exp_mmdet/$exp/run.sh


# # gfl X-101-32x4d-dcnv2, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch
# exp=exp016
# sh /workspace/exp_mmdet/$exp/run.sh


# # cascade resnest101, parameter調整, fold0, step gamma=0.5, min-max norm, long epoch
# exp=exp014
# sh /workspace/exp_mmdet/$exp/run.sh


# # vfnet_x101_64x4d_fpn_mstrain_2x_coco, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch
# exp=exp017
# sh /workspace/exp_mmdet/$exp/run.sh


# # cascade resnext101 dcn, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch
# exp=exp018
# sh /workspace/exp_mmdet/$exp/run.sh

# # cascade resnext101 dcn, parameter調整, 5fold, step gamma=0.5, min-max norm
# exp=exp018
# sh /workspace/exp_mmdet/$exp/run_2.sh

# # cascade swin base p4w7, parameter調整, 5fold, step gamma=0.5, min-max norm, long epoch
# exp=exp019
# sh /workspace/exp_mmdet/$exp/run.sh

# vfnet_x101_64x4d_fpn_mstrain_2x_coco, parameter調整, 1 2 3 4 fold, step gamma=0.5, min-max norm, long epoch
exp=exp017
sh /workspace/exp_mmdet/$exp/run.sh


















