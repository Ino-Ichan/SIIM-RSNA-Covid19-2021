# EXP="exp018"
EXP="exp029"
# MODEL_NAME="cascade_rcnn_x101_64x4d_fpn_20e_coco"
MODEL_NAME="cascade_rcnn_dcn_x101_64x4d_fpn_20e_coco"

# for cv in 1 2 3 4
# for cv in 1 3 4
for cv in 0 1 2 3 4
do
    CONFIG=/workspace/exp_mmdet/$EXP/fold$cv\_resume.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    python /workspace/customized_mmdetection/tools/train.py\
    $CONFIG --deterministic --work-dir $OUT_DIR
done