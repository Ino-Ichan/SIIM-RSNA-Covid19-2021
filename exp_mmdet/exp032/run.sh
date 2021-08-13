# EXP="exp006"
EXP="exp032"
MODEL_NAME="retinanet_r50_fpn_1x_coco"

export EXP
export MODEL_NAME

for cv in 0 1 2 3 4
# for cv in 0 4
do
    export cv
    CONFIG=/workspace/exp_mmdet/$EXP/fold$cv\_resume.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    NAME=$EXP\_cv$cv\_$MODEL_NAME
    python /workspace/customized_mmdetection/tools/train.py\
    $CONFIG --deterministic --work-dir $OUT_DIR
done