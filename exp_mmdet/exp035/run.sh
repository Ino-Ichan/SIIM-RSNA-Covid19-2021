EXP="exp035"
MODEL_NAME="retinanet_r101_fpn_1x_coco"

export EXP
export MODEL_NAME

for cv in 0 1 2 3 4
# for cv in 0 4
do
    export cv
    CONFIG=/workspace/exp_mmdet/$EXP/fold$cv.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    NAME=$EXP\_cv$cv\_$MODEL_NAME
    python /workspace/customized_mmdetection/tools/train.py\
    $CONFIG --deterministic --work-dir $OUT_DIR
    # --cfg-options data.train.ann_file=/workspace/exp_mmdet/config/train_cv$cv.json\
    # --cfg-options data.val.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json\
    # --cfg-options data.test.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json
done