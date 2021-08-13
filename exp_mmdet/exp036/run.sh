EXP="exp036"
MODEL_NAME="gfl_yolof_x101_32x4d_c5"

for cv in 0 1 2 3 4
# for cv in 0 2
# for cv in 1 3 4
do
    CONFIG=/workspace/exp_mmdet/$EXP/fold$cv.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    python /workspace/customized_mmdetection/tools/train.py\
    $CONFIG --deterministic --work-dir $OUT_DIR
done