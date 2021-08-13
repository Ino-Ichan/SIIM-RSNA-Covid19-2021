EXP="exp020"
MODEL_NAME="effdet_d4"

cd /workspace/customized_mmdetection

# for cv in 0 1 2 3 4
for cv in 0
# for cv in 1 3 4
do
    CONFIG=/workspace/exp_mmdet/$EXP/fold$cv.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    python /workspace/customized_mmdetection/tools/train.py\
    $CONFIG --deterministic --work-dir $OUT_DIR
done