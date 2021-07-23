EXP="exp004"

# for cv in 0 1 2 3 4
for cv in 0 4
do
    CONFIG=/workspace/exp_mmdet/$EXP/fold.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    python /workspace/customized_mmdetection/tools/train.py\
    $CONFIG --deterministic --work-dir $OUT_DIR\
    --cfg-options data.train.ann_file=/workspace/exp_mmdet/config/train_cv$cv.json\
    --cfg-options data.val.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json\
    --cfg-options data.test.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json
done