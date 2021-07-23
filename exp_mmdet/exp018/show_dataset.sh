EXP="exp003"

# for cv in 0 1 2 3 4
for cv in 4
do
    CONFIG=/workspace/exp_mmdet/$EXP/fold.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    OUT=/workspace/output/mmdet_$EXP\_cv$cv/log_box_map.pdf
    LOG=/workspace/output/mmdet_$EXP\_cv$cv/20210711_232056.log.json
    python /workspace/customized_mmdetection/tools/misc/browse_dataset.py\
    $CONFIG --output-dir OUT_DIR/dataset --show-interval 500\
    --cfg-options data.train.ann_file=/workspace/exp_mmdet/config/train_cv$cv.json\
    --cfg-options data.val.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json\
    --cfg-options data.test.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json
done
