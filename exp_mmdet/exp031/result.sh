EXP="exp003"

# for cv in 0 1 2 3 4
for cv in 4
do
    CONFIG=/workspace/exp_mmdet/$EXP/fold.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    OUT=/workspace/output/mmdet_$EXP\_cv$cv/log_box_map.pdf
    LOG=/workspace/output/mmdet_$EXP\_cv$cv/20210711_232056.log.json
    python /workspace/customized_mmdetection/tools/test.py\
    $CONFIG $OUT_DIR/epoch_12.pth --out $OUT_DIR/result.pkl\
    # --show-dir $OUT_DIR --show-score-thr 0.01\
    --cfg-options data.train.ann_file=/workspace/exp_mmdet/config/train_cv$cv.json\
    --cfg-options data.val.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json\
    --cfg-options data.test.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json
done

# for cv in 0 1 2 3 4
for cv in 4
do
    CONFIG=/workspace/exp_mmdet/$EXP/fold.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    OUT=/workspace/output/mmdet_$EXP\_cv$cv/log_box_map.pdf
    LOG=/workspace/output/mmdet_$EXP\_cv$cv/20210711_232056.log.json
    python /workspace/customized_mmdetection/tools/analysis_tools/analyze_results.py\
    $CONFIG $OUT_DIR/result.pkl $OUT_DIR --topk 30 --show-score-thr 0.01\
    --cfg-options data.train.ann_file=/workspace/exp_mmdet/config/train_cv$cv.json\
    --cfg-options data.val.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json\
    --cfg-options data.test.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json
done
