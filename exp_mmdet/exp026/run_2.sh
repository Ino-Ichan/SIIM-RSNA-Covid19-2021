EXP="exp026"
# MODEL_NAME="cascade_rcnn_x101_64x4d_fpn_20e_coco"
MODEL_NAME="cascade_rcnn_r2_101_fpn_20e_coco"

# for cv in 1 2 3 4
# for cv in 1 3 4
for cv in 0
do
    CONFIG=/workspace/exp_mmdet/$EXP/fold$cv\_resume.py
    OUT_DIR=/workspace/output/mmdet_$EXP\_cv$cv
    python /workspace/customized_mmdetection/tools/train.py\
    $CONFIG --deterministic --work-dir $OUT_DIR
    # --cfg-options data.train.ann_file=/workspace/exp_mmdet/config/train_cv$cv.json\
    # --cfg-options data.val.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json\
    # --cfg-options data.test.ann_file=/workspace/exp_mmdet/config/val_cv$cv.json\
    # --cfg-options log_config.hooks.1.init_kwargs.name=$EXP\_cv$cv\_$MODEL_NAME\
    # --cfg-options log_config.hooks.1.init_kwargs.tags="[$EXP,$MODEL_NAME,$cv]"
done