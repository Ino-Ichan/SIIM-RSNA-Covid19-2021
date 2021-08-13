
EXP="exp001"
MODEL_NAME="tf_efficientdet_d5_ap"
OUT_ROOt=/workspace/output/mmdet_$EXP
BATCH_SIZE=16

mkdir -p $OUT_ROOt

# for cv in 0 1 2 3 4
# for cv in 0 2
for cv in 0
do
    python /workspace/exp_effdet/$EXP/train.py\
    /data/ --num-classes 1 --pretrained --save-images\
    --model $MODEL_NAME \
    -b $BATCH_SIZE \
    --amp --sched cosine --lr 1e-4 --min-lr 1e-5 \
    --warmup-epochs 2 \
    --epochs 20 \
    --output $OUT_ROOt/cv$cv\
    --cv $cv\
    --exp_name $EXP\
    --model_name $MODEL_NAME\
    --df_train_path "/workspace/data/df_train_study_level_npy640_3_w_bbox_hw.csv"\
    --batch_size $BATCH_SIZE\
    --num_workers 8\
    --local_rank 0
done
