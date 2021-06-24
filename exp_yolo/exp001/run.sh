exp="exp001"

# rm -rf /workspace/output/yolo_$exp
# mkdir -p /workspace/output/yolo_$exp

# for cv in 0 1 2 3 4
for cv in 2 3 4
do
    python train_mAP05.py --img 512 --batch 32 --epochs 50\
     --data /workspace/exp_yolo/config_000/config_$cv.yaml\
     --cfg models/hub/yolov5x6.yaml --weights weights/yolov5x6.pt\
     --project siim_rsna_yolov5 --name /workspace/output/yolo_$exp\_cv$cv
done
