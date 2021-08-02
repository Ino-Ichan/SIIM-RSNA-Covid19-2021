exp="exp001"

# rm -rf /workspace/output/yolo_$exp
# mkdir -p /workspace/output/yolo_$exp

for cv in 0 1 2 3 4
do
    python test.py --img 512 --batch 32 --exist-ok\
     --data /workspace/exp_yolo/config_000/config_$cv.yaml\
     --weights /workspace/output/yolo_$exp\_cv$cv/weights/best.pt\
     --project siim_rsna_yolov5 --name /workspace/output/yolo_val_$exp\_cv$cv\
     --conf-thres=0.001 --iou-thres=0.5 --augment
done
