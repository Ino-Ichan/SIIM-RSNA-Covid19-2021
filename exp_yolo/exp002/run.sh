exp="exp002"

# for cv in 0 1 2 3 4
for cv in 0
do
    python train_mAP05.py --img 640 --batch 16 --epochs 50\
     --data /workspace/exp_yolo/config_000/config_$cv.yaml\
     --cfg models/hub/yolov5l6.yaml --weights weights/yolov5l6.pt\
     --hyp data/hyp.finetune.yaml --device 0\
     --project siim_rsna_yolov5 --name /workspace/output/yolo_$exp\_cv$cv
done
