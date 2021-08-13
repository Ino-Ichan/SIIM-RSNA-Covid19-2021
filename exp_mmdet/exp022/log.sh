EXP="exp003"

# for cv in 0 1 2 3 4
for cv in 4
do
    OUT=/workspace/output/mmdet_$EXP\_cv$cv/log_box_map.pdf
    LOG=/workspace/output/mmdet_$EXP\_cv$cv/20210711_232056.log.json
    python /workspace/customized_mmdetection/tools/analysis_tools/analyze_logs.py plot_curve\
    $LOG --out $OUT --keys bbox_mAP bbox_mAP_50
done