#!/bin/bash
img_size=256
use_mask=False
use_cam=True
use_sam=True
use_mam=True
dataset="BUSI"
model_name="resnet18_cbam_mask"
map_size=$(expr $img_size / 32)
exp="exp1"

declare -a StringArray=( "${exp}-${model_name}-use_cam=${use_cam}-use_sam=${use_sam}-use_mam=${use_mam}-size=${img_size}/best_model.pt" )
for model in ${StringArray[@]};
do
    full_path="${dataset}_train/$model"
    python eval.py --model_name=$model_name \
              --num_classes=2 \
              --model_weights=$full_path \
              --image_size=$img_size \
              --device="cuda:1" \
              --dataset=$dataset \
              --multi_gpu=False \
              --map_size=$map_size \
              --use_cam=$use_cam \
              --use_sam=$use_sam \
              --use_mam=$use_mam \
              --reduction_ratio=16 \
              --attention_num_conv=3 \
              --attention_kernel_size=3 \
              --return_mask=True \
              image2mask \
              --seg_image_list="data/draw_mask_sample.txt" \
              --mask_save_file="test/busi_pred_mask_sample.png" \
              --mask_thres=0.5
    
    echo "Model processed: $model"
    echo "======================="
done 
