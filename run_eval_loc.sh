#!/bin/bash
img_size=256
use_cam=False
use_sam=False
use_mam=False
dataset="BUSI"
test_file="data/busi_test_binary.txt"
model_name="deeplabv3"
map_size=$(expr $img_size / 32)
exp="exp8"
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
               --use_cam=$use_cam \
               --use_sam=$use_sam \
               --use_mam=$use_mam \
               --reduction_ratio=16 \
               --attention_num_conv=3 \
               --attention_kernel_size=3 \
               --map_size=$map_size \
               --adv_attack=False \
                   iou \
               --mask_thres=0.5 \
               --test_file=$test_file
    echo "Model processed: $model"
    echo "======================="
done 
