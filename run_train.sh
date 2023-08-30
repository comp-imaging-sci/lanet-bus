#!/bin/bash
model_name="resnet18_cbam_mask"
image_size=256
map_size=$(expr $image_size / 32)
#map_size=$image_size
datatype="BUSI"
exp="exp1"
num_classes=2
use_cam=True
use_sam=True
use_mam=True
pseudo_conf=0.8
pseudo_mask_weight=0.1
save_path="${datatype}_train/${exp}-${model_name}-use_cam=${use_cam}-use_sam=${use_sam}-use_mam=${use_mam}-size=${image_size}"

if [ ! -d $save_path ]; then
    mkdir $save_path
fi 

python train.py --model_name=$model_name \
                --image_size=$image_size \
                --num_classes=$num_classes \
                --batch_size=12 \
                --num_epochs=20 \
                --model_save_path=$save_path\
                --device="cuda:0" \
                --lr=0.0003 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset=$datatype \
                --num_gpus=1 \
                --dilute_mask=0 \
                --use_cam=$use_cam \
                --use_sam=$use_sam \
                --use_mam=$use_mam \
                --reduction_ratio=16 \
                --map_size=$map_size \
                --attention_kernel_size=3 \
                --attention_num_conv=3 \
                --mask_weight=1\
                --pseudo_conf=$pseudo_conf \
                --pseudo_mask_weight=$pseudo_mask_weight \
                # --lanet_weights="$saliency_weight"\
                # --backbone_weights="$backbone_weight"\
