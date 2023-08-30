#!/bin/bash
model_name="resnet18_cbam_mask"
image_size=256
map_size=$(expr $image_size / 32)
datatype="BUSI"
exp="exp2"
num_classes=2
use_cam=True
use_sam=True
use_mam=True
save_path="${datatype}_train/${exp}-pretrain_lanet-${model_name}-use_cam=${use_cam}-use_sam=${use_sam}-use_mam=${use_mam}-size=${image_size}"

if [ ! -d $save_path ]; then
    mkdir $save_path
fi 

python train_lanet.py --model_name=$model_name \
                --image_size=$image_size \
                --num_classes=$num_classes \
                --batch_size=12 \
                --num_epochs=10 \
                --model_save_path=$save_path \
                --device="cuda:0" \
                --lr=0.0001 \
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
                # --backbone_weights="$backbone_weight"\
                # --lanet_weights=$lanet_weight\
