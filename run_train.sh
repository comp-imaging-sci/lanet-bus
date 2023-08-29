#!/bin/bash
model_name="resnet18"
image_size=256
map_size=$(expr $image_size / 32)
#map_size=$image_size
datatype="BUSI"
exp="exp2"
num_classes=2
use_cam=False
use_sam=False
use_mam=False
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
                --num_epochs=50 \
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
                --pseudo_mask_weight=$pseudo_mask_weight
                # --backbone_weights="$backbone_weight"\
                # --lanet_weights="$saliency_weight"\

# exp 1-2, size = {256, 512}
# resnet50 mask=False channel_att=True class=2 BUSI

# exp 3-4, size = {256, 512}
# resnet18 mask=False channel_att=True class=2 BUSI

# exp 5-6, size = {256, 512}
# resnet50 mask=False channel_att=True class=2 All

# exp 7-8, size = {256, 512}
# resnet18 mask=False channel_att=True class=2 All

# exp 9-10, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True class=2 BUSI (load pre-trained saliency)

# exp 11-12, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True class=2 MAYO (load pre-trained saliency)

# exp 13-14, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True class=2 BUSI (load pre-trained saliency)

# exp 15-16, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True class=2 MAYO (load pre-trained saliency)

# exp 21-22, size = {256, 512}
# resnet50 mask=False channel_att=False class=2 MAYO_bbox

# exp 23-24, size = {256, 512}
# resnet18 mask=False channel_att=False class=2 MAYO_bbox

# exp 25-26, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True class=2 MAYO_bbox (load pre-trained saliency)

# exp 27-28, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True class=2 MAYO_bbox (load pre-trained saliency)

# ablation 
# exp 41-42, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=False spatial_att=True final_att=True class=2 BUSI (load pre-trained saliency)

# exp 43-44, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=False spatial_att=True final_att=True class=2 BUSI (load pre-trained saliency)

# exp 45-46, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True spatial_att=False final_att=True class=2 BUSI (load pre-trained saliency)

# exp 47-48, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True spatial_att=False final_att=True class=2 BUSI (load pre-trained saliency)

# exp 49-50, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True spatial_att=True final_att=False class=2 BUSI (load pre-trained saliency)

# exp 51-52, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True spatial_att=True final_att=False class=2 BUSI (load pre-trained saliency)

# exp53-64, repeat exp41-52 on MAYO_bbox

# exp65-66, size = {256, 512}
# resnet50-rasaee mask=True BUSI 

# exp67-68, size = {256, 512}
# resnet18-rasaee mask=True BUSI 

# exp69-70, size = {256, 512}
# resnet50-rasaee mask=True MAYO_bbox 

# exp71-72, size = {256, 512}
# resnet18-rasaee mask=True MAYO_bbox 

# partial label exp
# exp73-74, size = {256, 512}
# resnet50 mask=False class=2 MAYO

# exp75-76, size = {256, 512}
# resnet18 mask=False class=2 MAYO 

# vs exp 11-12, 15-16

# exp 77-78, size= {256, 512}
# resnet50-cbam-mask mask=True channel_att=True spatial_att=True final_att=True class=2 BUSI_0.25 

# exp 79-80, size= {256, 512}
# resnet18-cbam-mask mask=True channel_att=True spatial_att=True final_att=True class=2 BUSI_0.25 

# exp 81-82, size= {256, 512}
# resnet50-cbam-mask mask=True channel_att=True spatial_att=True final_att=True class=2 BUSI_0.5 

# exp 83-84, size= {256, 512}
# resnet18-cbam-mask mask=True channel_att=True spatial_att=True final_att=True class=2 BUSI_0.5 

# exp85-86, size= {256, 512}
# resnet50-cbam-mask mask=True channel_att=True spatial_att=True final_att=True class=2 BUSI_0.75 

# exp 87-88, size= {256, 512}
# resnet18-cbam-mask mask=True channel_att=True spatial_att=True final_att=True class=2 BUSI_0.75 

# exp 90
# resnet18 mask=False class=2 MAYO_4000

# exp 91
# resnet50 mask=False class=2 MAYO_4000

# exp 92
# resnet18 mask=False class=2 MAYO_2000

# exp 93
# resnet50 mask=False class=2 MAYO_2000

# exp 94
# resnet18 mask=False class=2 MAYO_1000

# exp 95
# resnet50 mask=False class=2 MAYO_1000

# exp 96-99
# EB0 mask=False class=2 MAYO_full/4000/2000/1000

# exp 100-103 size=256
# UNet class=2 BUSI, BUSI_0.75, BUSI_0.5, BUSI_0.25

# exp 104 size=256
# UNet class=2 MAYO_bbox

# exp 105-108 size=256
# EB0 class=2 BUSI, BUSI_0.75, BUSI_0.5, BUSI_0.25 

# exp 109 size=256
# EB0 class=2 MAYO_bbox

# exp 110-113 size=256
# ViT class=2 BUSI, BUSI_0.75, BUSI_0.5, BUSI_0.25 

# exp 114 size=256
# ViT class=2 MAYO_bbox

# exp 115-117 size=256
# resnet50-rasaee-mask BUSI_0.75, BUSI_0.5, BUSI_0.25

# exp 118-120 size=256
# resnet18-rasaee-mask BUSI_0.75, BUSI_0.5, BUSI_0.25

# exp121 size=256
# ViT class=2 MAYO

# exp 122-124 size=256
# resnet50 class=2, BUSI_0.75, BUSI_0.5, BUSI_0.25

# exp 125-127 size=256
# resnet18 class=2, BUSI_0.75, BUSI_0.5, BUSI_0.25

# exp 128-130 size=256
# resnet18_cbam_mask, BUSI_0.75, BUSI_0.5, BUSI_0.25 

# exp 131-133 size=256
# resnet50_cbam_mask, BUSI_0.75, BUSI_0.5, BUSI_0.25 

# exp 134-136 size=256
# resnet50-cbam-mask ablation: channel_att, spatial_att, final_att class=2 MAYO 

# exp 137, 138 size=256
# deeplabv3 BUSI, MAYO_bbox
