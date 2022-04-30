#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
model_name="resnet50_cbam_mask"
image_size=512
map_size=$(expr $image_size / 32)
datatype="BUSI"
exp="exp10"
num_classes=2
use_mask=True
channel_att=True
save_path="${datatype}_train/${exp}-${model_name}-mask=${use_mask}-channel_att=${channel_att}-size=${image_size}-cls=${num_classes}"

if [ ! -d $save_path ]; then
    mkdir $save_path
fi 


saliency_weight=""
backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp6-resnet50-mask=False-channel_att=False-size=512-cls=2/best_model.pt"
#backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp5-resnet50-mask=False-channel_att=False-size=256-cls=2/best_model.pt"
#backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp7-resnet18-mask=False-channel_att=False-size=256-cls=2/best_model.pt"
#backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp8-resnet18-mask=False-channel_att=False-size=512-cls=2/best_model.pt"

python train.py --model_name=$model_name \
                --image_size=$image_size \
                --num_classes=$num_classes \
                --batch_size=12 \
                --num_epochs=100 \
                --model_save_path=$save_path\
                --device="cuda:0" \
                --lr=0.0001 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset=$datatype \
                --num_gpus=1 \
                --dilute_mask=0 \
                --use_mask=$use_mask \
                --channel_att=$channel_att \
                --reduction_ratio=16 \
                --map_size=$map_size \
                --attention_kernel_size=3 \
                --attention_num_conv=3 \
                --backbone_weights="$backbone_weight"\
                --saliency_weights=$saliency_weight\
                --mask_weight=1
                #--mask_annotate_file="data/mayo_patient_info.csv" \
                #--pretrained_weights="/shared/anastasio5/COVID19/ultrasound_breast_cancer/MAYO_resnet50_mask_448/best_model.pt" \

# exp 1-2, size = {256, 512}
# resnet50 mask=False channel_att=True class=2 BUSI

# exp 3-4, size = {256, 512}
# resnet18 mask=False channel_att=True class=2 BUSI

# exp 5-6, size = {256, 512}
# resnet50 mask=False channel_att=True class=2 All

# exp 7-8, size = {256, 512}
# resnet18 mask=False channel_att=True class=2 All

# exp 9-10, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True class=2 BUSI

# exp 11-12, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True class=2 MAYO

# exp 13-14, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True class=2 BUSI

# exp 15-16, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True class=2 MAYO
