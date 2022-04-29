#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
model_name="resnet50_cbam_mask"
image_size=512
map_size=$(expr $image_size / 32)
datatype="MAYO"
exp="exp12"
num_classes=2
use_cbam=False
use_mask=True
no_channel=False
save_path="${datatype}_train/${exp}-${model_name}-cbam=${use_cbam}-mask=${use_mask}-no_channel=${no_channel}-size=${image_size}-cls=${num_classes}"

if [ ! -d $save_path ]; then
    mkdir $save_path
fi 

backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp6-resnet50-cbam=False-mask=False-no_channel=False-size=512-cls=2/best_model.pt"

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
                --use_cbam=$use_cbam \
                --use_mask=$use_mask \
                --no_channel=$use_channel \
                --reduction_ratio=16 \
                --map_size=$map_size \
                --attention_kernel_size=3 \
                --attention_num_conv=3 \
                --backbone_weights="$backbone_weight"\
                --mask_weight=1
                #--mask_annotate_file="data/mayo_patient_info.csv" \
                #--pretrained_weights="/shared/anastasio5/COVID19/ultrasound_breast_cancer/MAYO_resnet50_mask_448/best_model.pt" \

# exp 1-2, size = {256, 512}
# resnet50 cbam=False mask=False no_channel=False class=2 BUSI

# exp 3-4, size = {256, 512}
# resnet18 cbam=False mask=False no_channel=False class=2 BUSI

# exp 5-6, size = {256, 512}
# resnet50 cbam=False mask=False no_channel=False class=2 All

# exp 7-8, size = {256, 512}
# resnet18 cbam=False mask=False no_channel=False class=2 All

# exp 9-10, size = {256, 512}
# resnet50-cbam-mask cbam=False mask=True no_channel=False class=2 BUSI

# exp 11-12, size = {256, 512}
# resnet50-cbam-mask cbam=False mask=True no_channel=False class=2 MAYO

