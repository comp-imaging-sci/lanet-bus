#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
model_name="resnet50"
image_size=256
map_size=8
datatype="All"
exp="exp6"
num_classes=2
use_cbam=False
use_mask=False
no_channel=False
save_path="${datatype}_train/${exp}-${model_name}-cbam=${use_cbam}-no_channel=${no_channel}-size=${image_size}-cls=${num_classes}"
if [ ! -d $save_path ]; then
    mkdir $save_path
fi 

python train.py --model_name=$model_name \
                --image_size=$image_size \
                --num_classes=$num_classes \
                --batch_size=16 \
                --num_epochs=100 \
                --model_save_path="${datatype}_train/${exp}-${model_name}-cbam=${use_cbam}-no_channel=${no_channel}-size=${image_size}-cls=${num_classes}" \
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
                --attention_num_conv=3
                #--mask_annotate_file="data/mayo_patient_info.csv" \
                #--pretrained_weights="/shared/anastasio5/COVID19/ultrasound_breast_cancer/MAYO_resnet50_mask_448/best_model.pt" \

# exp 1
# resnet50_cbam_mask cbam=True mask=True no_channel=True size=256 class=2 BUSI

# exp 2
# resnet50_cbam_mask cbam=True mask=True no_channel=False size=256 class=2 BUSI

# exp 3
# resnet50_cbam cbam=True no_channel=True size=256 class=2 BUSI

# exp 4 
# resnet50_cbam cbam=True no_channel=False size=256 class=2 BUSI

# exp 5
# resnet50 size=256 class=2 BUSI

# exp 6
# resnet50 size=256 class=2 All

# exp 7 
# resnet50_cbam_mask cbam=False mask=True no_channel=True size=256 class=2 All

# exp 8
# resnet50_cbam_mask cbam=False mask=True no_channel=False size=256 class=2 All

  
