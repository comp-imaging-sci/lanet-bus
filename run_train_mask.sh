#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
model_name="resnet18_cbam_mask"
image_size=512
map_size=$(expr $image_size / 32)
datatype="All"
exp="exp40"
num_classes=2
use_mask=True
channel_att=True
spatial_att=True
final_att=False
save_path="${datatype}_train/${exp}-${model_name}-mask=${use_mask}-channel_att=${channel_att}-spatial_att=${spatial_att}-final_att=${final_att}-size=${image_size}-cls=${num_classes}"

if [ ! -d $save_path ]; then
    mkdir $save_path
fi 

# saliency_weights="test/best_model.pt"
# backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp6-resnet50-mask=False-channel_att=False-size=512-cls=2/best_model_1.pt"
# backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp5-resnet50-mask=False-channel_att=False-size=256-cls=2/best_model_1.pt"
#backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp7-resnet18-mask=False-channel_att=False-size=256-cls=2/best_model_1.pt"
#backbone_weight="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train/exp8-resnet18-mask=False-channel_att=False-size=512-cls=2/best_model_1.pt"

python train_mask.py --model_name=$model_name \
                --image_size=$image_size \
                --num_classes=$num_classes \
                --batch_size=12 \
                --num_epochs=120 \
                --model_save_path=$save_path \
                --device="cuda:0" \
                --lr=0.0001 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset=$datatype \
                --num_gpus=1 \
                --dilute_mask=0 \
                --use_mask=$use_mask \
                --channel_att=$channel_att \
                --spatial_att=$spatial_att \
                --final_att=$final_att \
                --reduction_ratio=16 \
                --map_size=$map_size \
                --attention_kernel_size=3 \
                --attention_num_conv=3 \
                --mask_weight=1 \
                # --backbone_weights="$backbone_weight"\
                # --saliency_weights=$saliency_weight\
                #--mask_annotate_file="data/mayo_patient_info.csv" \
                #--pretrained_weights="/shared/anastasio5/COVID19/ultrasound_breast_cancer/MAYO_resnet50_mask_448/best_model.pt" \


# exp 17-18, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True spatial_att=True final_att=True class=2 All

# exp 19-20, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True spatial_att=True final_att=True class=2 All

# exp 29-30, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=False spatial_att=True final_att=True class=2 All

# exp 31-32, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=False spatial_att=True final_att=True class=2 All

# exp 33-34, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True spatial_att=False final_att=True class=2 All

# exp 35-36, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True spatial_att=False final_att=True class=2 All

# exp 37-38, size = {256, 512}
# resnet50-cbam-mask mask=True channel_att=True spatial_att=True final_att=False class=2 All

# exp 39-40, size = {256, 512}
# resnet18-cbam-mask mask=True channel_att=True spatial_att=True final_att=False class=2 All
