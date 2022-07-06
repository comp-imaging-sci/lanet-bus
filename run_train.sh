#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
model_name="resnet50_cbam_mask"
image_size=512
map_size=$(expr $image_size / 32)
datatype="MAYO_bbox"
exp="exp62"
num_classes=2
use_mask=True
channel_att=True
spatial_att=True
final_att=False
save_path="${datatype}_train/${exp}-${model_name}-mask=${use_mask}-channel_att=${channel_att}-spatial_att=${spatial_att}-final_att=${final_att}-size=${image_size}-cls=${num_classes}"

if [ ! -d $save_path ]; then
    mkdir $save_path
fi 

ref_path="/shared/anastasio5/COVID19/ultrasound_breast_cancer/All_train"
#backbone_weight="${ref_path}/exp5-resnet50-mask=False-channel_att=False-size=256-cls=2/best_model_1.pt" #  res50,256
backbone_weight="${ref_path}/exp6-resnet50-mask=False-channel_att=False-size=512-cls=2/best_model_1.pt" # res50,512
#backbone_weight="${ref_path}/exp7-resnet18-mask=False-channel_att=False-size=256-cls=2/best_model_1.pt" # res18,256
#backbone_weight="${ref_path}/exp8-resnet18-mask=False-channel_att=False-size=512-cls=2/best_model_1.pt" # res18,512

# saliency_weight="${ref_path}/exp17-resnet50_cbam_mask-mask=True-channel_att=True-size=256-cls=2/best_model.pt"  # res50,256,T,T,T
#saliency_weight="${ref_path}/exp18-resnet50_cbam_mask-mask=True-channel_att=True-size=512-cls=2/best_model.pt" # res50,512,T,T,T
#saliency_weight="${ref_path}/exp19-resnet18_cbam_mask-mask=True-channel_att=True-size=256-cls=2/best_model.pt" # res18,256,T,T,T
#saliency_weight="${ref_path}/exp20-resnet18_cbam_mask-mask=True-channel_att=True-size=512-cls=2/best_model.pt" # res18,512,T,T,T
#saliency_weight="${ref_path}/exp29-resnet50_cbam_mask-mask=True-channel_att=False-spatial_att=True-final_att=True-size=256-cls=2/best_model.pt" # res50,256,FTT
#saliency_weight="${ref_path}/exp30-resnet50_cbam_mask-mask=True-channel_att=False-spatial_att=True-final_att=True-size=512-cls=2/best_model.pt" # res50,512,FTT
#saliency_weight="${ref_path}/exp31-resnet18_cbam_mask-mask=True-channel_att=False-spatial_att=True-final_att=True-size=256-cls=2/best_model.pt" # res18,256,FTT
#saliency_weight="${ref_path}/exp32-resnet18_cbam_mask-mask=True-channel_att=False-spatial_att=True-final_att=True-size=512-cls=2/best_model.pt" # res18,512,FTT
#saliency_weight="${ref_path}/exp33-resnet50_cbam_mask-mask=True-channel_att=True-spatial_att=False-final_att=True-size=256-cls=2/best_model.pt" # res50,256,TFT
#saliency_weight="${ref_path}/exp34-resnet50_cbam_mask-mask=True-channel_att=True-spatial_att=False-final_att=True-size=512-cls=2/best_model.pt" # res50,512,TFT
#saliency_weight="${ref_path}/exp35-resnet18_cbam_mask-mask=True-channel_att=True-spatial_att=False-final_att=True-size=256-cls=2/best_model.pt" # res18,256,TFT
#saliency_weight="${ref_path}/exp36-resnet18_cbam_mask-mask=True-channel_att=True-spatial_att=False-final_att=True-size=512-cls=2/best_model.pt" # res18,512,TFT
#saliency_weight="${ref_path}/exp37-resnet50_cbam_mask-mask=True-channel_att=True-spatial_att=True-final_att=False-size=256-cls=2/best_model.pt" # res50,256,TTF
saliency_weight="${ref_path}/exp37-resnet50_cbam_mask-mask=True-channel_att=True-spatial_att=True-final_att=False-size=256-cls=2/best_model.pt" # res50,512,TTF
#saliency_weight="${ref_path}/exp39-resnet18_cbam_mask-mask=True-channel_att=True-spatial_att=True-final_att=False-size=256-cls=2/best_model.pt" # res18,256,TTF
#saliency_weight="${ref_path}/exp40-resnet18_cbam_mask-mask=True-channel_att=True-spatial_att=True-final_att=False-size=512-cls=2/best_model.pt" # res18,512,TTF

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
                --spatial_att=$spatial_att \
                --final_att=$final_att \
                --reduction_ratio=16 \
                --map_size=$map_size \
                --attention_kernel_size=3 \
                --attention_num_conv=3 \
                --backbone_weights="$backbone_weight"\
                --saliency_weights="$saliency_weight"\
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
