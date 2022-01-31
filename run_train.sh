#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
model_name="resnet50"
image_size=256
python train.py --model_name=$model_name \
                --input_size=$image_size \
                --num_classes=2 \
                --batch_size=8 \
                --num_epochs=100 \
                --model_save_path="BUSI_train/${model_name}_${image_size}_tfs" \
                --device="cuda:0" \
                --lr=0.0001 \
                --moment=0.9 \
                --use_pretrained=False \
                --dataset="BUSI" \
                --num_gpus=1 \
                --dilute_mask=25 \
                --mask_annotate_file="data/mayo_patient_info.csv" \
                --use_cbam=False \
                --use_mask=False \
                --no_channel=True \
                --reduction_ratio=16 \
                --attention_kernel_size=3 \
                --attention_num_conv=3
                #--pretrained_weights="/shared/anastasio5/COVID19/ultrasound_breast_cancer/MAYO_resnet50_mask_448/best_model.pt" \

# exp 1
# resnet50 cbam mask no_channel size=256 class=2

# exp 2
# resnet50 cbam no_channel size=256 class=2

# exp 3
# resnet50 size=256 class=2

# exp 4
# resnet50 mask cbam no_channel size=256 class=2

# exp 5
# resnet50 cbam mask channel size=256 class=2

# exp 6 
# resnet50 cbam channel size=256 class=2

# exp 7
# resnet50 mask cbam channel size=256 class=2
