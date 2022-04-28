#!/bin/bash
model_dir="BUSI_train/resnet50_cbam_mask_channel_256_stage2"
declare -a StringArray=("best_model.pt")
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    #python eval.py --model_name="deeplabv3" \
    #           --num_classes=3 \
    #           --model_weights=$full_path \
    #           --image_size=448 \
    #           --device="cpu" \
    #           --dataset="BUSI" \
    #           --multi_gpu=True \
    #           image2mask \
    #           --seg_image_list="draw_mask_sample.txt" \
    #           --mask_save_file="eval/BUSI_resnet50_mask_448_mask.png"
    #           --image_path="/Users/zongfan/Projects/data/breas_cancer_us/Dataset_BUSI_with_GT/malignant/malignant (2).png" \
    #           --saliency_file="test/test_saliency.png" \
    #           --target_category=1 \
    python eval.py --model_name="resnet50_cbam_mask" \
               --num_classes=2 \
               --model_weights=$full_path \
               --image_size=448 \
               --device="cuda:0" \
               --dataset="BUSI" \
               --multi_gpu=False \
               --use_cbam=False \
               --use_mask=True \
               --no_channel=False \
               --reduction_ratio=16 \
               --attention_num_conv=3 \
               --attention_kernel_size=3 \
               accuracy 
               #--binary_class=True 
               #--test_file=/shared/anastasio5/COVID19/data/originals/orig_train_sample.txt
    echo "Model processed: $model"
    echo "======================="
done 
