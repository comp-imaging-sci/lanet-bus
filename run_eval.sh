#!/bin/bash
model_dir="model"
declare -a StringArray=("rasaee.pt")
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    python eval.py --model_name="resnet50_rasaee_mask" \
               --num_classes=3 \
               --model_weights=$full_path \
               --image_size=448 \
               --device="cpu" \
               --dataset="BUSI" \
               --multi_gpu=False \
               --use_cbam=False \
               --use_mask=False \
               --no_channel=True \ 
               saliency \
               --image_path="/Users/zongfan/Projects/data/breas_cancer_us/Dataset_BUSI_with_GT/malignant/malignant (2).png" \
               --saliency_file="test/test_saliency.png" \
               --target_category=1 \
            #    image2mask \
            #    --seg_image_list="draw_mask.txt" \
            #    --mask_save_file="test/test_mask.png" \
            #    --binary_mask=False \
    echo "Model processed: $model"
    echo "======================="
done 
