#!/bin/bash
model_dir="test"
declare -a StringArray=("best_model.pt")
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    python eval.py --model_name="deeplabv3" \
               --num_classes=1 \
               --model_weights=$full_path \
               --image_size=64 \
               --device="cpu" \
               --dataset="BUSI" \
               image2mask \
               --seg_image_list="draw_mask_sample.txt" \
               --mask_save_file="test/test_mask.png"
    echo "Model processed: $model"
    echo "======================="
done 
