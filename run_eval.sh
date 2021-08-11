#!/bin/bash
model_dir="model"
declare -a StringArray=("res50.pt")
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    python eval.py --model_name="resnet50" \
               --num_classes=2 \
               --model_weights=$full_path \
               --image_size=448 \
               --device="cpu" \
               --dataset="MAYO" \
               --multi_gpu=True \
            #    image2mask \
            #    --seg_image_list="draw_mask_debug.txt" \
            #    --mask_save_file="test/test_mask.png"
    echo "Model processed: $model"
    echo "======================="
done 
