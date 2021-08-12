#!/bin/bash
model_dir="MAYO_resnet50_mask_448"
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
    python eval.py --model_name="resnet50_mask" \
               --num_classes=2 \
               --model_weights=$full_path \
               --image_size=448 \
               --device="cuda:0" \
               --dataset="MAYO" \
               --multi_gpu=False \
               accuracy 
               #--test_file=/shared/anastasio5/COVID19/data/originals/orig_train_sample.txt
    echo "Model processed: $model"
    echo "======================="
done 
