#!/bin/bash
model_dir="BUSI_deeplabv3_448_class3"
declare -a StringArray=("best_model.pt")
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    python eval.py --model_name="deeplabv3" \
               --num_classes=3 \
               --model_weights=$full_path \
               --image_size=448 \
               --device="cuda:0" \
               --dataset="BUSI" \
               --multi_gpu=True \
               iou 
               #--seg_image_list="draw_mask_sample_debug.txt" \
               #--mask_save_file="eval/BUSI_deeplabv3_448_class3_mask_test.png"
    #python eval.py --model_name="resnet50" \
    #           --num_classes=3 \
    #           --model_weights=$full_path \
    #           --image_size=224 \
    #           --device="cuda:0" \
    #           --dataset="BUSI" \
    #           --multi_gpu=True \
    #           accuracy \
    #           --test_file=/shared/anastasio5/COVID19/data/originals/orig_train_sample.txt
    echo "Model processed: $model"
    echo "======================="
done 
