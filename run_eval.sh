#!/bin/bash
model_dir="test"
declare -a StringArray=("best_model.pt")
for model in ${StringArray[@]};
do
    full_path="$model_dir/$model"
    python eval.py --model_name=resnet50 \
               --num_classes=3 \
               --model_weights=$full_path \
               --input_size=64 \
               --device=cpu \
               --dataset="BUSI"
    echo "Model processed: $model"
    echo "======================="
done 
