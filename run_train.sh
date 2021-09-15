#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
python train.py --model_name="resnet50" \
                --input_size=448 \
                --num_classes=2 \
                --batch_size=16 \
                --num_epochs=100 \
                --model_save_path=MAYO_resnet50_448_001-150 \
                --device=cuda:0 \
                --lr=0.0001 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset="MAYO" \
                --num_gpus=1 \
                --dilute_mask=25 \
                --pretrained_weights="/shared/anastasio5/COVID19/ultrasound_breast_cancer/MAYO_resnet50_mask_448/best_model.pt" \
