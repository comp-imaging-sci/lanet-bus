#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
python train.py --model_name=deeplabv3 \
                --input_size=64 \
                --num_classes=3 \
                --batch_size=2 \
                --num_epochs=5 \
                --model_save_path=test \
                --device=cpu \
                --lr=0.001 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset="BUSI" \
                --num_gpus=0
