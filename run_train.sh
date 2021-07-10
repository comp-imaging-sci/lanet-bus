#!/bin/bash
# matlab -nodisplay -nosplash -nodesktop -r "run('/home/xiaohui8/Desktop/tube_samples_dataset/GoogLeNet/googlenet_pretrain.m');exit;"|tail -n +11
python train.py --model_name="resnet50_mask" \
                --input_size=448 \
                --num_classes=3 \
                --batch_size=16 \
                --num_epochs=200 \
                --model_save_path=BUSI_resnet50_mask_448 \
                --device=cuda:0 \
                --lr=0.0001 \
                --moment=0.9 \
                --use_pretrained=True \
                --dataset="BUSI" \
                --num_gpus=1 \
                --dilute_mask=20
