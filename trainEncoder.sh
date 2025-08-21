#!/bin/bash

python ./model/trainEncoder.py \
    --encoder_path ./checkpoints/llava-fastvithd_0.5b_stage3/ \
    --annotation E:/datasets/mimic/preprocess/micro_split.json \
    --root-dir E:/datasets/mimic/preprocess/resize_1024/ \
    --output_classes 4 \
    --batch_size 2 \
    --output_dir ./checkpoints/class-4-mixer-lora-mapper \
    --dim 768 \
    --mapper_out_dim 512 \
    --epochs 1 \
    --logging_interval 2 \
    --unfreeze_modules mixer \
    --train_mapper \
    --lora \

