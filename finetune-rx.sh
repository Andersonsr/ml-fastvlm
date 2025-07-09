#!/bin/bash

python llava\\train\\train_qwen.py \
    --model_name_or_path C:\\Users\\Usuario\\PycharmProjects\\RX\\src\\ml-fastvlm\\checkpoint\\llava-fastvithd_0.5b_stage3 \
    --vision_tower  C:\\Users\\Usuario\\PycharmProjects\\RX\\src\\ml-fastvlm\\checkpoint\\llava-fastvithd_0.5b_stage3\
    --mm_vision_tower "mobileclip_l_1024"\
    --version v1 \
    --data_path E:\\datasets\\mimic\\preprocess\\training_split_llava.json \
    --image_folder E:\\datasets\\mimic\\preprocess\\resize_1024 \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir D:\modelos_v2\decoder-tune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --mm_projector_type: "mlp2x_gelu" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \


