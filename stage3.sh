#!/bin/bash

python llava\\train\\train_rx.py \
    --vision_tower  mobileclip_l_1024 \
    --version qwen_2 \
    --model_name_or_path checkpoints/test \
    --data_path E:\\datasets\\mimic\\preprocess\\micro_split_llava.json \
    --image_folder E:\\datasets\\mimic\\preprocess\\resize_1024 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints\\test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
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
    --cls_feature_mm_vision_tower True \
    --mm_projector_type "mapper10" \
#
#   \
#   --tuned_projector checkpoints/projector-c3ml \
#    --lora_enable \
#    --lora_dropout 0.5 \
#    --lora_r 4 \
#    --lora_alpha 8 \


