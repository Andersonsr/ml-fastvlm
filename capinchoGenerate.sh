#!/bin/bash

python ./eval/generateCaptions.py \
    --embeddings  E:/datasets/mimic/embeddings/cls4-mixer-lora_test.pkl \
    --model checkpoints/teste-decoder/experiment.json \
    --num_images 10 \
    --dataset mimic \



