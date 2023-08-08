#!/usr/bin/env bash

# set -xe

CUDA_VISIBLE_DEVICES=0,1 python lora_tuning.py \
    --tokenized_dataset hc3_chatgpt_zh_specific_qa_baichuan-7B \
    --lora_rank 4 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 2 \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir weights/hc3_chatgpt_zh_specific_qa_baichuan-7B