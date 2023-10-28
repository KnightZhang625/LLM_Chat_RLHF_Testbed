#!/usr/bin/env bash

# set -xe

model_name=THUDM/chatglm3-6b
save_name=processed_school_math_0.25M
max_seq_length=2028
output_dir=saved_model

BASE_DIR=.
CODE_DIR=${BASE_DIR}
DATA_DIR=${BASE_DIR}/data
CACHED_DATA_DIR=${DATA_DIR}/cached_data

ARGS="--model_name ${model_name} \
      --processed_data ${CACHED_DATA_DIR}/${save_name} \
      --output_dir ${output_dir}
      "

# CUDA_VISIBLE_DEVICES=0,1 python ${CODE_DIR}/lora_fine_tune.py ${ARGS}
# torchrun --nproc_per_node=2 ${CODE_DIR}/lora_fine_tune.py ${ARGS}
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=INFO accelerate launch ${CODE_DIR}/lora_fine_tune.py ${ARGS}
# accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 ${CODE_DIR}/lora_fine_tune.py ${ARGS}
# NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO accelerate launch ${CODE_DIR}/lora_fine_tune.py ${ARGS}