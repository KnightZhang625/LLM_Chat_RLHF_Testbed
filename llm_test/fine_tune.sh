#!/usr/bin/env bash

# set -xe

model_name=THUDM/chatglm3-6b
save_name=processed_MetaMathQA-395K
max_seq_length=2028
prompt_key=question
target_key=human_answers
output_dir=saved_model/chatglm_for_math

BASE_DIR=.
CODE_DIR=${BASE_DIR}
CACHED_DATA_DIR=${BASE_DIR}/cached_data

ARGS="--model_name ${model_name} \
      --processed_data ${CACHED_DATA_DIR}/${save_name} \
      --output_dir ${output_dir}
      "

# CUDA_VISIBLE_DEVICES=0,1 python ${CODE_DIR}/lora_fine_tune.py ${ARGS}
# torchrun --nproc_per_node=2 ${CODE_DIR}/lora_fine_tune.py ${ARGS}
CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_BLOCKING_WAIT=1 NCCL_DEBUG=INFO accelerate launch ${CODE_DIR}/lora_fine_tune.py ${ARGS}
# accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 ${CODE_DIR}/lora_fine_tune.py ${ARGS}
# NCCL_IB_GID_INDEX=3 NCCL_DEBUG=INFO accelerate launch ${CODE_DIR}/lora_fine_tune.py ${ARGS}