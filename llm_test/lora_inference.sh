#!/usr/bin/env bash

# set -xe

model_name=THUDM/chatglm3-6b
lora_weights=500_chatglm3-6b

BASE_DIR=.
CODE_DIR=${BASE_DIR}
SAVE_WEIGHTS_DIR=${BASE_DIR}/saved_model

ARGS="--model_name ${model_name} \
      --lora_weights ${SAVE_WEIGHTS_DIR}/${lora_weights} \
     "

CUDA_VISIBLE_DEVICES=0 python ${CODE_DIR}/lora_inference.py ${ARGS}