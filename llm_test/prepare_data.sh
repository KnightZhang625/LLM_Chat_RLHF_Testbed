#!/usr/bin/env bash

# set -xe

model_name=EleutherAI/gpt-neo-1.3B 
data_name=MetaMathQA-395K.json
save_name=processed_MetaMathQA-395K
max_seq_length=2028
prompt_key=query
target_key=response

BASE_DIR=.
CODE_DIR=${BASE_DIR}/process_data
DATA_DIR=${BASE_DIR}/data
CACHED_DATA_DIR=${DATA_DIR}/cached_data

ARGS="--model_name ${model_name} \
      --input_file ${DATA_DIR}/${data_name} \
      --save_file ${CACHED_DATA_DIR}/${save_name} \
      --max_seq_length ${max_seq_length} \
      --prompt_key ${prompt_key} \
      --target_key ${target_key}
      "

python ${CODE_DIR}/prepare_data.py ${ARGS}