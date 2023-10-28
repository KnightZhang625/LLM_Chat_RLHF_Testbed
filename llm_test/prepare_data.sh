#!/usr/bin/env bash

# set -xe

model_name=THUDM/chatglm3-6b
data_name=school_math_0.25M.json
save_name=processed_school_math_0.25M
max_seq_length=2028
prompt_key=instruction
target_key=output

BASE_DIR=.
CODE_DIR=${BASE_DIR}/process_data
DATA_DIR=${BASE_DIR}/data
CACHED_DATA_DIR=${DATA_DIR}/cached_data

ARGS="--model_name ${model_name} \
      --input_file ${DATA_DIR}/${data_name} \
      --max_seq_length ${max_seq_length} \
      --save_file ${CACHED_DATA_DIR}/${save_name} \
      --prompt_key ${prompt_key} \
      --target_key ${target_key} \
      "

python ${CODE_DIR}/prepare_data.py ${ARGS}