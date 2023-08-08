#!/usr/bin/env bash

# set -xe

BASE_DIR=.
MAIN_DIR=..
CONFIG_DIR=${MAIN_DIR}/config
CONFIG_PATH=${CONFIG_DIR}/data_config.yaml

ARGS="--yaml_config ${CONFIG_PATH}"

python ${BASE_DIR}/tokenize_save_data.py ${ARGS}