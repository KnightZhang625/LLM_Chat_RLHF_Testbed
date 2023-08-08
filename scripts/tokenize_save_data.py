# coding:utf-8
# Author: Jiaxin Zhang
# Date: 8th-Aug-2023

import os
import sys
from pathlib import Path
MAIN_PATH = str(Path(__file__).absolute().parent.parent)
sys.path.insert(0, MAIN_PATH)

import datasets
from argparse import ArgumentParser

from utils import load_yaml_config
from data_loader import CustomizedDataLoader


def main(args):
    data_loader = CustomizedDataLoader[args.dataset_name](args)
    
    dataset = datasets.Dataset.from_generator(
        lambda: data_loader.load_data()
    )
    
    save_path = os.path.join(MAIN_PATH, "data", data_loader.name)
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(os.path.join(save_path, args.tokenized_file_save_name))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--yaml_config", type=str, required=True)
    args = parser.parse_args()
    
    args = load_yaml_config(path=args.yaml_config)
    main(args)