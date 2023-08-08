# coding:utf-8
# Author: Jiaxin Zhang
# Date: 8th-Aug-2023

import os
import sys
from pathlib import Path
MAIN_PATH = str(Path(__file__).absolute().parent.parent)
sys.path.insert(0, MAIN_PATH)

from argparse import ArgumentParser

from utils import load_yaml_config
from process_data import Datasets

def main(args):
    dataset_obj = Datasets[args.dataset_name]()
    ori_datas = dataset_obj.ori_datas
    
    if args.dataset_name == "HC3-Chinese":
        pairs = dataset_obj.read_qa_pairs(
            examples=ori_datas,
            ori_key=("question", "chatgpt_answers"),
            new_key=("question", "answer"),
        )
    else:
        raise ValueError(f"Not support dataset: {args.dataset_name}")

    save_dir = os.path.join(MAIN_PATH, args.dataset_dir, dataset_obj.dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.json_dataset_save_name)
    dataset_obj.to_json_file(datas=pairs, path=save_path)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--yaml_config", type=str, required=True)
    args = parser.parse_args()
    
    args = load_yaml_config(path=args.yaml_config)
    main(args)