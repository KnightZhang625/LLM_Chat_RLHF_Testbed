# coding:utf-8
# Author: Jiaxin Zhang
# Date: 8th-Aug-2023

import yaml
import codecs
from argparse import Namespace

def load_yaml_config(path: str) -> Namespace:
    with codecs.open(path, "r", "utf-8") as file:
        config = yaml.safe_load(file)
    
    args = {}
    for key, value in config.items():
        args[key] = value
    
    args = Namespace(**args)
    
    return args

if __name__ == "__main__":
    path = "../config/data_config.yaml"
    args = load_yaml_config(path)
    
    print(args)