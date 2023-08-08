# coding:utf-8
# Author: Jiaxin Zhang
# Date: 8th-Aug-2023

import os
import sys
from pathlib import Path
MAIN_PATH = str(Path(__file__).absolute().parent.parent)
sys.path.insert(0, MAIN_PATH)

import json
import transformers

class HC3DataLoader:
    def __init__(self, args):
        self.name = "HC3"
        # set tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.tokenizer_model, trust_remote_code=True,
        )
        self.tokenizer_config = transformers.AutoConfig.from_pretrained(
            args.tokenizer_model, trust_remote_code=True, device_map="auto",
        )
        
        self.max_seq_length = args.max_seq_length
        
        # load datas
        datas_path = os.path.join(MAIN_PATH, "data", "HC3", args.json_dataset_save_name)
        with open(datas_path, "r") as file:
            self.datas = file.readlines()
                
    def __len__(self):
        return len(self.datas)
    
    def process_line(self, example):
        prompt = example["q"]
        target = example["a"]
        
        prompt_ids = self.tokenizer.encode(prompt, max_length=self.max_seq_length, truncation=True)
        target_ids = self.tokenizer.encode(target, max_length=self.max_seq_length, truncation=True, add_special_tokens=False)        
        input_ids = prompt_ids + target_ids + [self.tokenizer_config.eos_token_id]
        
        return {
            "input_ids": input_ids,
            "seq_len": len(prompt_ids)
        }

    def load_data(self):
        for line in self.datas:
            example = json.loads(line)
            feature = self.process_line(example)
            feature["input_ids"] = feature["input_ids"][:self.max_seq_length]
            yield feature
    