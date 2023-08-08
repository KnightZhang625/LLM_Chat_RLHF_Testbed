# coding:utf-8
# Author: Jiaxin Zhang
# Date: 8th-Aug-2023
# This file is designed for loading dataset according to the dataset name.

import json
import functools

from datasets import load_dataset
from collections import defaultdict
from typing import Tuple, List, Dict, Optional, Union

class BaseDataset:
    def __init__(self, dataset_name: str) -> None:
        self.dataset_name = dataset_name
    
    def __select_read_func(data_type: str):
        def __read_pairs(spec_func):
            functools.wraps(spec_func)
            def __read_pairs_inner(*args, **kwargs):
                if data_type == "qa":
                    # call read_qa_pairs()
                    return spec_func(*args, examples=kwargs["examples"], ori_key=kwargs["ori_key"], new_key=kwargs["new_key"])
                else:
                    raise ValueError(f"Not supprot data type: {data_type}.")
            return __read_pairs_inner
        return __read_pairs
    
    @__select_read_func("qa")
    def read_qa_pairs(self, examples: Dict, ori_key: Tuple, new_key: Tuple) -> Dict:
        """read datasets consisting (question, answer) paris.

        Args:
            examples (dict): the dataset from the "load_dataset".
            ori_key: the key names in the examples.
            new_key: the key names for the return dict.

        Returns:
            dict: a new dict of "new_key" or ("question", "answer") as key names.
        """
        old_key_que, old_key_ans = ori_key
        new_key_que, new_key_ans = new_key
        datas = defaultdict(list)
        for que, ans_meta in zip(examples["train"][old_key_que], examples["train"][old_key_ans]):
            if type(ans_meta) == list:
                for ans in ans_meta:
                    datas[new_key_que].append(que)
                    datas[new_key_ans].append(ans)
            else:
                datas[new_key_que].append(que)
                datas[new_key_ans].append(ans_meta)        
        return datas

class HC3Dataset(BaseDataset):
    def __init__(self):
        super().__init__(dataset_name="HC3")
    
    @property
    def ori_datas(self):
        return load_dataset("Hello-SimpleAI/HC3-Chinese", "all")
            
    def to_json_file(self, datas: Dict, path: str) -> None:
        with open(path, 'w', encoding="utf-8") as file:
            que_list = datas["question"]
            ans_list = datas["answer"]
            for que, ans in zip(que_list, ans_list):
                line = {"q" : "问： " + que, "a": "答：" + ans}
                line_json = json.dumps(line, ensure_ascii=False)
                file.write(line_json)
                file.write("\n")

if __name__ == "__main__":
    dummy_data = {
        "question": ["hello", "hi"],
        "chatgpt_answers": ["hi", "hello"],
    }
    
    dataset = BaseDataset("dummy_data")
    datas = dataset.read_qa_pairs(examples=dummy_data, ori_key=("question", "chatgpt_answers"), new_key=("question", "answer"))
    print(datas)