# coding:utf-8

import json

from tqdm import tqdm
from datasets import load_dataset, Dataset
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoConfig

parser = ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--save_file", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--max_seq_length", type=int)
parser.add_argument("--prompt_key", type=str)
parser.add_argument("--target_key", type=str)

args = parser.parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True, device_map='auto', local_files_only=True)

def preprocess(tokenizer, config, example, max_seq_length, prompt_key, target_key):
    prompt = example[prompt_key]
    target = example[target_key]
    if type(target) == list:
        target = target[0]
    if prompt == None or target == None:
        return None
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    target_ids = tokenizer.encode(target, max_length=max_seq_length, truncation=True)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

def read_jsonl(path, max_seq_length, prompt_key, target_key):
    with open(path, "r", encoding="utf-8") as file:
        datas = json.load(file)
    for example in datas:
        feature = preprocess(tokenizer, config, example, max_seq_length, prompt_key, target_key)
        if feature == None:
            continue
        if len(feature["input_ids"]) > max_seq_length:
            continue
        yield feature

def main(args):
    input_path = args.input_file
    save_path = args.save_file
    dataset = Dataset.from_generator(
        lambda: read_jsonl(
            path=input_path,
            max_seq_length=args.max_seq_length,
            prompt_key=args.prompt_key,
            target_key=args.target_key,
        )
    )
    dataset.save_to_disk(save_path)

if __name__ == "__main__":
    main(args)