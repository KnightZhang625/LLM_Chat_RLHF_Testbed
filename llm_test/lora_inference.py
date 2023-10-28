# coding:utf-8

import torch
import argparse
    
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--lora_weights", type=str)
args = parser.parse_args()

def main(args):

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).half().to(device)
    model = model.eval()

    response, history = model.chat(tokenizer, "一共有10个纯文本模型, 12个多模态模型, 嘉鑫这周要训练5个纯文本模型, 明亮这周要训练5个多模态模型, 下周各个模型还剩多少?")
    print("Original: ", response)
    
    model = PeftModel.from_pretrained(model, args.lora_weights).half()
    retry_cnt=0
    while retry_cnt < 5:
        try:
            response, history = model.chat(
                tokenizer,
                 "一共有10个纯文本模型, 12个多模态模型, 嘉鑫这周要训练5个纯文本模型, 明亮这周要训练5个多模态模型, 下周各个模型还剩多少?",
                history=[],
            )
            break
        except Exception as e1:
            retry_cnt += 1
    print("Lora: ", response)

if __name__ == "__main__":
    main(args)