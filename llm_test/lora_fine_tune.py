# coding:utf-8

# import os
# os.environ["NCCL_DEBUG"]="INFO"
# os.environ["NCCL_BLOCKING_WAIT"]="1"

import torch
import torch.nn as nn
import datasets

from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.optim as optim
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator

parser = ArgumentParser()
parser.add_argument("--model_name", type=str)
parser.add_argument("--processed_data", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

model_path = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

def collate_fn(features):
    input_ids_len = [len(feature["input_ids"]) for feature in features]
    longest = max(input_ids_len)
    
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(input_ids_len, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
    
        input_ids.append(torch.LongTensor(ids))
        labels_list.append(torch.LongTensor(labels))
    
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

def main(args):
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = datasets.load_from_disk(args.processed_data)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = AutoModel.from_pretrained(
        args.model_name, load_in_8bit=False, trust_remote_code=True, local_files_only=True
    )
    
    model.gradient_checkpointing_enable() 
    model.enable_input_require_grads()
    model.transformer.output_layer = CastOutputToFloat(model.transformer.output_layer)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    accelerator = Accelerator()
    print("Accelerate prepare...")
    model, optimizer, data_loader = accelerator.prepare(
        model, optimizer, data_loader
    )
    model.train()
    num_training_steps = 100 * len(data_loader)
    for epoch in range(100):
        print(f"Epoch: {epoch}")
        for step, batch in enumerate(data_loader):
            
            # input_ids = batch["input_ids"].to(device)
            # labels = batch["labels"].to(device)
            
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            print("forward...")
            loss = model(input_ids=input_ids, labels=labels).loss
            # loss.backward()
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            accelerator.print(f"Step: {step}, Loss: {loss.detach().item()}")

            if step % 100 == 0 and accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main(args)