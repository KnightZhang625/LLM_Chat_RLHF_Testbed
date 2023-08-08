# coding:utf-8
# Author: Jiaxin Zhang
# Date: 8th-Aug-2023

import os
import sys
from pathlib import Path
MAIN_PATH = str(Path(__file__).absolute().parent.parent)
sys.path.insert(0, MAIN_PATH)

from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from dataclasses import dataclass, field
import datasets
import os
from pprint import pprint as print

model_checkpoint = "baichuan-inc/baichuan-7B"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)

@dataclass
class FinetuneArguments:
    tokenized_dataset: str = field(default=" ") # tokenized之后的数据集文件夹
    model_path: str = field(default=" ")
    lora_rank: int = field(default=8)
    previous_lora_weights: str = field(default=None) # 如果要在前面的 LoRA 上继续训练，就设置一下之前的地址

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

tokenizer.pad_token = tokenizer.unk_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
# DataCollatorForLanguageModeling 会自动帮你 padding, labels
# Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.
# 参考教程：https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt

class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        # 因为交给Trainer的model实际上是PeftModel类型，所以这里的 save_pretrained 会直接使用PeftModel的保存方法
        # 从而只保存 LoRA weights
        self.model.save_pretrained(output_dir)
        # from transformers.trainer import TRAINING_ARGS_NAME
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        # saved_params = {
        #     k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        # }
        # torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))

def main():
    writer = SummaryWriter()
    finetune_args, training_args = HfArgumentParser(
        (FinetuneArguments, TrainingArguments)
    ).parse_args_into_dataclasses()

    # load dataset
    dataset = datasets.load_from_disk('data/HC3/'+finetune_args.tokenized_dataset)
    print(f"\n{len(dataset)=}\n")

    # init model
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint, load_in_8bit=False, trust_remote_code=True, 
        device_map="auto" # 模型不同层会被自动分配到不同GPU上进行计算
    )
    print(model.hf_device_map)
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.lm_head)

    # setup peft
    if finetune_args.previous_lora_weights == None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=finetune_args.lora_rank,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules = ["W_pack"] # 把model打印出来，找跟attention相关的模块
        )
        
        model = get_peft_model(model, peft_config)
    else:
        model = PeftModel.from_pretrained(model, finetune_args.previous_lora_weights)
        # see: https://github.com/huggingface/peft/issues/184
        for name, param in model.named_parameters():
            if 'lora' in name or 'Lora' in name:
                param.requires_grad = True

    # start train
    model.save_pretrained(training_args.output_dir) # 因为adapter_config.json只能通过这个save_pretrained来生成，先这里生成一份，好在训练完之前就可以尝试中间的checkpoint
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    trainer.train()
    writer.close()
    # save model
    model.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()