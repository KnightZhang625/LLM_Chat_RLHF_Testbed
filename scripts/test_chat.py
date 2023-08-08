# coding:utf-8

# 看看训练之后baichuan是否具备了Chat能力：
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import TextStreamer

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/baichuan-7B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/baichuan-7B", device_map="auto", trust_remote_code=True)
# load LoRA:
model = PeftModel.from_pretrained(model, "weights/hc3_chatgpt_zh_specific_qa_baichuan-7B")

def chat(text):
    streamer = TextStreamer(tokenizer,skip_prompt=True,skip_special_tokens=True)
    inputs = tokenizer("问："+text+"答：", return_tensors='pt') # 这里添加 "问：","答："，是为了跟我构造的训练数据对应，从而更好地引导模型进行回答
    inputs = inputs.to('cuda:0')
    output = model.generate(**inputs, max_new_tokens=1024,repetition_penalty=1.1, streamer=streamer)