import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from models.qwen2 import Qwen2ForCausalLM
import time
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

model_name = "/share/models/opt-30b"
model_name = "Qwen/Qwen1.5-32B-Chat" 
model_name = "Qwen/Qwen1.5-4B-Chat"  # 输出全空
model_name = "Qwen/Qwen1.5-14B-Chat"  
model_name = "Qwen/Qwen1.5-1.8B-Chat"  
model_name = "Qwen/Qwen1.5-72B-Chat" 
model_name = "Qwen/Qwen1.5-7B-Chat"  
model_name = "Qwen/Qwen1.5-0.5B-Chat"  
model_name = "/share/models/Qwen2.5-7B-Instruct"  


# 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = Qwen2ForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             torch_dtype=torch.float16,
                                             ) # torch_dtype=torch.float16

# 输入与分词

input_text = """what is the name of justin bieber brother?"""
input_text = """Could you please give me some advice on programming?

Give me answer in English.<|endoftext|>"""
input_text = "我希望学习编程，给我一些实用的建议，以便我能更好的入门?<|im_end|>"
input_text = ["我希望学习编程，给我一些实用的建议，以便我能更好的入门?<|im_end|>","我希望学习大语言模型，我应该从GPT2入手吗？并且简要介绍一下KVCache。<|im_end|>"]

# 数据集
folder = "/share/datasets/flexflow/"
datasets = ["alpaca.json", "chatbot.json", "chatgpt.json", "piqa.json", "webqa.json"]
dataset_index = 4
with open(folder + datasets[dataset_index], 'r') as f:
    data = f.readlines()
    sentences = data[0][1:-1].split(",")
    sentences = [sentence.strip()[1:-1] for sentence in sentences]
input_text = sentences[:16]

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f'{text = }')

# prompt = tokenizer.encode(input_text,return_tensors="pt").to("cuda:1")
prompt = tokenizer(
    [text, text],                     # 输入文本列表
    padding=True,                     # 自动添加 padding，以便所有序列等长
    truncation=True,                  # 如果序列长度超过最大长度，则进行截断
    return_tensors='pt'               # 返回 PyTorch 张量
).to("cuda:2")

input_ids = prompt.input_ids
# print(input_ids.shape) # torch.Size([1, 6])
init_len = input_ids.shape[-1]
# GPU warmup

def warmup_gpu():
    model.generate(input_ids, do_sample=True, max_new_tokens=10)

# warmup_gpu()

# 推理
max_token_len = 50 # 生成的最大长度
do_sample = True # 是否使用采样
temperature = 1 # 采样温度

s1 = time.perf_counter_ns() / 1e6
input_ids = model.generate(input_ids, do_sample=do_sample, max_new_tokens=max_token_len, use_cache=True)
e1 = time.perf_counter_ns() / 1e6
inference_time = e1 - s1

total_generate_num = input_ids.shape[1] - init_len
result = tokenizer.batch_decode(input_ids,skip_special_tokens=True)
per_token_time = inference_time / total_generate_num
for i in range(len(result)):
    print(result[i])
print(str(per_token_time) + " ms/token.")