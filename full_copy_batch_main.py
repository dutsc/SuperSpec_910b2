import torch
import time
import torch.multiprocessing as mp
from full_copy_utils.batch_verify import compute_and_batch_verify_greedy, compute_and_batch_verify_stochastic
from full_copy_utils.draft_sampling import distribute_data
from full_copy_utils.utils import split_gpu_outputs, longest_common_prefix_length
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.full_copy_opt import OPTForCausalLM
from models.full_copy_qwen2 import Qwen2ForCausalLM
# from nsl_opt3 import OPTForCausalLM
from full_copy_utils.draft_sampling import draft_sample
from full_copy_utils.verify_kvcache import KVCacheModel
import os
import argparse
from full_copy_utils.token_tree import create_mask_seq
from tqdm import tqdm
import json
import numpy as np
import gc

# 读取配置文件
def read_config():
    with open('./config.json', 'r') as config_file:
        config = json.load(config_file)
    return config
config = read_config()

# 主程序
SSM_GPUs = config['ssm']['GPUS'] # 前三个GPU用于生成
LLM_GPU = config['llm']['GPU']  # 第四个GPU用于数据收集
temperature = config['llm']['temperature']
temperature_ssm = config['ssm']['temperature']
heuristic = config['heuristic']
llm_model_path = config['llm']['model']

original_text = """
dialogue:
Alex: "Awesome, my phone fell into the water again. This is the third time this year."
Casey: "Congratulations! You are now officially a VIP member of our club."
Jamie: "Haha, Alex, maybe it's time to consider getting a waterproof phone, or a bigger pocket."
Alex: "Or should I stop using my phone by the water? Who knows?"

question:
1. What emotion does Casey's reply contain? Explain the meaning behind Casey's reply.
2. Is Jamie's advice serious or a joke? Please provide reasons.
3. What attitude does Alex’s last reply show? How does Alex feel about his habit of dropping his phone into water?

please answer the three questions.
"""
original_text = "please summarize these sentences: This paper introduces SpecInfer, a system that enhances the efficiency of serving generative large language models (LLMs) through tree-based speculative inference and verification. SpecInfer's innovative approach utilizes smaller speculative models to forecast the outputs of LLMs; these forecasts are structured into a token tree, with each node signifying a potential token sequence. The validity of all token sequences represented by the token tree is concurrently verified against the LLM using a cutting-edge tree-based parallel decoding mechanism. By employing an LLM as a token tree verifier rather than an incremental decoder, SpecInfer notably lowers the end-to-end latency and computational demands for deploying generative LLMs while ensuring the preservation of model quality. Evaluations indicate that SpecInfer surpasses current LLM serving systems by a factor of 1.5-2.8 for distributed LLM inference and 2.6-3.5 for offloading-based LLM inference, all the while maintaining equivalent generative capabilities. SpecInfer is openly accessible at the provided GitHub link."
original_text = "please introduce Kobe Bryant, who played basketball in NBA."
original_text = "Could you please give me some advice on programming?|<im_end>|" #"Hello, my name is"

def warm_up_gpu(device,event) -> None:
    # 创建一些虚拟数据
    x = torch.rand((8192, 8192), device=device)
    # 执行一些轻量级的计算任务
    for _ in range(1000):
        y = x.mm(x)
    print("GPU warmed up!")
    event.set()
    
if __name__ == '__main__':
    mp.set_start_method('spawn',force=True)
    # 存放所有小模型推理结果的队列
    draft_queue = mp.Queue()
    # 存放n个小模型进程
    ssm_queue = []
    # event用于表示draft model是否完成推理
    manager = mp.Manager()
    event = manager.Event()
    
    # 数据集
    folder = "/share/datasets/flexflow/"
    datasets = ["alpaca.json", "chatbot.json", "chatgpt.json", "piqa.json", "webqa.json"]
    dataset_index = 0
    with open(folder + datasets[dataset_index], 'r') as f:
        data = f.readlines()
        sentences = data[0][1:-1].split(",")
        sentences = [sentence.strip()[1:-1] for sentence in sentences]
    # print(f'{len(sentences) = }')
    sentences = sentences[:1]
    
    for gpu in SSM_GPUs:
        # 存放小模型的推理结果
        task_queue = mp.Queue()
        p = mp.Process(target = draft_sample, args = (gpu, task_queue, draft_queue, config))
        ssm_queue.append((p, task_queue))
        p.start()
        
    if "Qwen" in llm_model_path:
        init_llm = Qwen2ForCausalLM.from_pretrained(llm_model_path,torch_dtype=torch.float16,device_map="auto")
    elif "opt" in llm_model_path:
        init_llm = OPTForCausalLM.from_pretrained(llm_model_path,torch_dtype=torch.float16,device_map="auto")
    llm = KVCacheModel(init_llm,temperature)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
    prompt = tokenizer(sentences[0], return_tensors="pt")['input_ids'].to(f'cuda:{LLM_GPU}')

    # 设置生成的最大长度
    max_length = config['max_length']
    lookaheads = config['lookaheads']
    num_of_spec = 0
    eos_token_id = tokenizer.eos_token_id
    
    # warm up
    for i in range(10):
        gpu_outputs = distribute_data(prompt, ssm_queue, draft_queue, event, True, config)
        new_tokens,all_accepted,accept_num,batch_winner_index, batch_infer_time = compute_and_batch_verify_greedy(gpu_outputs, llm, prompt, config)
        llm._past_key_values = None    
        llm._new_key_values = None
        llm._logits_history = None
    print(f'-----------------------------------warm up end---------------------------------------')

    total_generate_tokens = 0
    total_infer_time = 0
    total_accept_num = 0
    total_autoregressive_token_num = 0
    spec_num = 1
    
    # 存储输出信息
    wall_times = []
    generate_tokens = []
    generate_texts = []
    dataset_infer_times = []

    # time.sleep(5)
    
    for prompt in tqdm(sentences, desc="Processing sentences"):
        prompt = tokenizer(prompt, return_tensors="pt")['input_ids'].to(f'cuda:{LLM_GPU}')
        init_len = prompt.shape[1]
        clear_cache = True
        llm._past_key_values = None    
        llm._new_key_values = None
        llm._logits_history = None
        single_prompt_infer_times = []
        start_time = time.perf_counter_ns() / 1e6
        while True:
            epoch_start = time.time_ns() / 1_000_000
            gpu_outputs = distribute_data(prompt, ssm_queue, draft_queue, event, clear_cache, config)
            
            # 本次投机推理第一轮跑过之后，不再清理kvcache
            clear_cache = False
            verify_start = time.time_ns() / 1_000_000
            tokens,logits = split_gpu_outputs(gpu_outputs,LLM_GPU)
            prefix_l = longest_common_prefix_length(tokens)
            print(f'\nprefix_l: {prefix_l} \n')
            new_tokens,all_accepted,accept_num,batch_winner_index, batch_infer_time = compute_and_batch_verify_greedy(gpu_outputs, llm, prompt, config)
            single_prompt_infer_times.append(batch_infer_time)
            
            print(f'{accept_num = }')
            # llm._past_key_values = None    
            # llm._new_key_values = None
            # llm._logits_history = None
            
            # del llm._past_key_values
            # del llm._new_key_values
            # del llm._logits_history
            # gc.collect()
            
            # print(f'{batch_winner_index = }')
            total_accept_num += accept_num - 1
            total_autoregressive_token_num += max(lookaheads)
            
            if all_accepted:
                llm.rollback_with_batch_winner_full_copy(prompt.size(1) + len(new_tokens), batch_winner_index)
            else:
                llm.rollback_with_batch_winner_full_copy(prompt.size(1) + len(new_tokens) - 1, batch_winner_index)
            # 启发式变更lookahead  变更所有draft model的lookahead
            if heuristic:
                if accept_num == lookaheads[0]:
                    lookaheads = [x + 2 for x in lookaheads]
                else:
                    lookaheads = [max(1, x-1) for x in lookaheads]

            for token in new_tokens:
                if token == eos_token_id:  # 假设new_tokens是整数ID的列表
                    break
                
            spec_generated_text = torch.zeros_like(new_tokens[0]).to(f'cuda:{LLM_GPU}')
            for token in new_tokens:
                prompt = torch.cat([prompt, token], dim=-1)
                spec_generated_text = torch.cat([spec_generated_text, token], dim=-1)
            spec_generated_text = tokenizer.decode(spec_generated_text[0], skip_special_tokens=True)
            print(f"spec_generated_text = {spec_generated_text}")

            # 检查长度限制
            if prompt.size(1) >= max_length:
                if config['timeline']:
                    print(f'-----------------------------------epoch {spec_num} end---------------------------------------')
                break
            epoch_time = time.time_ns() / 1_000_000 - epoch_start
            if config['timeline']:
                print(f'-----------------------------------epoch {spec_num} end---------------------------------------')
            spec_num += 1
        end_time = time.perf_counter_ns() / 1e6
        prompt_generate_tokens = prompt.shape[1] - init_len
        single_prompt_time = end_time - start_time
        seconds_per_token = single_prompt_time / prompt_generate_tokens 
        single_prompt_mean_verify_time = np.mean(single_prompt_infer_times)
        dataset_infer_times.append(single_prompt_mean_verify_time)

        # 将生成的tokens转换为文本
        generated_text = tokenizer.batch_decode(prompt, skip_special_tokens=True)
        print("Generated Text:", generated_text)
        total_infer_time += single_prompt_time
        total_generate_tokens += prompt_generate_tokens
        
        wall_times.append(single_prompt_time)
        generate_texts.append(generated_text)
        generate_tokens.append(prompt_generate_tokens)
        
    print(f'{total_infer_time = } ms.')
    print(f'{total_generate_tokens = }')
    mean_per_token_infer_time = total_infer_time / total_generate_tokens
    print(f'{mean_per_token_infer_time = } ms/token.')
    print(f'{total_accept_num = }')
    print(f'{total_autoregressive_token_num = }')
    accept_rate = total_accept_num / total_autoregressive_token_num
    print(f'{accept_rate = }')
    
    for _, queue in ssm_queue:
        queue.put(None)  # 发送停止信号
    for process, _ in ssm_queue:
        process.join()  # 等待进程结束
        
    questions_with_ids = []
    for i, question in enumerate(sentences):
        question_dict = {
            "question_id": i,  # 从0开始编号
            "model": os.path.basename(os.path.normpath(llm_model_path)),
            "prompt": sentences[i],
            "output": generated_text,
            "new_tokens": generate_tokens[i],
            "wall_time": wall_times[i],
            "mean_verify_time": round(dataset_infer_times[i],2),
            "rounds": spec_num,
            "accept_rate": accept_rate,
            "ratio": round(generate_tokens[i] / spec_num,2),
            "throughput": round(generate_tokens[i] / wall_times[i] * 1000,2),
            "temperature_ssm": temperature_ssm,
            "temperature_llm": temperature,
            "max_length": max_length, 
            "lookaheads": lookaheads,
        }
        questions_with_ids.append(question_dict)
    json_data = json.dumps(questions_with_ids, indent=4)

    with open('./outputs/batch_output_' + datasets[dataset_index], 'w') as json_file:
        json_file.write(json_data)