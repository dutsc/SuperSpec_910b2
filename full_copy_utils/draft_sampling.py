from transformers import AutoModelForCausalLM, AutoTokenizer
from .draft_kvcache import KVCacheModel
import torch
from .token_tree import TokenLogitsPair
from .utils import get_distribution
import time
from utils.logger import init_logger
logger = init_logger(__name__)

def draft_sample(gpu_id, task_queue, result_queue, config):
    # /share/models/opt-30b
    temperature = config['ssm']['temperature']
    model = AutoModelForCausalLM.from_pretrained(config['ssm']['model_device_map'][gpu_id],  
                                                torch_dtype=torch.float16,
                                                device_map="cuda:" + str(gpu_id))
    
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.4B-Chat')
    model = KVCacheModel(model,temperature=temperature) 
    while True:
        task_data = task_queue.get()
        if task_data is None:  # 检测停止信号
            break
        if config['timeline']:
            print(f"\n$process {gpu_id} task queue\t@{time.time()}")
            print(f"#process {gpu_id} infer prepare \t@{time.time()}")
        # event, initial_prompt_seq, lookahead, temperature, task_id, warm_up, clear_cache = task_data
        # initial_prompt_seq, lookahead, temperature, task_id, warm_up = task_data
        event, initial_prompt_seq, lookahead, clear_cache = task_data
        if clear_cache: 
            model._past_key_values = None
            model._logits_history = None
        fin_prompt_seq = initial_prompt_seq.detach().clone()
        pair_list = []
        beginTime = time.time_ns() / 1_000_000
        if config['timeline']:
            print(f"$process {gpu_id} infer prepare \t@{time.time()}")
        tokens_name = ['one', 'two', 'three', 'four']
        for j in range(lookahead):
        # gen_new_token = 0 # 用于统计生成的new_token的数量
        # while True:
            if config['timeline']:
                print(f'#process {gpu_id} infer {tokens_name[j%4]} \t@{time.time()}')
            draft_outputs = model.generate(fin_prompt_seq, 1)
            draft_logits = model._logits_history[:, -1:, :]
            draft_probs = get_distribution(draft_logits,temperature)
            sample_token = torch.multinomial(draft_probs.squeeze(1), num_samples=1)
            
            # draft_token = tokenizer.decode(sample_token[0],skip_special_tokens=True)
            # print(f'gpu_id={gpu_id}, draft_token={draft_token}')
            
            fin_prompt_seq = torch.concat([fin_prompt_seq, sample_token], dim=-1)
            pairs = TokenLogitsPair(sample_token, draft_probs.squeeze(1))
            pair_list.append(pairs)
            # gen_new_token += 1
            # if gpu_id == 2 and gen_new_token == lookahead:
            #     event.set()
            #     break
            # if gpu_id != 2:
            #     if event.is_set():
            #         break
            if config['timeline']:
                print(f'$process {gpu_id} infer {tokens_name[j%4]} \t@{time.time()}')
        model.rollback(initial_prompt_seq.size(1))
        result_queue.put((gpu_id, lookahead, pair_list))


def distribute_data(prompt, ssm_queue, draft_queue, event, clear_cache, config):
    lookaheads = config['lookaheads']
    SSM_GPUs = config['ssm']['GPUS']
    gpu_outputs = [[] for _ in SSM_GPUs]
    for i, (gpu, lookahead) in enumerate(zip(SSM_GPUs, lookaheads)):
        if config['timeline']:
            print(f'\n#prompt to gpu_{gpu}      \t@{time.time()}')
        input_ids = prompt.to(f'cuda:{gpu}', non_blocking=True)
        if config['timeline']:
            print(f'$prompt to gpu_{gpu}      \t@{time.time()}')
        with torch.no_grad():
            ssm_queue[i][1].put((event, input_ids, lookahead, clear_cache))  # 示例参数
            if config['timeline']:
                print(f"#process {i} task queue\t@{time.time()}")

    # 收集结果
    for i in range(len(SSM_GPUs)):
        task_id, gen_token_num, pair_list = draft_queue.get()
        # print(f'task_id:{task_id}, {pair_list[0].token.device = }')
        # print(f'task_id:{task_id}, {pair_list[0].logits.device = }')
        gpu_outputs[task_id].extend(pair_list)
    # 清除event
    # event.clear()
    return gpu_outputs