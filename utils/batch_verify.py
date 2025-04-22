import torch
from random import random
from .utils import get_distribution
from .utils import split_gpu_outputs
import time

# TODO 
# 问题：
# 1. compute_and_batch_verify 生成结果与 compute_and_batch_verify_greedy 不一致
# 2. 两者的生成结果与 tree_verify 都不一致
# 解决方案：
# 1. 首先检查一下 compute_and_batch_verify 和 compute_and_batch_verify_greedy 的区别   ？？？？只看代码没区别
# 2. ✔ 检查验证时使用 _batch_logits_buffer 与使用 _logits_history 的区别，可能需要打印形状检查
# 3. ✔ 检查kvcache回滚操作的正确性
# 4. ✔ 检查验证logits与draft logits的对应关系

def compute_and_batch_verify_greedy(gpu_outputs, llm, prompt, config):
    llm_tem = config['llm']['temperature']
    LLM_GPU = config['llm']['GPU']
    len_ssms = len(config['ssm']['GPUS'])
    if config['timeline']:
        print(f"#collect \t@{time.time()}")
    draft_tokens,draft_probs = split_gpu_outputs(gpu_outputs,LLM_GPU)
    if config['timeline']:
        print(f"$collect \t@{time.time()}")
        print(f"#llm prepare@{time.time()}")
    
    draft_tokens = torch.tensor(draft_tokens).to(f'cuda:{LLM_GPU}', non_blocking=True)
    
    # [1,12] prompt shape
    # [3, 4] draft tokens
    prompt = prompt.repeat(len_ssms,1)
    # [3,12] prompt shape
    prompt = torch.concat([prompt,draft_tokens],dim=-1)
    # [3,16] prompt shape
    target_model = llm
    new_token_len = draft_tokens.shape[-1]  
    if config['timeline']:
        print(f"$llm prepare@{time.time()}")
        print(f"#llm forward@{time.time()}")
    start = time.perf_counter_ns() / 1e6
    target_model._batch_verify_forward_with_kvcache(input_ids=prompt)    #大模型forwad一次
    end = time.perf_counter_ns() / 1e6
    batch_infer_time = end - start
    # print(f'{batch_infer_time = } ms, {new_token_len = }')
    if config['timeline']:
        print(f"verify finished \t@{time.time()}")
    target_logits = target_model._batch_logits_buffer[:,-new_token_len-1:,:] #[bs, K+1, 50272]
    if config['timeline']:
        print(f"$llm forward          \t@{time.time()}")
        print(f"#llm result          \t@{time.time()}")
    target_prob = get_distribution(target_logits, llm_tem)
    V=[[] for _ in range(len_ssms)]
    for i in range(target_prob.shape[0]): # 分别验证每一个SSM的结果
        llm_prob = target_prob[i,:,:].squeeze(0) 
        all_accepted = True
        for j in range(new_token_len): # K
            draft_token = draft_tokens[i][j]
            llm_token = torch.multinomial(llm_prob[j,:], num_samples=1)
            if draft_token == llm_token:
                # print("[accept]")
                V[i].append(draft_token)
            else:
                # print("[reject]")
                all_accepted = False
                V[i].append(llm_token)
                break
        if all_accepted:
            V[i].append(torch.multinomial(llm_prob[-1,:], num_samples=1))

    # 如果长度相同，会返回idx较小的 也就是前面那个
    v_index = V.index(max(V,key=len))
    #print(f'v:{v}')
    v = V[v_index]
    v = [t.reshape(1,-1) for t in v]
    if config['timeline']:
        print(f"$llm result          \t@{time.time()}")
    return v,all_accepted,len(v),v_index, batch_infer_time




def compute_and_batch_verify_stochastic(gpu_outputs, llm, prompt, config):
    llm_tem = config['llm']['temperature']
    LLM_GPU = config['llm']['GPU']
    len_ssms = len(config['ssm']['GPUS'])
    if config['timeline']:
        print(f"#collect \t@{time.time()}")
    llm_tem = 1
    draft_tokens,draft_probs = split_gpu_outputs(gpu_outputs,LLM_GPU)
    # print(f'{len(draft_probs) = }') # SSM_nums
    # print(f'{draft_probs = }')
    if config['timeline']:
        print(f"$collect \t@{time.time()}")
        print(f"#llm prepare@{time.time()}")
    draft_tokens = torch.tensor(draft_tokens).to(f'cuda:{LLM_GPU}', non_blocking=True)
    draft_probs = torch.stack(draft_probs,dim=0)
    # print(f'{draft_probs.shape = }')
    
    prompt = prompt.repeat(len_ssms,1)
    prompt = torch.concat([prompt,draft_tokens],dim=-1)
    # print(f'{prompt.shape = }')
    new_token_len = draft_tokens.shape[-1]  
    target_model = llm
    if config['timeline']:
        print(f"$llm prepare@{time.time()}")
        print(f"#llm forward@{time.time()}")
    start = time.perf_counter_ns() / 1e6
    target_model._batch_verify_forward_with_kvcache(input_ids=prompt)
    end = time.perf_counter_ns() / 1e6
    batch_infer_time = end - start
    print(f'{batch_infer_time = } ms, {new_token_len}')
    if config['timeline']:
        print(f"verify finished \t@{time.time()}")
    target_logits = target_model._batch_logits_buffer[:, -new_token_len-1:, :]
    target_prob = get_distribution(target_logits, llm_tem)
    if config['timeline']:
        print(f"$llm forward          \t@{time.time()}")
        print(f"#llm result          \t@{time.time()}")
    
    V=[[] for _ in range(len_ssms)]
    # print(f'{len_ssms = }')
    # print(f'{target_prob.shape[0] = }')
    # print(f'{len(V) = }')
    # print(f'{V = }')
    
    for i in range(target_prob.shape[0]): # 分别验证每一个SSM的结果
        llm_prob = target_prob[i,:,:].squeeze(0) 
        ssm_prob = draft_probs[i,:,:].squeeze(0)
        # print(f'llm_logit.shape: {llm_prob.shape}')
        all_accepted = True
        for j in range(new_token_len): # K
            draft_token = draft_tokens[i][j]
            print(f'{draft_token = }')
            
            #print(f'ssm[j].shape {ssm_logit[j].shape}')
            llm_token = torch.multinomial(llm_prob[j,:], num_samples=1)
            
            llmp = llm_prob[j,:].gather(-1,draft_token) + 1e-9
            ssmp = ssm_prob[j,:].gather(-1,draft_token) + 1e-9
            
            # print(f'{llmp = }')
            # print(f'{ssmp = }')
            
            ratio = llmp / ssmp
            # print(f'{ratio = }')
            
            randnum = random.random()
            # print(randnum)
            if ratio >= randnum:
                print("[accept]")
                V[i].append(draft_token)
            else:
                print("[reject]")
                all_accepted = False
                # print(f'{llm_prob.shape = }')
                # print(f'{ssm_prob.shape = }')
                
                new_dist = llm_prob[j,:] - ssm_prob[j,:]
                # 这里已经全是0了??
                # print(torch.all(new_dist == 0))

                new_dist = torch.where(new_dist > 0, new_dist, 0.0) # 难道这里所有的值都是0了吗
                new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
                # print(f'{new_dist.shape = }')
                # print(torch.all(new_dist == 0))
                # llm_prob[j,:] = new_dist
                # print(f'{i = }')
                # print(f'{V[i] = }')
                
                V[i].append(torch.multinomial(new_dist, num_samples=1))
                break
        if all_accepted:
            V[i].append(torch.multinomial(llm_prob[-1,:], num_samples=1))
            
    v_index = V.index(max(V,key=len))
    #print(f'v:{v}')
    v = V[v_index]
    v = [t.reshape(1,-1) for t in v]
    if config['timeline']:
        print(f"$llm result          \t@{time.time()}")
    return v,all_accepted,len(v),v_index, batch_infer_time