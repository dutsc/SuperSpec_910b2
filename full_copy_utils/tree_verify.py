from .token_tree import create_index_mapping, print_index_mapping, TreeNode, create_sequence_and_mask
import torch
import random
from .utils import get_distribution
import time

def compute_and_tree_verify(gpu_outputs, llm, prompt, config):
    llm_tem = config['llm']['temperature']
    LLM_GPU = config['llm']['GPU']
    if config['timeline']:
        print(f"#collect \t@{time.time()}")
    # 将数据从各副GPU收集到主GPU
    for outputs in gpu_outputs:
        for output in outputs:
            output.to(f'cuda:{LLM_GPU}', non_blocking=True)

    # 等待所有数据传输到主GPU完成
    torch.cuda.synchronize(LLM_GPU)
    if config['timeline']:
        print(f"$collect@{time.time()}")
        print(f"#llm prepare@{time.time()}")
    # 处理收集到的数据
    root = TreeNode()
    for i in gpu_outputs:
        root.add_sequence(i)
        
    # 打印出现在的output树
    root.print_tree()
    seq, mask = create_sequence_and_mask(root)
    new_token_len = len(seq)
    print(f'{new_token_len = }')
    
    seq = torch.tensor([seq], dtype = prompt.dtype, device = prompt.device)
    prompt = torch.cat([prompt, seq], dim=1)
    mask = mask.unsqueeze(dim=0)

    mask_N = prompt.shape[1]
    mask_init = 1 - torch.triu(torch.ones(mask.shape[0], mask_N, mask_N, dtype=float), diagonal=1).to(prompt.device)
    mask_init[:, -mask.shape[1]:, -mask.shape[2]:] = mask
    if config['timeline']:
        print(f"$llm prepare@{time.time()}")
        print(f"#llm forward@{time.time()}")
    
    # 调用一次大模型进行前向传播
    print(f'{prompt.shape = }')
    target_model = llm
    s1 = time.perf_counter_ns() / 1e6
    target_model._forward_with_kvcache(input_ids=prompt,attention_mask=mask_init)
    e1 = time.perf_counter_ns() / 1e6
    tree_infer_time = e1 - s1
    print(f'{tree_infer_time = }')
    if config['timeline']:
        print(f'verify finished      \t@{time.time()}')
    target_logits = target_model._logits_history[:, -new_token_len-1:, :]
    if config['timeline']:
        print(f"$llm forward          \t@{time.time()}")
        print(f"#llm result          \t@{time.time()}")
    target_logits = get_distribution(target_logits, llm_tem)
    
    if llm_tem == 0:
        verification, accept_num = verify_greedy(root,target_logits)
    else:
        verification, accept_num = verify_stochastic(root, target_logits)
    if config['timeline']:
        print(f"$llm result          \t@{time.time()}")
    return verification, accept_num, root, tree_infer_time

def verify_greedy(root, llm_logits):
    V = []
    u = root
    index_mapping = {}
    index_mapping[root]=-1
    create_index_mapping(root, index_mapping,0)
    # print_index_mapping(index_mapping)
    H = list(u.children.values()) 
    all_accepted=False
    accept_num = 1 # 初始值为1，因为即使小模型产生的所有token全部被拒绝，大模型也会至少产生一个token
    while len(H) > 0:
        s_idx = random.randint(0, len(H) - 1)
        chosen_child = H[s_idx]
        chosen_token = chosen_child.token_logits_pair.token
        llm_token = torch.multinomial(llm_logits[:,index_mapping[chosen_child.parent]+1, :],num_samples=1)
        if  llm_token == chosen_token:
            print("[accept]")
            V.append(chosen_token)
            # 为了rollback kvcache用
            chosen_child.mask = 1 
            u = chosen_child  # Move to the chosen child
            H = list(u.children.values()) 
            accept_num += 1
            if len(H)==0:
                all_accepted=True
        else:
            print("[reject]")
            H.remove(chosen_child)
    if all_accepted:
        append_new_token = torch.multinomial(llm_logits[:,index_mapping[chosen_child]+1,:], num_samples=1)
        V.append(append_new_token)
    else:
        resample_token = torch.multinomial(llm_logits[:,index_mapping[chosen_child.parent]+1,:], num_samples=1)
        V.append(resample_token) 
    return V, accept_num

def verify_stochastic(root, llm_logits):
    V = []  
    u = root  
    index_mapping = {}
    index_mapping[root]=-1
    all_accepted = False
    accept_num = 1
    accepted_index = 0
    create_index_mapping(root, index_mapping)
        
    H = list(u.children.values()) 
    # print(f'init_H.len= {len(H)}')
    while len(H) > 0:
        choices = random.choices(H, k=1)
        chosen_child = choices[0]
        t = chosen_child
        t_token = t.token_logits_pair.token

        ssmp = chosen_child.token_logits_pair.logits[:,t_token].squeeze(0) + 1e-9
        llmp = llm_logits[:,index_mapping[t.parent]+1, t_token].squeeze(0) + 1e-9
        
        ratio = llmp / ssmp
        # print(f'ratio: {ratio}')
        if ratio >= random.random():
            print("[accept]")
            accept_num += 1
            chosen_child.mask = 1
            V.append(t_token)
            u = chosen_child  # Move to the chosen child
            H = list(u.children.values()) 
            # print(f"accepted_index: {accepted_index}")
            accepted_index += 1
            if len(H)==0:
                all_accepted=True
        else:
            print("[reject]")
            llm_logits1 = llm_logits[:, index_mapping[t.parent]+1, :]
            ssm_logits1 = chosen_child.token_logits_pair.logits[:,:].squeeze(0)
            # print(f'{llm_logits1.shape = }')
            # print(f'{ssm_logits1.shape = }')
            
            new_dist = llm_logits1 - ssm_logits1
            
            new_dist = torch.where(new_dist > 0, new_dist, 0.0)
            # new_dist = new_dist / new_dist.sum()
            new_dist = torch.max(torch.zeros_like(new_dist), new_dist)
            new_dist = new_dist / new_dist.sum(dim=-1, keepdim=True)
            llm_logits[:, index_mapping[t.parent]+1, :] = new_dist
            # 从子节点中移除被拒绝的孩子节点
            H.remove(chosen_child)
    # 大模型已经归一化过，sample中重复归一化了
    # 最后都要增加一个大模型采样
    if all_accepted:
        new_token_logits = llm_logits[:, index_mapping[t]+1, :]  
        V.append(torch.multinomial(new_token_logits, num_samples=1))
    else:
        llm_logits1 = llm_logits[:, index_mapping[t.parent]+1, :]        
        token_id = torch.multinomial(llm_logits1, num_samples=1)  # 从new_dist的概率中进行随机采样
        V.append(token_id)
    return V, accept_num