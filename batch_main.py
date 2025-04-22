import torch
import time
import torch.multiprocessing as mp
from utils.batch_verify import (
    compute_and_batch_verify_greedy,
    compute_and_batch_verify_stochastic,
)
from utils.draft_sampling import distribute_data
from utils.utils import split_gpu_outputs, longest_common_prefix_length
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.layer_opt import OPTForCausalLM
from models.layer_qwen2 import Qwen2ForCausalLM

# from nsl_opt3 import OPTForCausalLM
from utils.draft_sampling import draft_sample
from utils.verify_kvcache import KVCacheModel
import os
import argparse
from utils.token_tree import create_mask_seq
from tqdm import tqdm
import json
import numpy as np
import gc


# 读取配置文件
def read_config():
    with open("./config.json", "r") as config_file:
        config = json.load(config_file)
    return config


config = read_config()

# 主程序
SSM_GPUs = config["ssm"]["GPUS"]  # 前三个GPU用于生成
LLM_GPU = config["llm"]["GPU"]  # 第四个GPU用于数据收集
temperature = config["llm"]["temperature"]
temperature_ssm = config["ssm"]["temperature"]
heuristic = config["heuristic"]
llm_model_path = config["llm"]["model"]

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
original_text = "Could you please give me some advice on programming?|<im_end>|"  # "Hello, my name is"


def warm_up_gpu(device, event) -> None:
    # 创建一些虚拟数据
    x = torch.rand((8192, 8192), device=device)
    # 执行一些轻量级的计算任务
    for _ in range(1000):
        y = x.mm(x)
    print("GPU warmed up!")
    event.set()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    # 存放所有小模型推理结果的队列
    draft_queue = mp.Queue()
    # 存放n个小模型进程
    ssm_queue = []
    # event用于表示draft model是否完成推理
    manager = mp.Manager()
    event = manager.Event()

    # 数据集
    folder = "/share/datasets/flexflow/"
    datasets = [
        "alpaca.json",
        "chatbot.json",
        "chatgpt.json",
        "piqa.json",
        "webqa.json",
    ]
    dataset_index = 0
    
    with open(folder + datasets[dataset_index], "r") as f:
        data = f.readlines()
        sentences = data[0][1:-1].split(",")
        sentences = [sentence.strip()[1:-1] for sentence in sentences]
    # print(f'{len(sentences) = }')
    sentences = sentences[:100]

    for gpu in SSM_GPUs:
        # 存放小模型的推理结果
        task_queue = mp.Queue()
        p = mp.Process(target=draft_sample, args=(gpu, task_queue, draft_queue, config))
        ssm_queue.append((p, task_queue))
        p.start()

    if "Qwen" in llm_model_path:
        init_llm = Qwen2ForCausalLM.from_pretrained(
            llm_model_path, torch_dtype=torch.float16, device_map="auto"
        )
    elif "opt" in llm_model_path:
        init_llm = OPTForCausalLM.from_pretrained(
            llm_model_path, torch_dtype=torch.float16, device_map="auto"
        )
    print("llm:", llm_model_path, "load successfully ! ")
    llm = KVCacheModel(init_llm, temperature)
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path)

    prompt = tokenizer(original_text, return_tensors="pt")["input_ids"].to(
        f"cuda:{LLM_GPU}"
    )

    # 设置生成的最大长度
    max_length = config["max_length"]
    lookaheads = config["lookaheads"]
    num_of_spec = 0
    eos_token_id = tokenizer.eos_token_id

    # warm up
    print("-----------------------------------warm up -----------------------------")
    for i in range(10):
        # print(f"wairm up {i}")
        gpu_outputs = distribute_data(
            prompt, ssm_queue, draft_queue, event, True, config
        )
        new_tokens, all_accepted, accept_num, batch_winner_index, batch_infer_time = (
            compute_and_batch_verify_greedy(gpu_outputs, llm, prompt, config)
        )
        if llm._past_key_values is not None:
            del llm._past_key_values
        if llm._new_key_values is not None:
            del llm._new_key_values
        if llm._logits_history is not None:
            del llm._logits_history
    print(
        f"-----------------------------------warm up end----------------------------------"
    )
    
    
    llm._past_key_values = None
    llm._new_key_values = None
    llm._logits_history = None

    total_generate_tokens = 0  # 该数据集所有prompt生成的token数量的累计
    total_infer_time = 0  # 该数据集所有pompt的端到端时延的累计
    total_accept_num = 0  # 尚未计算，应该是
    total_autoregressive_token_num = 0

    # 存储输出信息
    wall_times = []  # 该数据集，每个prompt的时延组成的列表
    generate_tokens = []  # 该数据集，每个prompt生成的token组成的列表
    generate_texts = []  ##该数据集，每个prompt生存的token转换为文本生成的列表
    dataset_infer_times = (
        []
    )  # 该数据集，每个prompt大模型verify时延（prompt大模型verify时延=每次投机推理的verify时间的均值，不是累计）组成的列表

    # time.sleep(5)
    i = 1
    
    with open("./outputs/opt/batch_output_" + datasets[dataset_index], "w") as json_file:
        for prompt in tqdm(sentences, desc="Processing sentences"):
            print(f"prompt  {i}  :", prompt)
            prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to(
                f"cuda:{LLM_GPU}"
            )
            # print(f"prompt  {i}  :", prompt)
            init_len = prompt.shape[1]
            clear_cache = True
            if llm._past_key_values is not None:
                del llm._past_key_values
            if llm._new_key_values is not None:
                del llm._new_key_values
            if llm._logits_history is not None:
                del llm._logits_history
            # gc.collect()
            llm._past_key_values = None
            llm._new_key_values = None
            llm._logits_history = None
            single_prompt_infer_times = (
                []
            )  # 该promtp的每次投机推理的大模型verify花费的时间的列表
            start_time = time.perf_counter_ns() / 1e6
            verify_i = 1
            spec_num = 1
            accept_nums = 0
            autoregressive_token_nums = 0
            while True:
                # print(f"-------------这是prompt  {i}  的第{verify_i}次核验----------------")

                epoch_start = time.time_ns() / 1_000_000
                gpu_outputs = distribute_data(
                    prompt, ssm_queue, draft_queue, event, clear_cache, config
                )

                # 本次投机推理第一轮跑过之后，不再清理kvcache
                clear_cache = False
                verify_start = time.time_ns() / 1_000_000
                tokens, logits = split_gpu_outputs(gpu_outputs, LLM_GPU)

                prefix_l = longest_common_prefix_length(tokens)
                # print(f"\nprefix_l: {prefix_l} \n")
                (
                    new_tokens,
                    all_accepted,
                    accept_num,
                    batch_winner_index,
                    batch_infer_time,
                ) = compute_and_batch_verify_greedy(gpu_outputs, llm, prompt, config)

                # [[198,1677],
                # [198,1677,1354],
                # [198]]
                single_prompt_infer_times.append(batch_infer_time)

                # print(f"{accept_num = }")
                # llm._past_key_values = None
                # llm._new_key_values = None
                # llm._logits_history = None

                # del llm._past_key_values
                # del llm._new_key_values
                # del llm._logits_history
                # gc.collect()

                # print(f'{batch_winner_index = }')
                accept_nums += accept_num - 1 # 减一因为大模型还会自己推一个
                total_autoregressive_token_num += max(lookaheads) #该数据集累计的小模型自回归次数
                #因为每个小模型并行，每投机推理自回归次数只要最大的lookahead
                autoregressive_token_nums += max(lookaheads)  #该prompt累计的小模型自回归次数
                # ..................  ...used to be ....................................
                # if all_accepted:
                #     llm.rollback_with_batch_winner(prompt.size(1) + len(new_tokens), batch_winner_index)
                # else:
                #     llm.rollback_with_batch_winner(prompt.size(1) + len(new_tokens) - 1, batch_winner_index)
                # .................... change................................................
                llm.rollback_with_batch_winner(
                    prompt.size(1) + len(new_tokens) - 1, batch_winner_index
                )
                # ....................................................................

                # 启发式变更lookahead  变更所有draft model的lookahead
                if heuristic:
                    if accept_num == lookaheads[0]:
                        lookaheads = [x + 2 for x in lookaheads]
                    else:
                        lookaheads = [max(1, x - 1) for x in lookaheads]

                for token in new_tokens:
                    if token == eos_token_id:  # 假设new_tokens是整数ID的列表
                        # print(
                        #     "*********************end because of eos*********************"
                        # )
                        break

                spec_generated_text = torch.zeros_like(new_tokens[0]).to(
                    f"cuda:{LLM_GPU}"
                )
                for token in new_tokens:
                    prompt = torch.cat([prompt, token], dim=-1)
                    spec_generated_text = torch.cat(
                        [spec_generated_text, token], dim=-1
                    )

                spec_generated_text = tokenizer.decode(
                    spec_generated_text[0], skip_special_tokens=True
                )
                # print(f"spec_generated_text = {spec_generated_text[1:]}")

                # 检查长度限制
                if prompt.size(1) >= max_length:
                    if config["timeline"]:
                        print(
                            f"-----------------------------------epoch {spec_num} end---------------------------------------"
                        )
                    # print(
                    #     "**********************end because of max_length****************************"
                    # )
                    break
                epoch_time = time.time_ns() / 1_000_000 - epoch_start
                if config["timeline"]:
                    print(
                        f"-----------------------------------epoch {spec_num} end---------------------------------------"
                    )
                spec_num += 1
                verify_i += 1
            end_time = time.perf_counter_ns() / 1e6
            # 将生成的tokens转换为文本
            generated_text = tokenizer.batch_decode(prompt, skip_special_tokens=True)
            print("Generated Text:", generated_text)
            
            #----------------------以上已经完成该prompt的推理---------------------------
            
            # 计算该prompt推理过程的各个指标
            prompt_generate_tokens = (
                prompt.shape[1] - init_len
            )  # 该prompt的生成token数量
            single_prompt_time = end_time - start_time  # 该promt的端到端时延
            seconds_per_token = (
                single_prompt_time / prompt_generate_tokens
            )  # 该prompt生成每个token花费的时间
            single_prompt_mean_verify_time = np.mean(
                single_prompt_infer_times
            )  # 该promt的大模型verify时间等于每次投机推理的
            dataset_infer_times.append(
                single_prompt_mean_verify_time
            )  # 加入该prompt的大模型核验时间
            accept_rate = accept_nums / autoregressive_token_nums #该prompt的接受率 = 接受数量 / 自回归产生token数
            
            #将本次prompt的指标汇总到整个数据集的指标
            total_infer_time += single_prompt_time       #数据集的端到端时延
            total_generate_tokens += prompt_generate_tokens   #数据集生成的总token数
            wall_times.append(single_prompt_time)  #该数据集中每个prompt的端到端时延
            generate_tokens.append(prompt_generate_tokens)  #该数据集每个prompt生成的token数
            generate_texts.append(generated_text) #该数据集的每个prompt生成的文本
            

            # 写入json文件
            # questions_with_ids = []
            question_dict = {
                "question_id": i-1,  # 从0开始编号
                "model": os.path.basename(os.path.normpath(llm_model_path)),
                "prompt": sentences[i-1],
                "output": generated_text,
                "new_tokens": prompt_generate_tokens,
                "wall_time": single_prompt_time,
                "mean_verify_time": round(single_prompt_mean_verify_time, 2),
                "rounds": spec_num,
                "accept_rate": accept_rate,
                "ratio": round(prompt_generate_tokens / spec_num, 2),
                "throughput": round(
                    prompt_generate_tokens / single_prompt_time * 1000, 2
                ),
                "temperature_ssm": temperature_ssm,
                "temperature_llm": temperature,
                "max_length": max_length,
                "lookaheads": lookaheads,
            }
            json_data = json.dumps([question_dict], indent=4)
            json_file.write(json_data + ",")
            i += 1

        print(f"{total_infer_time = } ms.")
        print(f"{total_generate_tokens = }")
        mean_per_token_infer_time = total_infer_time / total_generate_tokens
        print(f"{mean_per_token_infer_time = } ms/token.")
        print(f"{total_accept_num = }")
        print(f"{total_autoregressive_token_num = }")
        total_accept_rate = total_accept_num / total_autoregressive_token_num
        print(f"{total_accept_rate = }")

        for _, queue in ssm_queue:
            queue.put(None)  # 发送停止信号
        for process, _ in ssm_queue:
            process.join()  # 等待进程结束

        # questions_with_ids = []
        # for i, question in enumerate(sentences):
        #     question_dict = {
        #         "question_id": i,  # 从0开始编号
        #         "model": os.path.basename(os.path.normpath(llm_model_path)),
        #         "prompt": sentences[i],
        #         "output": generated_text,
        #         "new_tokens": generate_tokens[i],
        #         "wall_time": wall_times[i],
        #         "mean_verify_time": round(dataset_infer_times[i],2),
        #         "rounds": spec_num,
        #         "accept_rate": total_accept_rate,
        #         "ratio": round(generate_tokens[i] / spec_num,2),
        #         "throughput": round(generate_tokens[i] / wall_times[i] * 1000,2),
        #         "temperature_ssm": temperature_ssm,
        #         "temperature_llm": temperature,
        #         "max_length": max_length,
        #         "lookaheads": lookaheads,
        #     }
        #     questions_with_ids.append(question_dict)
        # json_data = json.dumps(questions_with_ids, indent=4)
