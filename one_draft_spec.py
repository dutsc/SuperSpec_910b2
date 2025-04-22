from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from full_copy_utils.verify_kvcache import KVCacheModel as vKVCacheModel
from full_copy_utils.draft_kvcache import KVCacheModel as dKVCacheModel
from utils.utils import get_distribution
from tqdm import tqdm
import time
import threading


def warm_up_gpu(device, event) -> None:
    x = torch.rand((8192, 8192), device=device)
    for _ in range(100):
        y = x.mm(x)
    event.set()
    print("GPU warmed up!")


if __name__ == "__main__":
    print("[start]")

    # 数据集
    folder = "/share/datasets/flexflow/"
    datasets = [
        "alpaca.json",
        "chatbot.json",
        "chatgpt.json",
        "piqa.json",
        "webqa.json",
    ]
    dataset_index = 4
    with open(folder + datasets[dataset_index], "r") as f:
        data = f.readlines()
        sentences = data[0][1:-1].split(",")
        sentences = [sentence.strip()[1:-1] for sentence in sentences]
    # print(f'{len(sentences) = }')
    sentences = sentences[:100]

    # input_text = "Give three tips for staying healthy."

    # 模型
    init_llm = AutoModelForCausalLM.from_pretrained(
        "/share/models/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    llm = vKVCacheModel(init_llm, 0)
    init_ssm = AutoModelForCausalLM.from_pretrained(
        "/share/models/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    ssm = dKVCacheModel(init_ssm, 0)
    tokenizer = AutoTokenizer.from_pretrained("/share/models/Qwen2.5-7B-Instruct")
    events = [threading.Event() for _ in range(4)]
    threads = []
    for i in range(4):
        device = torch.device(f"cuda:{i}")
        thread = threading.Thread(target=warm_up_gpu, args=(device, events[i]))
        threads.append(thread)
        thread.start()
    # 等待所有warm_up线程完成
    for event in events:
        event.wait()
    print("All GPUs are warmed up.")

    # 推理
    max_length = 128
    lookahead = 10
    total_generate_token_num = 0
    total_infer_time = 0
    total_accept_num = 0
    total_autoregressive_token_num = 0
    for prompt in tqdm(sentences, desc="Processing sentences"):
        prompt = tokenizer(prompt, return_tensors="pt")["input_ids"].to(f"cuda:{3}")
        init_len = prompt.shape[1]
        start = time.perf_counter_ns() / 1e6
        llm._past_key_values = None
        llm._logits_history = None
        ssm._past_key_values = None
        ssm._logits_history = None
        while prompt.shape[1] - init_len < max_length:

            # 小模型推理
            for i in range(lookahead):
                # ssm._forward_with_kvcache(prompt, use_debug = False)
                ssm.generate(prompt, 1)
                ssm_prob = ssm._logits_history[:, -1, :]  # [bs, 50272]
                ssm_logits = ssm._logits_history[:, -1, :]  # [bs, 50272]
                ssm_prob = get_distribution(ssm_logits, 0)
                ssm_token = torch.argmax(ssm_prob, dim=-1, keepdim=True)  # [bs, 1]
                prompt = torch.concat([prompt, ssm_token], dim=-1)
            total_autoregressive_token_num += lookahead
            # 大模型验证
            llm._forward_with_kvcache(prompt, use_debug=False)
            llm_logits = llm._logits_history[:, -1 - lookahead :, :]
            llm_prob = get_distribution(llm_logits, 0)

            llm_tokens = torch.argmax(llm_prob, dim=-1)
            true_indices = prompt[:, -lookahead:] == llm_tokens[:, -1 - lookahead : -1]
            all_true = torch.cumsum(
                torch.ones([1, lookahead], device=true_indices.device), dim=-1
            )
            false_idx = torch.sum(
                torch.cumsum(true_indices, dim=-1) == all_true, dim=-1
            ).item()
            total_accept_num += false_idx

            # 更新prompt
            prompt = torch.concat(
                [prompt[:, :-lookahead], llm_tokens[:, : false_idx + 1]], dim=-1
            )
            ssm.rollback(prompt.shape[1] - 1)
            llm.rollback(prompt.shape[1] - 1)

        end = time.perf_counter_ns() / 1e6
        infer_time = end - start
        generated_token_num = prompt.shape[1] - init_len
        per_token_infer_time = infer_time / generated_token_num
        print(f"{per_token_infer_time = } ms/token.")
        output_text = tokenizer.batch_decode(prompt, skip_special_tokens=True)
        total_infer_time += infer_time
        total_generate_token_num += generated_token_num
    print(f"{total_infer_time = } ms.")
    print(f"{total_generate_token_num = }")
    mean_per_token_infer_time = total_infer_time / total_generate_token_num
    print(f"{ mean_per_token_infer_time = } ms/token.")
    print(f"{total_accept_num = }")
    print(f"{total_generate_token_num = }")
    accept_rate = total_accept_num / total_generate_token_num
    print(f"{accept_rate = }")

    print("-" * 80)
    # print(output_text[0])
