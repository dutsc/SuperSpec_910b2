import torch
import torch.nn.functional as F
import numpy as np
import json

from utils.logger import init_logger
logger = init_logger(__name__)

    
def extract_q_and_a(file_path = "/datasets/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json"):
    with open(file_path, "r") as file:
        data = json.load(file)
    QandA = {}
    i = 0
    for item in data:
        conv = item["conversations"]
        if len(conv) >= 2:
            q = conv[0]["value"]
            a = conv[1]["value"]
            QandA[i] = {"question": q, "answer": a}
        i += 1
    return QandA


def get_distribution(logits, temperature):
    # 减去 logits 中的最大值
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits -= max_logits
    # 计算 softmax
    exp_logits = torch.exp(logits / (temperature + 1e-10))
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    return probs

def longest_common_prefix_length(gpu_outputs_tokens):
    # 确定所有列表中最少Tensor的数量
    min_length = min(len(tensors) for tensors in gpu_outputs_tokens)
    
    # for i in range(len(gpu_outputs_tokens)):
    #     for j in range(len(gpu_outputs_tokens[i])):
    #         print(gpu_outputs_tokens[i][j].item())
    # 初始化最长公共前缀长度为 0
    lcp_length = 0
    
    # 逐个位置比较
    for i in range(min_length):
        # 获取第一个列表中当前位置的Tensor值作为参考
        ref_value = gpu_outputs_tokens[0][i].cpu().item()
        
        # 检查其他所有列表在当前位置的Tensor值是否与参考值相同
        if all(tensors[i].cpu().item() == ref_value for tensors in gpu_outputs_tokens):
            lcp_length += 1  # 如果相同，增加公共前缀长度
        else:
            break  # 一旦发现不匹配，结束比较
    return lcp_length


def split_gpu_outputs(gpu_outputs,LLM_GPU):
    tokens=[]
    logits=[]
    for outputs in gpu_outputs:
        t_tokens=[]
        t_logits=[]
        for output in outputs:
            t_tokens.append(output.token)
            t_logits.append(output.logits.squeeze(0))
        t_logits=torch.stack(t_logits,dim=0).to(f'cuda:{LLM_GPU}', non_blocking=True)
        #print(f'aaaaaaaaaaaaa{t_logits.shape}')
        tokens.append(t_tokens)
        logits.append(t_logits)

    #print(logits[0][0].shape)
    #print(f'tokens shape ={tokens.shape}')
    return tokens,logits
    
COLORS = {
    "DEFAULT": "\x1b[0m",
    "BOLD": "\x1b[1m",
    "ITALIC": "\x1b[3m",
    "UNDERLINE": "\x1b[4m",
    "UNDERLINE_THICK": "\x1b[21m",
    "HIGHLIGHTED": "\x1b[7m",
    "HIGHLIGHTED_BLACK": "\x1b[40m",
    "HIGHLIGHTED_RED": "\x1b[41m",
    "HIGHLIGHTED_GREEN": "\x1b[42m",
    "HIGHLIGHTED_YELLOW": "\x1b[43m",
    "HIGHLIGHTED_BLUE": "\x1b[44m",
    "HIGHLIGHTED_PURPLE": "\x1b[45m",
    "HIGHLIGHTED_CYAN": "\x1b[46m",
    "HIGHLIGHTED_GREY": "\x1b[47m",
    "HIGHLIGHTED_GREY_LIGHT": "\x1b[100m",
    "HIGHLIGHTED_RED_LIGHT": "\x1b[101m",
    "HIGHLIGHTED_GREEN_LIGHT": "\x1b[102m",
    "HIGHLIGHTED_YELLOW_LIGHT": "\x1b[103m",
    "HIGHLIGHTED_BLUE_LIGHT": "\x1b[104m",
    "HIGHLIGHTED_PURPLE_LIGHT": "\x1b[105m",
    "HIGHLIGHTED_CYAN_LIGHT": "\x1b[106m",
    "HIGHLIGHTED_WHITE_LIGHT": "\x1b[107m",
    "STRIKE_THROUGH": "\x1b[9m",
    "MARGIN_1": "\x1b[51m",
    "MARGIN_2": "\x1b[52m",
    "BLACK": "\x1b[30m",
    "RED_DARK": "\x1b[31m",
    "GREEN_DARK": "\x1b[32m",
    "YELLOW_DARK": "\x1b[33m",
    "BLUE_DARK": "\x1b[34m",
    "PURPLE_DARK": "\x1b[35m",
    "CYAN_DARK": "\x1b[36m",
    "GREY_DARK": "\x1b[37m",
    "BLACK_LIGHT": "\x1b[90m",
    "RED": "\x1b[91m",
    "GREEN": "\x1b[92m",
    "YELLOW": "\x1b[93m",
    "BLUE": "\x1b[94m",
    "PURPLE": "\x1b[95m",
    "CYAN": "\x1b[96m",
    "WHITE": "\x1b[97m",
}


def colored(input_str: str, code: str):
    """
    wraps string into coloring escape sequences for colored printout to terminal
    based on https://stackoverflow.com/a/75054413/10396469
    usage: print(colored("this is blue", "BLUE"), colored("this is green", "GREEN"))
    for more options and cross-platform use consider colorama or yachalk
    """
    assert code.upper() in COLORS, f"invalid color code {code}"
    return COLORS[code.upper()] + input_str + COLORS["DEFAULT"]