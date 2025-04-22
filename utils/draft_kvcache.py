import torch
from .logger import init_logger
logger = init_logger(__name__)

def get_distribution(logits, temperature):
    # 减去 logits 中的最大值
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits -= max_logits
    # 计算 softmax
    exp_logits = torch.exp((logits + 1e-5) / (temperature + 1e-5))
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    # print(f'{logits.shape = }')
    # print(f'{logits.dtype = }')
    
    # probs = torch.nn.functional.softmax(logits,dim=-1)
    return probs

def _debug_show_kvcache(past_key_values):
    if  past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        break

class KVCacheModel():
    def __init__(self, model : torch.nn.Module, temperature : float = 0, top_k : int = 0, top_p : float = 0) -> None:
        self._model = model
        self._past_key_values = None
        self._logits_history = None
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        
        # batch verify
        # self._batch_kvcache_buffer = None
        self._batch_logits_buffer = None
        self._batch_first = True
        self._new_key_values = None

    def _forward_with_kvcache(self, input_ids : torch.Tensor, use_debug = True, attention_mask = None, **model_kwargs) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask
        elif "attention_mask" in model_kwargs:
            attention_mask = model_kwargs['attention_mask']
        else:
            attention_mask = None
            
        if self._past_key_values is None:
            # 如果不为None 抛出异常 prefill时确保_logits_history为None
            assert self._logits_history is None, f"{self._logits_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            if attention_mask is not None:
                outputs = self._model(input_ids,attention_mask=attention_mask)
            else:
                outputs = self._model(input_ids)
            self._logits_history = outputs.logits
            self._past_key_values = outputs.past_key_values
            last_q = self._logits_history[:, -1, :]
        else:
            # 走这个分支 则表明之前已经缓存了kv 需要读取缓存的kv的长度 以确定缓存了多少个token
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[2]  # (batch_size, num_heads, sequence_length, embed_size_per_head)
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            
            if use_debug:
                _debug_show_kvcache(self._past_key_values)
            
            if attention_mask is not None:
                outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True, attention_mask=attention_mask)
            else:
                outputs = self._model(last_input_id, past_key_values=self._past_key_values, use_cache=True)
            # 未缓存的部分的输出
            not_cached_q = outputs.logits
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
            self._logits_history = torch.cat([self._logits_history, not_cached_q], dim=1)
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
        return last_q


    def _generate_with_kvcache(self, prefix : torch.Tensor, 
                                    gamma : int, 
                                    use_debug = False,
                                    **input_kwargs,
                                    ) -> torch.Tensor:
        """ forward the model gamma times

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses
            input_kwargs (dict): some parameterr like attention_mask

        Returns:
            Torch.Tensor: prefix+generated tokens
        """
        x = prefix

        for _ in range(gamma):
            q = self._forward_with_kvcache(x, use_debug, **input_kwargs)
            q = get_distribution(q,self._temperature)
            has_inf = torch.isinf(q).any()
            has_negative = (q < 0).any()
            has_nan = torch.isnan(q).any()

            # print("包含NaN元素:", has_nan.item())
            # print("包含inf元素:", has_inf.item())
            # print("包含小于0的元素:", has_negative.item())
            next_tok = torch.multinomial(q,1)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input : torch.Tensor, gamma : int, **input_kwargs) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma, input_kwargs)
        return output
    
    @torch.no_grad()
    def rollback(self, end_pos : int):
        past_key_values_trimmed = []
        assert self._past_key_values
        for kv in self._past_key_values:
            k, v = kv
            # NOTE() the indexing is specific for bloom. This won't work for other models
            # For example llama k, v should be (batch, num_head, seq_len, hidden_dim)
            
            # k, v (batch, head, seq, hidden_dim)
            k = k[:, :, :end_pos, :]
            v = v[:, :, :end_pos, :]
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        
        self._past_key_values = past_key_values_trimmed
        self._logits_history = self._logits_history[:, :end_pos, :]