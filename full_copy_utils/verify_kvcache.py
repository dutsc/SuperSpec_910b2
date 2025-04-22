import torch
from .logger import init_logger

logger = init_logger(__name__)


def get_distribution(logits, temperature):
    # 减去 logits 中的最大值
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits -= max_logits
    # 计算 softmax
    exp_logits = torch.exp(logits / (temperature + 1e-10))
    probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
    return probs


def _debug_show_kvcache(past_key_values):
    if past_key_values is None:
        return
    for elem in past_key_values:
        k, v = elem
        break


class KVCacheModel:
    def __init__(
        self,
        model: torch.nn.Module,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
    ) -> None:
        self._model = model
        self._past_key_values = None
        self._logits_history = None
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        self._tree_first = True

        # batch verify
        # self._batch_kvcache_buffer = None
        self._batch_logits_buffer = None
        self._batch_first = True
        self._new_key_values = None

    def _forward_with_kvcache(
        self,
        input_ids: torch.Tensor,
        use_debug=True,
        attention_mask=None,
        **model_kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask
        elif "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
        else:
            attention_mask = None

        if self._past_key_values is None:
            # 如果不为None 抛出异常 prefill时确保_logits_history为None
            # assert self._logits_history is None, f"{self._logits_history.shape}"
            # the first forward (prefill) returns the prompt's logits
            if attention_mask is not None:
                outputs = self._model(input_ids, attention_mask=attention_mask)
            else:
                outputs = self._model(input_ids)
            self._logits_history = outputs.logits
            self._past_key_values = outputs.past_key_values
            # self._new_key_values = outputs.new_key_values
            # print(f'{self._past_key_values = }')

            # print(f'{self._new_key_values[0][0].device = }')
            # print(f'{self._new_key_values[4][0].device = }')
            # print(f'{self._new_key_values[14][0].device = }')
            # print(f'{self._new_key_values[24][0].device = }')

            last_q = self._logits_history[:, -1, :]
        else:
            # 走这个分支 则表明之前已经缓存了kv 需要读取缓存的kv的长度 以确定缓存了多少个token
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[
                    2
                ]  # (batch_size, num_heads, sequence_length, embed_size_per_head)

            # print(f'{cached_len = }')
            last_input_id = input_ids[:, cached_len:]
            # print(f'not cached_len:{last_input_id.shape  = }')

            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            if use_debug:
                _debug_show_kvcache(self._past_key_values)

            if attention_mask is not None:
                outputs = self._model(
                    last_input_id,
                    past_key_values=self._past_key_values,
                    use_cache=True,
                    attention_mask=attention_mask,
                )
            else:
                outputs = self._model(
                    last_input_id, past_key_values=self._past_key_values, use_cache=True
                )
            # 未缓存的部分的输出
            not_cached_q = outputs.logits  # 这个地方确定对吗？？
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
            self._logits_history = torch.cat(
                [self._logits_history, not_cached_q], dim=1
            )
            last_q = not_cached_q[:, -1, :]
            self._past_key_values = outputs.past_key_values
            # self._new_key_values = outputs.new_key_values
            # print(f'{self._past_key_values[0][0].device = }')
            # print(f'{self._past_key_values[4][0].device = }')
            # print(f'{self._past_key_values[14][0].device = }')
            # print(f'{self._past_key_values[24][0].device = }')
            # print(f'{self._new_key_values[0][0].device = }')
            # print(f'{self._new_key_values[4][0].device = }')
            # print(f'{self._new_key_values[14][0].device = }')
            # print(f'{self._new_key_values[24][0].device = }')
        return last_q

    def _batch_verify_forward_with_kvcache(
        self,
        input_ids: torch.Tensor,
        use_debug=True,
        attention_mask=None,
        **model_kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask
        elif "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
        else:
            attention_mask = None
        if self._past_key_values is None:
            # assert self._logits_history is None, f"{self._logits_history.shape}"
            if attention_mask is not None:
                outputs = self._model(input_ids, attention_mask=attention_mask)
            else:
                outputs = self._model(input_ids)
            self._logits_history = outputs.logits
            # self._past_key_values = outputs.past_key_values
            self._new_key_values = outputs.new_key_values
            # print(f'{self._past_key_values = }')
            # print(f'{self._new_key_values[0][0].device = }')
            # print(f'{self._new_key_values[4][0].device = }')
            # print(f'{self._new_key_values[14][0].device = }')
            # print(f'{self._new_key_values[24][0].device = }')
            self._batch_first = True
            self._batch_logits_buffer = self._logits_history
            last_q = self._logits_history[:, -1, :]
        else:
            # 走这个分支 则表明之前已经缓存了kv 需要读取缓存的kv的长度 以确定缓存了多少个token
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[
                    2
                ]  # [batch_size, num_heads, sequence_length, embed_size_per_head]
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            if use_debug:
                # print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            if attention_mask is not None:
                outputs = self._model(
                    last_input_id,
                    past_key_values=self._past_key_values,
                    use_cache=True,
                    attention_mask=attention_mask,
                )
            else:
                outputs = self._model(
                    last_input_id, past_key_values=self._past_key_values, use_cache=True
                )
            not_cached_q = outputs.logits  # [bsz, lookahead+1, vocab_size]

            # 不会走这条分支 因为调用此函数时batchsize维度一定不为 1
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
            self._batch_logits_buffer = not_cached_q  # [3, K+1, embed_dim]
            self._new_key_values = outputs.new_key_values
            # print(f'{self._past_key_values[0][0].device = }')
            # print(f'{self._past_key_values[4][0].device = }')
            # print(f'{self._past_key_values[14][0].device = }')
            # print(f'{self._past_key_values[24][0].device = }')
            # print(f'{self._new_key_values[0][0].device = }')
            # print(f'{self._new_key_values[4][0].device = }')
            # print(f'{self._new_key_values[14][0].device = }')
            # print(f'{self._new_key_values[24][0].device = }')
            # last_q = not_cached_q[:, -1, :]
            last_q = self._batch_logits_buffer[:, -1, :]
        return last_q

    def _batch_verify_forward_with_kvcache_full_copy(
        self,
        input_ids: torch.Tensor,
        use_debug=True,
        attention_mask=None,
        **model_kwargs,
    ) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask
        elif "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
        else:
            attention_mask = None
        if self._past_key_values is None:
            # assert self._logits_history is None, f"{self._logits_history.shape}"
            if attention_mask is not None:
                outputs = self._model(input_ids, attention_mask=attention_mask)
            else:
                outputs = self._model(input_ids)
            self._logits_history = outputs.logits
            self._past_key_values = outputs.past_key_values
            self._batch_first = True
            last_q = self._logits_history[:, -1, :]
        else:
            # 走这个分支 则表明之前已经缓存了kv 需要读取缓存的kv的长度 以确定缓存了多少个token
            cached_len = 0
            for kv in self._past_key_values:
                k, v = kv
                cached_len = k.shape[
                    2
                ]  # [batch_size, num_heads, sequence_length, embed_size_per_head]
            last_input_id = input_ids[:, cached_len:]
            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)
            if use_debug:
                # print(f"last_input_id shape {last_input_id.shape}")
                _debug_show_kvcache(self._past_key_values)
            if attention_mask is not None:
                outputs = self._model(
                    last_input_id,
                    past_key_values=self._past_key_values,
                    use_cache=True,
                    attention_mask=attention_mask,
                )
            else:
                outputs = self._model(
                    last_input_id, past_key_values=self._past_key_values, use_cache=True
                )
            not_cached_q = outputs.logits  # [bsz, lookahead+1, vocab_size]

            # 不会走这条分支 因为调用此函数时batchsize维度一定不为 1
            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)
            self._logits_history = outputs.logits  # [3, total_len, embed_dim]
            self._past_key_values = outputs.past_key_values
            last_q = self._logits_history[:, -1, :]
        return last_q

    def _generate_with_kvcache(
        self,
        prefix: torch.Tensor,
        gamma: int,
        use_debug=False,
        **input_kwargs,
    ) -> torch.Tensor:
        """forward the model gamma times

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
            q = get_distribution(q, self._temperature)  # sc does
            next_tok = torch.multinomial(q, 1)
            x = torch.cat((x, next_tok), dim=1)
        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int, **input_kwargs) -> torch.Tensor:
        output = self._generate_with_kvcache(input, gamma, input_kwargs)
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int):
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

    @torch.no_grad()
    def rollback_with_mask(self, mask_seq):
        print("call in rollback_with_mask!")
        # if self._past_key_values[0][0] is not None:
        #     print(f'{self._past_key_values[0][0].shape = }')
        print(f"{self._new_key_values[0][0].shape = }")
        past_key_values_trimmed = []
        assert self._new_key_values
        seq_len_to_adjust = len(mask_seq)  # 需要调整的区间长度
        if self._tree_first:
            for kv in self._new_key_values:
                k, v = kv
                extended_mask_seq = [True] * (
                    k.shape[-2] - seq_len_to_adjust
                ) + mask_seq
                mask = torch.tensor(
                    extended_mask_seq, device=k.device, dtype=torch.bool
                )
                # print(f'{mask = }')
                k = k[:, :, mask, :]
                v = v[:, :, mask, :]
                # print(f'{k.device = }')
                kv_adjusted = (k, v)
                past_key_values_trimmed.append(kv_adjusted)
            self._tree_first = False
        else:
            for idx, kv in enumerate(self._new_key_values):
                k, v = kv
                extended_mask_seq = [True] * (
                    k.shape[-2] - seq_len_to_adjust
                ) + mask_seq
                mask = torch.tensor(
                    extended_mask_seq, device=k.device, dtype=torch.bool
                )
                # print(f'{mask = }')
                k = k[:, :, mask, :]
                v = v[:, :, mask, :]
                past_k, past_v = self._past_key_values[idx]
                # print(f'{past_k.device = }')
                # print(f'{k.device = }')
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)
                kv_adjusted = (k, v)
                past_key_values_trimmed.append(kv_adjusted)

        self._past_key_values = past_key_values_trimmed
        # print(f'{self._past_key_values[0][0].device = }')
        # print(f'{self._past_key_values[4][0].device = }')
        # print(f'{self._past_key_values[14][0].device = }')
        # print(f'{self._past_key_values[24][0].device = }')
        # 如果需要，也应相应调整self._logits_history TODO

    @torch.no_grad()
    def rollback_with_mask_(self, mask_seq):
        # print('call in rollback_with_mask!')
        past_key_values_trimmed = []
        assert self._past_key_values
        seq_len_to_adjust = len(mask_seq)  # 需要调整的区间长度

        for kv in self._past_key_values:
            k, v = kv
            # 计算调整区间的起始位置
            start_pos = max(0, k.shape[-2] - seq_len_to_adjust)
            assert k.shape[-2] >= seq_len_to_adjust
            # 创建一个全1的mask，长度为k的第三维度大小
            extended_mask_seq = [True] * (k.shape[-2] - seq_len_to_adjust) + mask_seq
            # 将extended_mask_seq转换成适合Tensor操作的布尔mask
            mask = torch.tensor(extended_mask_seq, device=k.device, dtype=torch.bool)
            # print(f'{mask.shape = }')
            # print(f'{mask = }')

            k_adjusted = k[:, :, mask, :]
            v_adjusted = v[:, :, mask, :]

            kv_adjusted = (k_adjusted, v_adjusted)
            past_key_values_trimmed.append(kv_adjusted)
        self._past_key_values = past_key_values_trimmed

    @torch.no_grad()
    def rollback_with_batch_winner(self, end_pos, batch_winner_index):
        # print(f'rollback_with_batch_winner!')
        past_key_values_trimmed = []
        assert self._new_key_values
        if self._batch_first:
            for idx, kv in enumerate(self._new_key_values):  # 遍历n_layers
                k, v = kv
                k = k[batch_winner_index].unsqueeze(0)[:, :, :end_pos, :]
                v = v[batch_winner_index].unsqueeze(0)[:, :, :end_pos, :]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            self._logits_history = self._batch_logits_buffer[
                batch_winner_index, :, :
            ].unsqueeze(0)[:, :end_pos, :]
            self._batch_first = False
            self._past_key_values = past_key_values_trimmed

        else:
            for idx, kv in enumerate(self._new_key_values):
                k, v = kv
                past_k, past_v = self._past_key_values[idx]
                k = torch.cat([past_k, k[batch_winner_index].unsqueeze(0)], dim=2)[
                    :, :, :end_pos, :
                ]
                v = torch.cat([past_v, v[batch_winner_index].unsqueeze(0)], dim=2)[
                    :, :, :end_pos, :
                ]
                kv_trimmed = (k, v)
                past_key_values_trimmed.append(kv_trimmed)
            self._logits_history = torch.cat(
                [
                    self._logits_history,
                    self._batch_logits_buffer[batch_winner_index, :, :].unsqueeze(0),
                ],
                dim=1,
            )[:, :end_pos, :]
            self._past_key_values = past_key_values_trimmed

    @torch.no_grad()
    def rollback_with_batch_winner_full_copy(self, end_pos, batch_winner_index):
        print(f"rollback_with_batch_winner_full_copy!")
        past_key_values_trimmed = []
        assert self._past_key_values
        for idx, kv in enumerate(self._past_key_values):  # 遍历n_layers
            k, v = kv
            bsz = k.shape[0]
            # 保存所有的kvcache
            k = (
                k[batch_winner_index]
                .unsqueeze(0)
                .repeat(bsz, 1, 1, 1)[:, :, :end_pos, :]
            )
            v = (
                v[batch_winner_index]
                .unsqueeze(0)
                .repeat(bsz, 1, 1, 1)[:, :, :end_pos, :]
            )
            kv_trimmed = (k, v)
            past_key_values_trimmed.append(kv_trimmed)
        self._logits_history = (
            self._logits_history[batch_winner_index]
            .unsqueeze(0)
            .repeat(bsz, 1, 1)[:, :end_pos, :]
        )
        self._past_key_values = past_key_values_trimmed
