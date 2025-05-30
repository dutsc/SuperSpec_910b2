# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch OPT model."""
from typing import List, Optional, Tuple, Union
import gc
import torch
import torch.utils.checkpoint
from torch import nn
import sys
import time
import torch.nn.functional as F

from transformers.activations import ACT2FN
from .utils.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    logging,
)
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.utils import is_flash_attn_2_available
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
# from models.attention.flash_attn_v2 import flash_attention_triton
# from flashinfer.prefill import single_prefill_with_kv_cache


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]

def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    if mask.dim() == 2: 
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    else: 
        bsz, src_len, _ = mask.size() 
        tgt_len = tgt_len if tgt_len is not None else src_len
        if tgt_len==src_len:
            expanded_mask = mask[:, None, :, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        else:
            expanded_mask = mask[:, None, :, :].expand(bsz, 1, src_len, src_len).to(dtype)[:,:,-tgt_len:,:]
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()
        if attention_mask.dim() == 2:
            # create positions depending on attention_mask
            positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        else: 
            positions = torch.count_nonzero(attention_mask, dim=2).type_as(attention_mask) - 1
        positions = positions[:, past_key_values_length:]
        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()#[bsz,num_heads,seq_len,head_dim]

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # past_key_value就是对应的old_kvcache 已经取到该层
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        #拼接cache
        if past_key_value is not None:
            k, v = past_key_value
            # 新计算出的k和v
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            new_key_len = key_states.shape[-2]
            if bsz == 1:
                cat_key_states = torch.cat([k, key_states], dim=2)
                cat_value_states = torch.cat([v, value_states], dim=2)
            else:  #复制bsz份
                cat_key_states = torch.cat([k.repeat(bsz,1,1,1), key_states], dim=2)  #(bsz,num_heads,tgt_len,head_dim)
                cat_value_states = torch.cat([v.repeat(bsz,1,1,1), value_states], dim=2)#(bsz,num_heads,tgt_len,head_dim)
            total_key_len = cat_key_states.shape[-2]
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            cat_key_states = key_states
            cat_value_states = value_states
            new_key_len = key_states.shape[-2]
            total_key_len = new_key_len
            
        new_key_value = (key_states, value_states)
        
        #调整形状，保证batch_metix_multiple顺利进行
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # (bsz*num_heads,tgt_len,head_dim)  其中tgt_len指新来的序列长度  
        cat_key_states = cat_key_states.view(*proj_shape) #(bsz*num_heads,seq_len,head_dim)
        cat_value_states = cat_value_states.view(*proj_shape) #(bsz*num_heads,seq_len,head_dim)
        src_len = cat_key_states.size(1) # seq_len == src_len
        #q*tra（k） 得到注意力分数
        attn_weights = torch.bmm(query_states, cat_key_states.transpose(1, 2))  # (bsz*num_heads,tgt_len,seq_len)
        
        #计算完注意力分数及时删除kcat_key_states
        del cat_key_states
        
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        #对注意力分数应用掩码
        if attention_mask is not None:
            # print(f'{attention_mask.shape = }')
            # print(f'{attention_mask[0][0][0] = }')
            # print(f'{attention_mask[0][0][1] = }')
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            #调整注意力分数的形状才能应用掩码
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            #再次调整注意力分数的形状为抽取k做准备
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        #用注意力分数抽取k得到output
        # (bsz*num_heads,tgt_len,seq_len) *  #(bsz*num_heads,seq_len,head_dim)
        attn_output = torch.bmm(attn_weights, cat_value_states)
        #(bsz*num_heads,tgt_len,head_dim)
        del cat_value_states
        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        #调整输出的形状，返回
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value, new_key_value
    
class OPTFlashAttention2(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()#[bsz,num_heads,seq_len,head_dim]

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # past_key_value就是对应的old_kvcache 已经取到该层
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, tgt_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling

        if past_key_value is not None:
            k, v = past_key_value
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            new_key_len = key_states.shape[-2]
            if bsz == 1:
                cat_key_states = torch.cat([k, key_states], dim=2)
                cat_value_states = torch.cat([v, value_states], dim=2)
            else:
                cat_key_states = torch.cat([k.repeat(bsz,1,1,1), key_states], dim=2)
                cat_value_states = torch.cat([v.repeat(bsz,1,1,1), value_states], dim=2)
            total_key_len = cat_key_states.shape[-2]
        else:
            
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            
            cat_key_states = key_states
            cat_value_states = value_states
            # print(f'{cat_key_states.shape = }')
            # print(f'{cat_value_states.shape = }')
            
            new_key_len = key_states.shape[-2]
            total_key_len = new_key_len
            
        new_key_value = (key_states, value_states)
        
        
        # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # (bsz*num_heads,tgt_len,head_dim)
        # # print(query_states.shape)
        
        # cat_key_states = cat_key_states.view(*proj_shape)
        # cat_value_states = cat_value_states.view(*proj_shape) #(bsz*num_heads,seq_len,head_dim)
        src_len = cat_key_states.size(1)
        
        
        # proj_shape = (-1, self.num_heads, self.head_dim)
        # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) 
        # cat_key_states = self._shape(cat_key_states, src_len, bsz).view(*proj_shape) 
        # cat_value_states = self._shape(cat_value_states, src_len, bsz).view(*proj_shape) 
        # mask_shape = (bsz, tgt_len, src_len)
        # print(f'{attention_mask.shape = }')
        # print(f'{cat_key_states.shape = }')
        # print(f'{cat_value_states.shape = }')
        
        # print(f'{mask_shape = }')
        
        # attention_mask = attention_mask.view(*mask_shape)
        # print(f'{attention_mask.shape = }')
        print(f'{cat_key_states.shape = }')
        print(f'{query_states.shape = }')
        
        causal = True
        scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=query_states.dtype))
        # attn_output = flash_attention_triton(query=query_states,
        #                                      key=cat_key_states,
        #                                      value=cat_value_states, 
        #                                      is_causal=causal, 
        #                                      scale=scale)
        
        query_length = tgt_len
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens
        
        print(f'{query_states.shape = }')
        print(f'{cat_key_states.shape = }')
        print(f'{cat_value_states.shape = }')
        print(f'{scale.shape = }')
        print(f'{max_seqlen_in_batch_q = }')
        print(f'{max_seqlen_in_batch_k = }')
        print(f'{cu_seqlens_q.shape = }')
        print(f'{cu_seqlens_k.shape = }')
        
        attn_output = flash_attn_varlen_func(query_states,
                                             cat_key_states,
                                             cat_value_states,
                                             softmax_scale=scale,
                                             causal=causal,
                                             max_seqlen_q=max_seqlen_in_batch_q,
                                             max_seqlen_k=max_seqlen_in_batch_k,
                                             cu_seqlens_q=cu_seqlens_q,
                                             cu_seqlens_k=cu_seqlens_k,)
        
        # 释放拼接复制的v 以释放其显存 实现按层复制
        # cat_value_states = None
        del cat_key_states
        del cat_value_states
        
        # if attn_output.size().item() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value, new_key_value
    
class OPTFlashinferAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()#[bsz,num_heads,seq_len,head_dim]

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,  # past_key_value就是对应的old_kvcache 已经取到该层
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        bsz, tgt_len, _ = hidden_states.size()
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        if past_key_value is not None:
            k, v = past_key_value
            
            # 新计算出的k和v
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            new_key_len = key_states.shape[-2]
            if bsz == 1:
                cat_key_states = torch.cat([k, key_states], dim=2)
                cat_value_states = torch.cat([v, value_states], dim=2)
            else:
                cat_key_states = torch.cat([k.repeat(bsz,1,1,1), key_states], dim=2)
                cat_value_states = torch.cat([v.repeat(bsz,1,1,1), value_states], dim=2)
            total_key_len = cat_key_states.shape[-2]
            
            # print(f'{key_states.device = }')
            # print(f'{cat_key_states.device = }')
            # print(f'{past_key_value[0].device = }')
        else:
            
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            
            cat_key_states = key_states
            cat_value_states = value_states
            # print(f'{cat_key_states.shape = }')
            # print(f'{cat_value_states.shape = }')
            
            new_key_len = key_states.shape[-2]
            total_key_len = new_key_len
            
            
        new_key_value = (key_states, value_states)
        
        
        # proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        # query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) # (bsz*num_heads,tgt_len,head_dim)
        # # print(query_states.shape)
        
        # cat_key_states = cat_key_states.view(*proj_shape)
        # cat_value_states = cat_value_states.view(*proj_shape) #(bsz*num_heads,seq_len,head_dim)
        src_len = cat_key_states.size(2)
        
        proj_shape = (-1, self.num_heads, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape) 
        cat_key_states = self._shape(cat_key_states, src_len, bsz).view(*proj_shape) 
        cat_value_states = self._shape(cat_value_states, src_len, bsz).view(*proj_shape) 
        mask_shape = (bsz, tgt_len, src_len)
        # print(f'{attention_mask.shape = }')
        # print(f'{cat_key_states.shape = }')
        # print(f'{cat_value_states.shape = }')
        
        # print(f'{mask_shape = }')
        
        attention_mask = attention_mask.view(*mask_shape)
        attn_output = single_prefill_with_kv_cache(q=query_states,
                                                   k=cat_key_states,
                                                   v=cat_value_states,
                                                   custom_mask=attention_mask,
                                                   kv_layout="NHD",
                                                   rope_scale=0,
                                                   )
        output_shape = (-1, tgt_len, self.head_dim)
        attn_output = attn_output.view(*output_shape)
        # 释放拼接复制的v 以释放其显存 实现按层复制
        # cat_value_states = None
        del cat_key_states
        del cat_value_states
        
        # if attn_output.size().item() != (bsz * self.num_heads, tgt_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, None, past_key_value, new_key_value
    

OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention,
    "flash_attention2": OPTFlashAttention2,  # bug
    "flashinfer": OPTFlashinferAttention,     # bug
} 

class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OPT_ATTENTION_CLASSES['eager'](
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            is_decoder=True,
            bias=config.enable_bias,
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        
        # Self Attention
        hidden_states, _, present_key_value, new_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (present_key_value, new_key_value)
        return outputs

class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (OPTDecoder)):
            module.gradient_checkpointing = value


class OPTDecoder(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        #print(f'{combined_attention_mask = }') # add by zwx
        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
        elif attention_mask.shape[1] != mask_seq_length:
            raise ValueError(
                f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                f"{mask_seq_length} (sum of the lengths of current and past inputs)"
            )
        causal_attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)
        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        present_key_value = () if use_cache else None
        new_key_values = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            ) # len(layer_outputs) = 3
            
            hidden_states = layer_outputs[0]

            if use_cache:
                present_key_value += (layer_outputs[1],)
                new_key_values += (layer_outputs[2],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = present_key_value if use_cache else None
        new_cache = new_key_values if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, new_cache] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            new_key_values=new_cache,
        )

class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )

class OPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)
        #print('use the nsl-opt,m3?')
        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        import time 
        s1 = time.perf_counter_ns() / 1e6   
        
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        e1 = time.perf_counter_ns() / 1e6
        infer_time = e1 - s1 
        # print(f'{infer_time = }')

        logits = self.lm_head(outputs[0]).contiguous()
        
        # if outputs.past_key_values[0] is not None:
        #     print(f'{outputs.past_key_values[0][0].device = }')
        #     print(f'{outputs.past_key_values[4][0].device = }')
        #     print(f'{outputs.past_key_values[14][0].device = }')
        #     print(f'{outputs.past_key_values[24][0].device = }')
        # print(f'{outputs.new_key_values[0][0].device = }')
        # print(f'{outputs.new_key_values[4][0].device = }')
        # print(f'{outputs.new_key_values[14][0].device = }')
        # print(f'{outputs.new_key_values[24][0].device = }')
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            new_key_values=outputs.new_key_values,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past