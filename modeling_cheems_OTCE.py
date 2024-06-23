# coding=utf-8
# Copyright 2024 Jingze Shi and the HuggingFace Inc. team.    All rights reserved.
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
""" PyTorch Cheems OTCE model."""
import inspect
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
    is_mamba_ssm_available,
    is_causal_conv1d_available
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_cheems_OTCE import CheemsOTCEConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

if is_mamba_ssm_available():
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
else:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_fn, causal_conv1d_update = None, None


is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
)


logger = logging.get_logger(__name__)


def load_balancing_loss_func(
    gate_logits: torch.Tensor, 
    num_experts: torch.Tensor = None, 
    top_k=2, 
    attention_mask: Optional[torch.Tensor] = None
) -> float:
    r"""
    计算辅助负载平衡损失, 如Switch Transformer中所述 - 在Pytorch中实现.

    有关更多详细信息, 请参见Switch Transformer (https://arxiv.org/abs/2101.03961). 该函数实现了论文中方程(4) - (6)中呈现的损失函数.
    它的目的是惩罚专家之间路由太不平衡的情况.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            `router`的logits, 应该是一个形状为[batch_size X sequence_length, num_experts]的model.config.num_hidden_layers张量的元组.
        attention_mask (`torch.Tensor`, None):
            在forward函数中使用的attention_mask
            如果不为None, 形状为[batch_size X sequence_length].
        num_experts (`int`, *optional*):
            专家的数量

    Returns:
        辅助损失.


    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `router`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits if layer_gate.shape[1] > 1], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is not None:
        # 计算路由到每个专家的tokens百分比
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # 计算路由到这些专家的平均概率
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # 计算掩盖所有填充tokens为0的掩码, 其形状与expert_mask相同
        # Compute the mask that masks all padding tokens to 0, with the same shape as expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
        )

        # 计算路由到每个专家的tokens百分比
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # 计算掩盖所有填充tokens为0的掩码, 其形状与tokens_per_expert相同
        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
        )

        # 计算路由到这些专家的平均概率
        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    这是torch.repeat_interleave(x, dim=1, repeats=n_rep)的等效版本. 隐藏状态从(batch, num_key_value_heads, seqlen, head_dim)变为(batch, num_attention_heads, seqlen, head_dim)
    
    This is an equivalent version of torch.repeat_interleave(x, dim=1, repeats=n_rep). Hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """
    旋转输入的一半隐藏维度.
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_QK_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """
    将Rotary Position Embedding应用于查询和键张量.

    Args:
        q (`torch.Tensor`): 查询张量.
        k (`torch.Tensor`): 键张量.
        cos (`torch.Tensor`): 旋转嵌入的余弦部分.
        sin (`torch.Tensor`): 旋转嵌入的正弦部分.
        position_ids (`torch.Tensor`):
            与查询和键张量对应的令牌的位置索引. 例如, 这可以用于在使用KV缓存时传递偏移的位置id.
        unsqueeze_dim (`int`, *optional*, 默认为1):
            'unsqueeze_dim'参数指定沿其展开cos[position_ids]和sin[position_ids]的维度, 以便它们可以正确广播到q和k的维度. 例如, 请注意cos[position_ids]和sin[position_ids]的形状为[batch_size, seq_len, head_dim]. 然后, 如果q和k的形状为[batch_size, heads, seq_len, head_dim], 那么设置unsqueeze_dim=1使cos[position_ids]和sin[position_ids]可以广播到q和k的形状. 类似地, 如果q和k的形状为[batch_size, seq_len, heads, head_dim], 则设置unsqueeze_dim=2.
    Returns:
        旋转使用Rotary Position Embedding的查询和键张量的元组.
    Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_BC_rotary_pos_emb(b, c, cos, sin, position_ids):
    """
    将Rotary Position Embedding应用于B和C张量.

    Args:
        b (`torch.Tensor`): B张量. [batch_size, seq_len, ssm_state_size]
        c (`torch.Tensor`): C张量. [batch_size, seq_len, ssm_state_size]
        cos (`torch.Tensor`): 旋转嵌入的余弦部分.
        sin (`torch.Tensor`): 旋转嵌入的正弦部分.
        position_ids (`torch.Tensor`): 令牌的位置索引.
    Returns:
        旋转使用Rotary Position Embedding的B和C张量.

    Applies Rotary Position Embedding to the B and C tensors.

    Args:
        b (`torch.Tensor`): The B tensor. [batch_size, seq_len, ssm_state_size]
        c (`torch.Tensor`): The C tensor. [batch_size, seq_len, ssm_state_size]
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`): The position indices of the tokens.
    Returns:
        `tuple(torch.Tensor)` comprising of the B and C tensors rotated using the Rotary Position Embedding.
    """
    # cos 和 sin 的形状为 [batch_size, seq_len, dim]
    cos = cos[position_ids]
    sin = sin[position_ids]
    b_embed = (b * cos) + (rotate_half(b) * sin)
    c_embed = (c * cos) + (rotate_half(c) * sin)
    return b_embed, c_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 在这里构建以使`torch.jit.trace`工作
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        
        # 与论文不同, 但它使用不同的排列顺序以获得相同的计算
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype), 
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6, elementwise_affine: bool = True, bias: bool = True):
        """
        RMSNorm 是T5LayerNorm的等效
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        if isinstance(hidden_size, int):
            hidden_size = (hidden_size,)
        self.hidden_size = tuple(hidden_size)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.empty(self.hidden_size))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.hidden_size))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()
    

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)


    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # 权重和偏置
        # weight and bias
        if self.elementwise_affine:
            hidden_states = (hidden_states * self.weight).to(input_dtype)
            if self.bias is not None:
                hidden_states = (hidden_states + self.bias).to(input_dtype)

        return hidden_states


class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    一个动态缓存, 可以处理注意力缓存(具有seq_len维度)和mamba缓存(无论seq_len如何都具有恒定形状).

    此缓存有两组张量列表: `key_cache` 和 `value_cache` 用于注意力缓存, `conv_states` 和 `ssm_states` 用于mamba缓存.
    每个列表都有`num_layers`张张量. 每个张量的预期形状
    对于注意力层, `key_cache` 和 `value_cache` 的形状为`(batch_size, num_heads, seq_len, head_dim)`,
    而 `conv_states` 和 `ssm_states` 的形状为`(batch_size, 0)`(空张量).
    对于mamba层, `key_cache` 和 `value_cache` 的形状为`(batch_size, 0)`(空张量),
    而 `conv_states` 表示卷积状态, 形状为`(batch_size, d_inner, d_conv)`,
    而 `ssm_states` 表示ssm状态, 形状为`(batch_size, d_inner, d_state)`.
    
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors. The expected shape for each tensor
    For attention layers, `key_cache` and `value_cache` have a shape of `(batch_size, num_heads, seq_len, head_dim)`,
    while `conv_states` and `ssm_states` have a shape of `(batch_size, 0)` (empty tensors).
    For mamba layers, `key_cache` and `value_cache` have a shape of `(batch_size, 0)` (empty tensors),
    while `conv_states` represents the convolution state and has a shape of `(batch_size, d_inner, d_conv)`,
    and `ssm_states` represents the ssm state and has a shape of `(batch_size, d_inner, d_state)`.
    """

    def __init__(self, config: CheemsOTCEConfig, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.has_previous_state = False  # only used by mamba 只有mamba使用
        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
        conv_kernel_size = config.mamba_d_conv
        self.conv_states = []
        self.ssm_states = []
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "mamba":
                self.conv_states += [
                    torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
                ]
                self.ssm_states += [
                    torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]


    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 更新缓存
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


    def reorder_cache(self, beam_idx: torch.LongTensor):
        """
        重新排序缓存以进行beam搜索, 给定选择的beam索引.
        Reorders the cache for beam search, given the selected beam indices.
        """
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))


    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")


    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")


class CheemsAttention(nn.Module):
    """
    Multi-headed attention 来自 'Attention Is All You Need' 论文. 修改为使用滑动窗口注意力: Longformer 和 "Generating Long Sequences with Sparse Transformers".

    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: CheemsOTCEConfig, layer_idx:Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning(
                f"实例化 {self.__class__.__name__} 时没有传递 `layer_idx` 不推荐, 并且在使用缓存时会导致在前向调用期间出现错误. 请确保在创建此类时提供 `layer_idx`."

                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is deprecated and will lead to errors during forward calls when using caches. Please make sure to provide `layer_idx` when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size 必须能被 num_heads 整除 (得到 `hidden_size`: {self.hidden_size}"
                f" 和 `num_heads`: {self.num_heads})."

                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        self.q_proj = nn.Linear(
            self.hidden_size, 
            self.num_heads * self.head_dim, 
            bias=config.hidden_bias, 
        )
        self.k_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.hidden_bias, 
        )
        self.v_proj = nn.Linear(
            self.hidden_size, 
            self.num_key_value_heads * self.head_dim, 
            bias=config.hidden_bias, 
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, 
            self.hidden_size, 
            bias=config.hidden_bias, 
        )

        self.attention_rope = config.attention_rope
        if self.attention_rope:
            self.QK_rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2) # [bsz, num_key_value_heads, q_len, head_dim]

        kv_seq_len = key_states.size[-2]
        if self.attention_rope:
            cos, sin = self.QK_rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_QK_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)
        
        # 重复k/v头部, 如果n_kv_heads < n_heads
        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        offset = 64
        query_length = query_states.size(1)
        key_length = key_states.size(1)
        logn = torch.arange(offset+1, offset+key_length+1, dtype=torch.float32, device=query_states.device)[-query_length:] # [query_length]
        base = torch.tensor(256).to(query_states.device) # 训练数据的平均长度 Training data average length
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(query_states.dtype).view(1, query_length, 1, 1)
        query_states = query_states * logn

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it 不管长度如何, 我们只是切片它
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 将注意力上升到fp32
        # Upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output`应该是大小为{(bsz, self.num_heads, q_len, self.head_dim)}, 但是是{attn_output.size()}"

                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class CheemsFlashAttention2(CheemsAttention):
    """
    cheems flash attention 模块. 此模块继承自 `CheemsAttention`, 因为模块的权重保持不变. 唯一需要更改的是在前向传递中, 它需要正确调用flash attention的公共API, 并在输入包含任何填充标记的情况下处理它们.

    cheems flash attention module. This module inherits from `CheemsAttention` as the weights of the module remain the same. The only thing that needs to be changed is to correctly call the public API of flash attention in the forward pass and handle them in case the input contains any padding tokens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: 一旦RoCm的Flash Attention升级到2.1, 就应该删除这个. flash_attn<2.1生成左上对齐的因果掩码, 而这里需要的是右下对齐, 这是flash_attn>=2.1的默认设置. 这个属性用于处理这种差异. 参考: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # 请注意, 对于flash_attn<2.1, 使用q_seqlen != k_seqlen(除了q_seqlen == 1的情况)会产生一个错误的掩码(左上).
        # TODO: Remove this once RoCm's Flash Attention is upgraded to 2.1. flash_attn<2.1 generates a top-left aligned causal mask, while we need a bottom-right aligned one here, which is the default setting for flash_attn>=2.1. This attribute is used to handle this difference. Refer to: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Note that for flash_attn<2.1, using q_seqlen != k_seqlen (except for q_seqlen == 1) will produce a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.size[-2]
        # 由于输入可能被填充, 绝对序列长度取决于最大位置id.
        # Because the input can be padded, the absolute sequence length depends on the max position id.
        if self.attention_rope:
            rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
            cos, sin = self.QK_rotary_emb(value_states, seq_len=rotary_seq_len)
            query_states, key_states = apply_QK_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "当前的flash attention版本不支持滑动窗口注意力, 为了更高效的内存实现, 请确保升级flash-attn库."

                "The current version of flash attention does not support sliding window attention. For more memory-efficient implementation, make sure to upgrade the flash-attn library."
            )

        if past_key_value is not None:
            # 激活切片缓存, 只有在配置中有一个值`sliding_windows`属性时
            cache_has_contents = cache_position[0] > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"过去的键必须具有形状(`batch_size, num_heads, self.config.sliding_window-1, head_dim`), 得到{past_key.shape}"

                        f"Past keys must have shape (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)
            if self.attention_rope:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        # 如果n_kv_heads < n_heads, 重复k/v头
        # Repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # 在PEFT中, 通常我们为了训练稳定性的原因将层规范转换为float32, 因此输入隐藏状态会被静默地转换为float32. 因此, 我们需要将它们转换回float16, 以确保一切都按预期工作.
        # In PEFT, we usually convert layer norms to float32 for stability reasons, so input hidden states are silently converted to float32. Therefore, we need to convert them back to float16 to ensure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # 处理模型被量化的情况
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"输入隐藏状态似乎被静默地转换为float32, 这可能与您已经将嵌入或层规范层转换为float32有关. 我们将把输入转换回{target_dtype}."

                f"Input hidden states seem to have been silently converted to float32, which might be related to you already converting embeddings or layer norm layers to float32. We will convert the input back to {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 重新调整形状以符合Flash Attention的预期形状
        # Reshape to fit the expected shapes for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
 
        offset = 64
        query_length = query_states.size(1)
        key_length = key_states.size(1)
        logn = torch.arange(offset+1, offset+key_length+1, dtype=torch.float32, device=query_states.device)[-query_length:] # [query_length]
        base = torch.tensor(256).to(query_states.device)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(query_states.dtype).view(1, query_length, 1, 1)
        query_states = query_states * logn

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        告知Flash Attention的forward方法, 如果输入隐藏状态至少包含一个填充标记, 首先取消填充输入, 然后计算注意力分数并填充最终注意力分数.

        args:
            query_states (`torch.Tensor`):
                要传递给Flash Attention API的输入查询状态
            key_states (`torch.Tensor`):
                要传递给Flash Attention API的输入键状态
            value_states (`torch.Tensor`):
                要传递给Flash Attention API的输入值状态
            attention_mask (`torch.Tensor`):
                填充掩码 - 对应于大小为`(batch_size, seq_len)`的张量, 其中0表示填充标记的位置, 1表示非填充标记的位置.
            dropout (`int`, *optional*):
                注意力dropout
            softmax_scale (`float`, *optional*):
                在应用softmax之前对QK^T进行缩放. 默认为1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                是否激活滑动窗口注意力.
            
        Call the forward method of Flash Attention to first unpad the input if the input hidden states contain at least one padding token, then compute the attention scores and pad the final attention scores.

        args:
            query_states (`torch.Tensor`):
                Input query states to pass to the Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to pass to the Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to pass to the Flash Attention API
            attention_mask (`torch.Tensor`):
                Padding mask - tensor corresponding to size `(batch_size, seq_len)` where 0 represents the position of padding tokens and 1 represents the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                Scale to apply to QK^T before applying softmax. Default is 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: 一旦RoCm的Flash Attention升级到2.1, 就应该删除`query_length != 1`检查. 有关详细信息, 请参见LlamaFlashAttention2 __init__中的注释.
            # TODO: Remove the `query_length != 1` check once RoCm's Flash Attention is upgraded to 2.1. For more details, refer to the comments in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # 序列中至少包含一个填充标记
        # At least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output


    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # 在第一次迭代中, 我们需要通过在正确的位置切片它来正确重新创建填充掩码
        # In the first iteration, we need to correctly recreate the padding mask by slicing it at the right positions
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
            )  # 这里有一个memcpy, 这是非常糟糕的. This is very bad as there is a memcpy here.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # -q_len:切片假设左填充.
            # -q_len: slicing assumes left padding.
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


class CheemsSdpaAttention(CheemsAttention):
    """
    cheems attention 模块使用torch.nn.functional.scaled_dot_product_attention. 该模块继承自`CheemsAttention`, 因为模块的权重保持不变. 唯一的更改是在前向传递中, 以适应SDPA API.

    cheems attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from `CheemsAttention` as the weights of the module remain the same. The only thing that needs to be changed is to adapt to the SDPA API in the forward pass.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: 一旦实现了这一点, 通过例如`model.config.attn_implementation = "manual"`来改进这个警告.
            # TODO: Improve this warning by implementing it once, e.g. by setting `model.config.attn_implementation = "manual"`.
            logger.warning_once(
                "CheemsModel正在使用CheemsSdpaAttention, 但`torch.nn.functional.scaled_dot_product_attention`不支持`output_attentions=True`. 回退到手动注意力实现, 但是从Transformers版本v5.0.0开始, 将需要指定手动实现. 可以在加载模型时使用参数`attn_implementation='eager'`来删除此警告."

                "CheemsModel is using CheemsSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to manual attention implementation, but specifying manual implementation will be required starting from Transformers version v5.0.0. You can remove this warning by specifying manual implementation when loading the model using the parameter `attn_implementation='eager'`."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if self.attention_rope:
            cos, sin = self.QK_rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_QK_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        else:
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # 当具有自定义attn_mask的非连续输入时, 使用内存高效后端的SDPA目前(torch==2.1.2)存在错误, 参考: https://github.com/pytorch/pytorch/issues/112577.
        # When having non-contiguous input with custom attn_mask, there is currently (torch==2.1.2) a bug in the memory-efficient backend of SDPA, see:
        
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        offset = 64
        query_length = query_states.size(1)
        key_length = key_states.size(1)
        logn = torch.arange(offset+1, offset+key_length+1, dtype=torch.float32, device=query_states.device)[-query_length:]
        base = torch.tensor(256).to(query_states.device)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(query_states.dtype).view(1, query_length, 1, 1)
        query_states = query_states * logn

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # q_len > 1是必要的, 以匹配AttentionMaskConverter.to_causal_4d, 如果q_len == 1, 它不会创建一个因果掩码.
            # q_len > 1 is necessary to match AttentionMaskConverter.to_causal_4d, if q_len == 1, it won't create a causal mask.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


ATTENTION_CLASSES = {
    "eager": CheemsAttention,
    "flash_attention_2": CheemsFlashAttention2,
    "sdpa": CheemsSdpaAttention,
}


class CheemsMambaMixer(nn.Module):
    """
    计算∆, A, B, C和D状态空间参数, 并计算`contextualized_states`.
    A, D是独立于输入的(参见Mamba论文[1]第3.5.2节"对A的解释", 了解为什么A不是选择性的)
    ∆, B, C是依赖于输入的(这是Mamba和线性时不变S4之间的一个关键区别, 这就是为什么Mamba被称为**选择性**状态空间)

    Compute the ∆, A, B, C, and D state space parameters and calculate the `contextualized_states`.
    A, D are independent of the input (see Mamba paper [1] section 3.5.2 "Interpretation of A" to understand why A is not selective)
    ∆, B, C are dependent on the input (this is a key difference between Mamba and Linear Time-Invariant S4, which is why Mamba is called **selective** state space)
    """

    def __init__(self, config: CheemsOTCEConfig, layer_idx):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias
        self.max_position_embeddings = config.max_position_embeddings
        self.ropr_theta = config.rope_theta

        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = config.mamba_act
        self.act = ACT2FN[config.mamba_act]

        self.use_fast_kernels = config.use_mamba_kernels

        # 输入隐藏状态的投影
        # Projection of the input hidden states
        self.in_proj = nn.Linear(
            self.hidden_size, 
            self.intermediate_size * 2, 
            bias=self.use_bias, 
        )
 
        # 用于使dt, B和C依赖于输入的选择性投影
        # Selective projection to make dt, B, and C dependent on the input
        self.x_proj = nn.Linear(
            self.intermediate_size, 
            self.time_step_rank + self.ssm_state_size * 2, 
            bias=False, 
        )
        # 时间步投影(离散化)
        # Time step projection (discretization)
        self.dt_proj = nn.Linear(
            self.time_step_rank, 
            self.intermediate_size, 
            bias=self.use_bias, 
        )

        # S4D真实初始化. 这些不是离散化的!
        # 核心是加载它们, 计算离散状态, 然后写入更新的状态. 保持内存有界
        # Real initialization for S4D. These are not discretized!
        # The core is to load them, compute the discrete state, and then write the updated state. Keeping memory bounded
        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size, dtype=config.torch_dtype))
        self.out_proj = nn.Linear(
            self.intermediate_size, 
            self.hidden_size, 
            bias=self.use_bias, 
        )

        self.dt_layernorm = RMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.B_layernorm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.C_layernorm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

        self.mamba_rope = config.mamba_rope
        if self.mamba_rope:
            self.BC_rotary_emb = RotaryEmbedding(
                self.ssm_state_size, 
                max_position_embeddings=self.max_position_embeddings,
                base=self.ropr_theta,
            )
    
        if not is_fast_path_available:
            logger.warning_once(
                "快速路径不可用, 因为`(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`中的一个是None. 要安装, 请访问 https://github.com/state-spaces/mamba/#installation 和 https://github.com/Dao-AILab/causal-conv1d. 如果要使用朴素实现, 请在模型配置中设置`use_mamba_kernels=False`"

                "Fast path is not available because one of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)` is None. To install, visit https://github.com/state-spaces/mamba/#installation and https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model configuration."
            )


    def cuda_kernels_forward(
        self, 
        hidden_states: torch.Tensor, 
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: HybridMambaAttentionDynamicCache = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_params.conv_states[self.layer_idx].shape[0]
            == cache_params.ssm_states[self.layer_idx].shape[0]
            == batch_size
        )
        # 1. 门控MLP的线性投影
        # 1. Linear projection of the gated MLP
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # 我们不能在训练中使用`mamba_inner_fn`, 即使没有缓存参数, 因为我们有内部layernorms, 这不受此融合内核支持
        # We can't use `mamba_inner_fn` even if in training and without cache params because we have the inner layernorms which isn't supported by this fused kernel 
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. 卷积序列转换
        # 2. Convolutional sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if use_precomputed_states:
            hidden_states = causal_conv1d_update(
                hidden_states.squeeze(-1),
                cache_params.conv_states[self.layer_idx].to(hidden_states.dtype),
                conv_weights,
                self.conv1d.bias,
                self.activation,
            )
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(
                    hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx].copy_(conv_states)
            hidden_states = causal_conv1d_fn(
                hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
            )

        # 3. 状态空间模型序列转换
        # 3.a. 时间步, B和C的输入变化初始化
        # 3. State space model sequence transformation
        # 3.a. Initialization of input changes for time step, B, and C
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.B_layernorm(B)
        C = self.C_layernorm(C)

        # 旋转位置嵌入
        # Rotary position embeddings
        if self.mamba_rope:
            cos, sin = self.BC_rotary_emb(hidden_states, seq_len=seq_len)
            B, C = apply_BC_rotary_pos_emb(B, C, cos, sin, position_ids)
        
        offset = 64
        C_length = C.size(1)
        B_length = B.size(1)
        logn = torch.arange(offset+1, offset+B_length+1, dtype=torch.float32, device=hidden_states.device)[-C_length:]
        base = torch.tensor(256).to(hidden_states.device)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(hidden_states.dtype).view(1, C_length, 1)
        C = C * logn

        # 这里我们需要应用没有偏差的dt_proj, 因为偏差是在选择性扫描内核中添加的.
        # 这是一个应用dt_proj的hack, 同时仍然使用`torch.nn.Linear`的前向传递, 这是为了使量化工作.
        # 量化代码将`torch.nn.Linear`层替换为量化的线性层, 并要求直接调用前向传递.
        # 这里的原始代码是: ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
        # Here we need to apply dt_proj without bias, as the bias is added in the selective scan kernel.
        # This is a hack to apply dt_proj while still using the forward pass of `torch.nn.Linear`, which is necessary for quantization.
        # The quantization code replaces the `torch.nn.Linear` layer with a quantized linear layer and requires a direct call to the forward pass.
        # The original code here was: ```discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)```
        
        time_proj_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = discrete_time_step.transpose(1, 2)
        self.dt_proj.bias = time_proj_bias

        A = -torch.exp(self.A_log.float())
        # 3.c 执行循环 y ← SSM(A, B, C)(x)
        # 3.c Perform the loop y ← SSM(A, B, C)(x)
        time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
        if use_precomputed_states:
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        # 4. 最终线性投影
        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))

        return contextualized_states


    # fmt: off
    def slow_forward(
        self, 
        input_states, 
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: HybridMambaAttentionDynamicCache = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        # 1. 门控MLP的线性投影
        # 1. Linear projection of the gated MLP
        projected_states = self.in_proj(input_states).transpose(1, 2) # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        use_cache = isinstance(cache_params,HybridMambaAttentionDynamicCache)
        # 2. 卷积序列转换
        # 2. Convolutional sequence transformation
        if use_cache and cache_params.ssm_states[self.layer_idx].shape[0] == batch_size:
            if self.training:
                # 在训练模式下, 我们不希望对ssm_state执行原地操作, 以便我们可以计算反向传递
                # In training mode, we don't want to perform in-place operations on ssm_state so we can compute the backward pass
                ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]

            if cache_params.has_previous_state and seq_len == 1 and \
                    cache_params.conv_states[self.layer_idx].shape[0] == batch_size:
                conv_state = cache_params.conv_states[self.layer_idx] # [batch, intermediate_size, conv_kernel_size]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1) # [batch, intermediate_size, 1] : decoding
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len]) # [batch, intermediate_size, seq_len]
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len]) # [batch, intermediate_size, seq_len]

        # 3. 状态空间模型序列转换
        # 3.a. 选择: [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        # 3. State space model sequence transformation
        # 3.a. Select: [batch, seq_len, self.time_step_rank + self.ssm_state_size * 2]
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )
        time_step = self.dt_layernorm(time_step)
        B = self.B_layernorm(B) # [batch, seq_len, ssm_state_size]
        C = self.C_layernorm(C) # [batch, seq_len, ssm_state_size]

        # 旋转位置嵌入
        # Rotary position embeddings
        if self.mamba_rope:
            cos, sin = self.BC_rotary_emb(hidden_states, seq_len=seq_len)
            B, C = apply_BC_rotary_pos_emb(B, C, cos, sin, position_ids)

        offset = 64
        C_length = C.size(1)
        B_length = B.size(1)
        logn = torch.arange(offset+1, offset+B_length+1, dtype=torch.float32, device=hidden_states.device)[-C_length:]
        base = torch.tensor(256).to(hidden_states.device)
        logn = torch.log(logn) / torch.log(base)
        logn[logn < 1.0] = 1.0
        logn = logn.to(hidden_states.dtype).view(1, C_length, 1)
        C = C * logn

        discrete_time_step = self.dt_proj(time_step) # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2) # [batch, intermediate_size, seq_len]

        # 3.b. 离散化: B和C到[batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        # 3.b. Discretize: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.float()) # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None]) # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float() # [batch, intermediade_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        # 3.c 执行循环 y ← SSM(A, B, C)(x)
        # 3.c Perform the loop y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :] # [batch, intermediade_size, ssm_state]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1)) # [batch, intermediade_size, 1]
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1) # [batch, intermediade_size, seq_len]
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if use_cache:
            cache_params.ssm_states[self.layer_idx] = ssm_state

        # 4. 最终线性投影
        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2)) # [batch, seq_len, hidden_size]
        return contextualized_states
    # fmt: on


    def forward(
        self, 
        hidden_states: torch.Tensor, 
        position_ids: Optional[torch.LongTensor] = None,
        cache_params: HybridMambaAttentionDynamicCache = None
    ) -> torch.Tensor:
        if self.use_fast_kernels:
            if not is_fast_path_available or "cuda" not in self.x_proj.weight.device.type:
                raise ValueError(
                    "快速Mamba内核不可用. 确保它们已安装, 并且mamba模块在CUDA设备上"

                    "Fast Mamba kernels are not available. Make sure they are installed and the mamba module is on a CUDA device"
                )
            return self.cuda_kernels_forward(hidden_states, position_ids, cache_params)
        return self.slow_forward(hidden_states, position_ids, cache_params)


class CheemsMLP(nn.Module):
    def __init__(self, config: CheemsOTCEConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.gate_proj = nn.Linear(
            self.hidden_dim, 
            self.ffn_dim, 
            bias=config.hidden_bias,  
        )
        self.act_gate_fn = ACT2FN[config.hidden_act]

        self.up_proj = nn.Linear(
            self.hidden_dim, 
            self.ffn_dim, 
            bias=config.hidden_bias,  
        )
        self.act_up_fn = nn.Tanh()

        self.down_proj = nn.Linear(
            self.ffn_dim, 
            self.hidden_dim, 
            bias=config.hidden_bias, 
        )


    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.down_proj(self.act_up_fn(self.up_proj(hidden_states)) * self.act_gate_fn(self.gate_proj(hidden_states)))


# 共享参数的扩展MLP层
# Shared parameter Expansive MLP layer
class CheemsSharedExpansiveMLP(nn.Module):
    def __init__(self, config: CheemsOTCEConfig, shared_gate_proj:nn.Linear, shared_up_proj: nn.Linear, shared_down_proj: nn.Linear):
        super().__init__()
        self.private_expert_intermediate_dim = config.private_expert_intermediate_size
        self.hidden_dim = config.hidden_size

        # 共享参数的MLP
        # Shared parameter MLP
        self.shared_gate_proj = shared_gate_proj
        self.shared_up_proj = shared_up_proj
        self.shared_down_proj = shared_down_proj

        # 独立参数的MLP
        # Independent parameter MLP
        self.gate_proj = nn.Linear(
            self.hidden_dim, 
            self.private_expert_intermediate_dim, 
            bias=config.hidden_bias, 
        )
        self.gate_act_fn = ACT2FN[config.hidden_act]

        self.up_proj = nn.Linear(
            self.hidden_dim, 
            self.private_expert_intermediate_dim, 
            bias=config.hidden_bias, 
        )
        self.up_act_fn = nn.Tanh()

        self.down_proj = nn.Linear(
            self.private_expert_intermediate_dim, 
            self.hidden_dim, 
            bias=config.hidden_bias, 
        )

        self.shared_expert_gate = nn.Linear(
            self.hidden_dim,
            1,
            bias=False, 
        )

    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:

        hidden_states = self.shared_down_proj(self.up_act_fn(self.shared_up_proj(hidden_states)) * self.gate_act_fn(self.shared_gate_proj(hidden_states)))

        hidden_states = F.sigmoid(self.shared_expert_gate(hidden_states)) * hidden_states

        return self.down_proj(self.up_act_fn(self.up_proj(hidden_states)) * self.gate_act_fn(self.gate_proj(hidden_states)))


# 共享参数的内聚MLP层
# Shared parameter Cohesive MLP layer
class CheemsSharedCohesiveMLP(nn.Module):
    def __init__(self, config: CheemsOTCEConfig, shared_up_proj: nn.Linear):
        super().__init__()
        self.private_expert_intermediate_size = config.private_expert_intermediate_size
        self.hidden_dim = config.hidden_size

        self.gate_proj = nn.Linear(
            self.hidden_dim, 
            self.private_expert_intermediate_size, 
            bias=config.hidden_bias, 
        )
        self.gate_act_fn = ACT2FN[config.hidden_act]

        # 共享参数的Up Linear
        # Shared parameter Up Linear
        self.shared_up_proj = shared_up_proj
        self.act_up_fn = nn.Tanh()

        self.down_proj = nn.Linear(
            self.private_expert_intermediate_size, 
            self.hidden_dim, 
            bias=config.hidden_bias, 
        )


    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.down_proj(self.act_up_fn(self.shared_up_proj(hidden_states)) * self.gate_act_fn(self.gate_proj(hidden_states)))


# 共享参数的MOE层
# Shared parameter MOE layer
class CheemsSharedMoeLayer(nn.Module):
    """
    一个共享参数的MOE层, 以实现交叉领域的参数共享.

    A shared parameter MOE layer to achieve parameter sharing across domains.
    """
    def __init__(self, config: CheemsOTCEConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.shared_expert_intermediate_dim = config.shared_expert_intermediate_size
        self.expert_type = config.expert_type
        self.private_expert_intermediate_dim = config.private_expert_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        # 专家路由
        # Expert routing
        self.router = nn.Linear(
            self.hidden_dim, 
            self.num_experts, 
            bias=False, 
        )

        # 专家
        # Experts
        if self.expert_type == "expansive":
            # 共享参数的Gate Linear
            # Shared parameter Gate Linear
            self.shared_gate_proj = nn.Linear(
                self.hidden_dim,
                self.shared_expert_intermediate_dim, 
                bias=config.hidden_bias,
            )
            # 共享参数的Up Linear
            # Shared parameter Up Linear
            self.shared_up_proj = nn.Linear(
                self.hidden_dim, 
                self.shared_expert_intermediate_dim, 
                bias=config.hidden_bias, 
            )
            # 共享参数的Down Linear
            # Shared parameter Down Linear
            self.shared_down_proj = nn.Linear(
                self.shared_expert_intermediate_dim,
                self.hidden_dim, 
                bias=config.hidden_bias, 
            )
            self.experts = nn.ModuleList([CheemsSharedExpansiveMLP(config, self.shared_gate_proj, self.shared_up_proj, self.shared_down_proj) for _ in range(self.num_experts)])
        elif self.expert_type == "cohesive":
            # 共享参数的Up Linear
            # Shared parameter Up Linear
            self.shared_up_proj = nn.Linear(
                self.hidden_dim, 
                self.shared_expert_intermediate_dim,
                bias=config.hidden_bias, 
            )
            self.experts = nn.ModuleList([CheemsSharedCohesiveMLP(config, self.shared_up_proj) for _ in range(self.num_experts)])
        else:
            self.experts = nn.ModuleList([CheemsMLP(config) for _ in range(self.num_experts)])


    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        # 我们回到输入的dtype
        # We go back to the dtype of the input
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 对选定的专家进行独热编码, 以创建一个专家掩码, 这将用于轻松索引哪个专家将被请求
        # One-hot encode the selected experts to create an expert mask, which will be used to easily index which expert will be requested
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # 循环遍历模型中的所有可用专家, 并在每个专家上执行计算
        # Loop through all available experts in the model and perform computations on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # 对正确的隐藏状态进行索引, 并计算当前专家的专家隐藏状态. 我们需要确保将输出隐藏状态乘以`routing_weights`在相应的token上(top-1和top-2)
            # Index the correct hidden states and calculate the expert hidden states for the current expert. We need to make sure to multiply the output hidden states by `routing_weights` at the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # 但是`index_add_`只支持torch张量进行索引, 因此我们将在这里使用`top_x`张量.
            # However, `index_add_` only supports indexing with torch tensors, so we will use the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


# 观察层
# Observation layer
class CheemsObservationLayer(nn.Module):
    def __init__(self, config: CheemsOTCEConfig, num_experts: int, layer_idx: int):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = CheemsMambaMixer(config=config, layer_idx=layer_idx)

        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ffn_layer_class = CheemsSharedMoeLayer if num_experts > 1 else CheemsMLP
        self.feed_forward = ffn_layer_class(config)
       
        self.hidden_dropout = config.hidden_dropout


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 形状为`(batch, seq_len, embed_dim)`的层输入
            attention_mask (`torch.FloatTensor`, *optional*): 大小为`(batch, sequence_length)`的注意力掩码, 其中填充元素用0表示
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的过去键和值投影状态
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量. 有关更多详细信息, 请参见返回的张量下的`attentions`
            output_router_logits (`bool`, *optional*):
                是否返回所有路由器的对数. 它们对于计算路由器损失很有用, 在推理期间不应返回.
            use_cache (`bool`, *optional*):
                如果设置为`True`, `past_key_values`键值状态将被返回, 并且可以用于加速解码(参见`past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                代表序列中输入序列标记的位置的索引.
        
            Args:
                hidden_states (`torch.FloatTensor`): Layer input of shape `(batch, seq_len, embed_dim)`
                attention_mask (`torch.FloatTensor`, *optional*): Attention mask of size `(batch, sequence_length)` where padding elements are represented by 0
                past_key_value (`Tuple(torch.FloatTensor)`, *optional*): Cached past key and value projection states
                output_attentions (`bool`, *optional*):
                    Whether to return attention tensors of all attention layers. See `attentions` under returned tensors for more details
                output_router_logits (`bool`, *optional*):
                    Whether to return logits of all routers. They are useful for computing router loss and should not be returned during inference.
                use_cache (`bool`, *optional*):
                    If set to `True`, `past_key_values` key value states will be returned and can be used to speed up decoding (see `past_key_values`).
                cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                    Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache_params=past_key_value,
        )
        self_attn_weights = None
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # mamba后的残差连接
        # Residual connection after mamba
        hidden_states = residual + hidden_states

        # 前馈
        # Feed forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, router_logits = ff_outputs
        else:
            hidden_states, router_logits = ff_outputs, None
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # 前馈后的残差连接
        # Residual connection after feed forward
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# 思考层
# Thinking layer
class CheemsThinkingLayer(nn.Module):
    def __init__(self, config: CheemsOTCEConfig, num_experts: int, layer_idx: int):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ffn_layer_class = CheemsSharedMoeLayer if num_experts > 1 else CheemsMLP
        self.feed_forward = ffn_layer_class(config)   

        self.hidden_dropout = config.hidden_dropout


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 形状为`(batch, seq_len, embed_dim)`的层输入
            attention_mask (`torch.FloatTensor`, *optional*): 大小为`(batch, sequence_length)`的注意力掩码, 其中填充元素用0表示
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的过去键和值投影状态
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量. 有关更多详细信息, 请参见返回的张量下的`attentions`
            output_router_logits (`bool`, *optional*):
                是否返回所有路由器的对数. 它们对于计算路由器损失很有用, 在推理期间不应返回.
            use_cache (`bool`, *optional*):
                如果设置为`True`, `past_key_values`键值状态将被返回, 并且可以用于加速解码(参见`past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                代表序列中输入序列标记的位置的索引.

            Args:
                hidden_states (`torch.FloatTensor`): Layer input of shape `(batch, seq_len, embed_dim)`
                attention_mask (`torch.FloatTensor`, *optional*): Attention mask of size `(batch, sequence_length)` where padding elements are represented by 0
                past_key_value (`Tuple(torch.FloatTensor)`, *optional*): Cached past key and value projection states
                output_attentions (`bool`, *optional*):
                    Whether to return attention tensors of all attention layers. See `attentions` under returned tensors for more details
                output_router_logits (`bool`, *optional*):
                    Whether to return logits of all routers. They are useful for computing router loss and should not be returned during inference.
                use_cache (`bool`, *optional*):
                    If set to `True`, `past_key_values` key value states will be returned and can be used to speed up decoding (see `past_key_values`).
                cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                    Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # 注意力后的残差连接
        # Residual connection after attention
        hidden_states = residual + hidden_states

        # 前馈
        # Feed forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, router_logits = ff_outputs
        else:
            hidden_states, router_logits = ff_outputs, None
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # 前馈后的残差连接
        # Residual connection after feed forward
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# 构思层
# Conception layer
class CheemsConseptionLayer(nn.Module):
    def __init__(self, config: CheemsOTCEConfig, num_experts: int, layer_idx: int):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mamba = CheemsMambaMixer(config=config, layer_idx=layer_idx)

        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ffn_layer_class = CheemsSharedMoeLayer if num_experts > 1 else CheemsMLP
        self.feed_forward = ffn_layer_class(config)

        self.hidden_dropout = config.hidden_dropout


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HybridMambaAttentionDynamicCache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 形状为`(batch, seq_len, embed_dim)`的层输入
            attention_mask (`torch.FloatTensor`, *optional*): 大小为`(batch, sequence_length)`的注意力掩码, 其中填充元素用0表示
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的过去键和值投影状态
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量. 有关更多详细信息, 请参见返回的张量下的`attentions`
            output_router_logits (`bool`, *optional*):
                是否返回所有路由器的对数. 它们对于计算路由器损失很有用, 在推理期间不应返回.
            use_cache (`bool`, *optional*):
                如果设置为`True`, `past_key_values`键值状态将被返回, 并且可以用于加速解码(参见`past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                代表序列中输入序列标记的位置的索引.

            Args:
                hidden_states (`torch.FloatTensor`): Layer input of shape `(batch, seq_len, embed_dim)`
                attention_mask (`torch.FloatTensor`, *optional*): Attention mask of size `(batch, sequence_length)` where padding elements are represented by 0
                past_key_value (`Tuple(torch.FloatTensor)`, *optional*): Cached past key and value projection states
                output_attentions (`bool`, *optional*):
                    Whether to return attention tensors of all attention layers. See `attentions` under returned tensors for more details
                output_router_logits (`bool`, *optional*):
                    Whether to return logits of all routers. They are useful for computing router loss and should not be returned during inference.
                use_cache (`bool`, *optional*):
                    If set to `True`, `past_key_values` key value states will be returned and can be used to speed up decoding (see `past_key_values`).
                cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                    Indices depicting the position of the input sequence tokens in the sequence.
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.mamba(
            hidden_states=hidden_states,
            position_ids=position_ids,
            cache_params=past_key_value,
        )
        self_attn_weights = None
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # mamba后的残差连接
        # Residual connection after mamba
        hidden_states = residual + hidden_states

        # 前馈
        # Feed forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        ff_outputs = self.feed_forward(hidden_states)
        if isinstance(ff_outputs, tuple):
            hidden_states, router_logits = ff_outputs
        else:
            hidden_states, router_logits = ff_outputs, None
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # 前馈后的残差连接
        # Residual connection after feed forward
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# 表达层
# Expression layer
class CheemsExpressionLayer(nn.Module):
    
    def __init__(self, config: CheemsOTCEConfig, layer_idx: int):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attention = ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.pre_ff_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        ffn_layer_class = CheemsMLP
        self.feed_forward = ffn_layer_class(config)

        self.hidden_dropout = config.hidden_dropout

        self.pre_dense_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dense = nn.Linear(
            config.hidden_size, 
            config.hidden_size, 
            bias=config.hidden_bias, 
        )
        self.activation = ACT2FN[config.hidden_act]
        self.projection = nn.Linear(
            config.hidden_size, 
            config.hidden_size, 
            bias=config.hidden_bias, 
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): 形状为`(batch, seq_len, embed_dim)`的层输入
            attention_mask (`torch.FloatTensor`, *optional*): 大小为`(batch, sequence_length)`的注意力掩码, 其中填充元素用0表示
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): 缓存的过去键和值投影状态
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量. 有关更多详细信息, 请参见返回的张量下的`attentions`
            output_router_logits (`bool`, *optional*):
                是否返回所有路由器的对数. 它们对于计算路由器损失很有用, 在推理期间不应返回.
            use_cache (`bool`, *optional*):
                如果设置为`True`, `past_key_values`键值状态将被返回, 并且可以用于加速解码(参见`past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                代表序列中输入序列标记的位置的索引.

            Args:
                hidden_states (`torch.FloatTensor`): Layer input of shape `(batch, seq_len, embed_dim)`
                attention_mask (`torch.FloatTensor`, *optional*): Attention mask of size `(batch, sequence_length)` where padding elements are represented by 0
                past_key_value (`Tuple(torch.FloatTensor)`, *optional*): Cached past key and value projection states
                output_attentions (`bool`, *optional*):
                    Whether to return attention tensors of all attention layers. See `attentions` under returned tensors for more details
                output_router_logits (`bool`, *optional*):
                    Whether to return logits of all routers. They are useful for computing router loss and should not be returned during inference.
                use_cache (`bool`, *optional*):
                    If set to `True`, `past_key_values` key value states will be returned and can be used to speed up decoding (see `past_key_values`).
                cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                    Indices depicting the position of the input sequence tokens in the sequence.
        """

        # 保存原始隐藏状态
        # Save the original hidden states
        original_hidden_states = hidden_states

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # 注意力后的残差连接
        # Residual connection after attention
        hidden_states = residual + hidden_states

        # 前馈
        # Feed forward
        residual = hidden_states
        hidden_states = self.pre_ff_layernorm(hidden_states)
        hidden_states, router_logits = self.feed_forward(hidden_states), None
        # 失活
        # Dropout
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # 前馈后的残差连接
        # Residual connection after feed forward
        hidden_states = residual + hidden_states

        # 与原始隐藏状态相加
        # Add with the original hidden states
        hidden_states = self.pre_dense_layernorm(hidden_states)
        hidden_states = hidden_states + original_hidden_states
        # 稠密激活
        # Dense activation
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.hidden_dropout, training=self.training)
        # 最终线性投影
        # Final linear projection
        hidden_states = self.projection(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class CheemsOTCEPreTrainedModel(PreTrainedModel):
    config_class = CheemsOTCEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CheemsObservationLayer", "CheemsThinkingLayer", "CheemsConseptionLayer", "CheemsExpressionLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True


    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Embedding(torch.nn.Module):
    def __init__(self, config: CheemsOTCEConfig):
        super(Embedding, self).__init__()
        
        self.hidden_size = config.hidden_size
        # 单词嵌入(并行).
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.vocab_size, 
            self.hidden_size, 
            padding_idx=config.pad_token_id, 
        )
        # 类型嵌入.
        # Type embeddings.
        if config.type_vocab_size is not None:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, 
                self.hidden_size, 
            )
        else:
            self.token_type_embeddings = None


    def forward(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 词嵌入.
        # Word embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        # 类型嵌入. 本模型的tokenizer不包含类型嵌入.手动添加.
        # Type embeddings. The tokenizer for this model does not include type embeddings. Add them manually.
        if self.token_type_embeddings is not None:
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long, device=input_ids.device)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            words_embeddings += token_type_embeddings
        
        return words_embeddings


class CheemsOTCEModel(CheemsOTCEPreTrainedModel):

    def __init__(self, config: CheemsOTCEConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = Embedding(config)

        decoder_layers = []

        # OTCE: observation + thinking + conseption + expression
        # 首先是观察层
        # First is the observation layer
        observation_idx = 0
        for i in range(observation_idx, observation_idx + config.num_observation_layers):
            # 第一个观察层是MLP
            # The first observation layer is an MLP
            if i == observation_idx:
                decoder_layers.append(CheemsObservationLayer(config, num_experts=1, layer_idx=i))
            # 其他观察层是MOE
            # Other observation layers are MOE
            else:
                decoder_layers.append(CheemsObservationLayer(config, num_experts=config.num_experts, layer_idx=i))

        # 然后是思考层
        # Then is the thinking layer
        thinking_idx = observation_idx + config.num_observation_layers
        for i in range(thinking_idx, thinking_idx + config.num_thinking_layers):
            decoder_layers.append(CheemsThinkingLayer(config, num_experts=config.num_experts, layer_idx=i))

        # 然后是构思层
        # Then is the conseption layer
        conseption_idx = thinking_idx + config.num_thinking_layers
        for i in range(conseption_idx, conseption_idx + config.num_conseption_layers):
            # 最后一层是MLP
            # The last layer is an MLP
            if i == conseption_idx + config.num_conseption_layers - 1:
                decoder_layers.append(CheemsConseptionLayer(config, num_experts=1, layer_idx=i))
            # 其他构思层是MOE
            # Other conseption layers are MOE
            else:
                decoder_layers.append(CheemsConseptionLayer(config, num_experts=config.num_experts, layer_idx=i))

        # 最后是表达层
        # Finally is the expression layer
        expression_idx = conseption_idx + config.num_conseption_layers
        for i in range(expression_idx, expression_idx + config.num_expression_layers):
            # 表达层是MLP
            # The expression layer is an MLP
            decoder_layers.append(CheemsExpressionLayer(config, layer_idx=i))
        
        self.layers = nn.ModuleList(decoder_layers)

        self._attn_implementation = config._attn_implementation

        # 最终的LayerNorm
        # Final LayerNorm
        self.final_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], HybridMambaAttentionDynamicCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检索input_ids和inputs_embeds
        # Retrieve input_ids and inputs_embeds
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("你不能同时指定input_ids和inputs_embeds You cannot specify both input_ids and inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True`与梯度检查点不兼容. 设置`use_cache=False`..."

                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            logger.warning_once(
                "OTCE需要一个初始化的`HybridMambaAttentionDynamicCache`来返回一个缓存. 没有提供, 因此不会返回缓存."

                "OTCE requires an initialized `HybridMambaAttentionDynamicCache` to return a cache. None was "
                "provided, so no cache will be returned."
            )
        
        if cache_position is None:
            cache_position = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # 解码器层
        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                if layer_outputs[1] is not None:
                    # 仅附加注意力层的注意力. Mamba层返回`None`作为注意力权重
                    # append attentions only of attention layers. Mamba layers return `None` as the attention weights
                    all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                if layer_outputs[-1] is not None:
                    # 仅附加专家层的路由器对数. 常规MLP层返回`None`作为路由器对数
                    # append router logits only of expert layers. Regular MLP layers return `None` as the router logits
                    all_router_logits += (layer_outputs[-1],)

        hidden_states = self.final_layernorm(hidden_states)

        # 添加来自最后一个解码器层的隐藏状态
        # Add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if past_key_values and not past_key_values.has_previous_state:
            past_key_values.has_previous_state = True

        next_cache = None if not use_cache else past_key_values

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
    

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = cache_position[-1] + 1

        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit 复制到连续内存以进行原地编辑
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            # 注意 causal_mask 中完全掩码行中的所有令牌, 例如在使用左填充时相关的第一行. 这是由F.scaled_dot_product_attention节省内存的注意力路径所需的.
            # 详细信息: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


# [CLS]分类器
# [CLS] classifier
class DefaultSequenceClassifier(nn.Module):
    def __init__(self, config: CheemsOTCEConfig):
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False, 
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.classifier(hidden_states[:, -1])
        return logits


# 最大池化分类器
# Max pooling classifier
class MaxPoolSequenceClassifier(nn.Module):
    def __init__(self, config: CheemsOTCEConfig):
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False, 
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.classifier(torch.max(hidden_states, dim=1).values)
        return logits


# 平均池化分类器
# Mean pooling classifier
class MeanPoolSequenceClassifier(nn.Module):
    def __init__(self, config: CheemsOTCEConfig):
        super().__init__()
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels, 
            bias=False, 
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.classifier(torch.mean(hidden_states, dim=1))
        return logits


class CheemsOTCEForSequenceClassification(CheemsOTCEPreTrainedModel):
    def __init__(self, config: CheemsOTCEConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = CheemsOTCEModel(config)
        self.classifier = DefaultSequenceClassifier(config)
        # 初始化权重并应用最终处理
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            类别标签, 用于计算序列分类(或回归)损失. 索引应在 `[0, ..., config.num_labels - 1]` 范围内. 如果 `config.num_labels == 1`, 则计算回归损失(Mean-Square loss), 如果 `config.num_labels > 1`, 则计算分类损失(Cross-Entropy).

        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0] # [batch_size, seq_len, hidden_size]
  
        pooled_logits = self.classifier(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "如果没有定义填充标记, 无法处理批量大小 > 1."

                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # 如果没有找到pad token, 为了ONNX兼容性, 使用模运算而不是反向索引
                # If pad token is not found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(pooled_logits.device)
            else:
                sequence_lengths = -1

        loss = None
        if labels is not None:
            labels = labels.to(pooled_logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class CheemsOTCEForCausalLM(CheemsOTCEPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: CheemsOTCEConfig):
        super().__init__(config)
        self.config = config
        self.model = CheemsOTCEModel(config)
        self.vocab_size = config.vocab_size

        self.lm_head = nn.Linear(
            config.hidden_size, 
            config.vocab_size, 
            bias=False, 
        )
        
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # 初始化权重并应用最终处理
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        cache_position=None,
        **kwargs,
    ):
        empty_past_kv = past_key_values is None

        # Omit tokens covered by past_key_values
        # 忽略被past_key_values覆盖的标记
        if not empty_past_kv:
            past_length = cache_position[0] if cache_position is not None else attention_mask.shape[1]
            max_cache_length = self.config.sliding_window
            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            # 仅保留未处理的标记:
            # 1 - 如果attention_mask的长度超过了input_ids的长度, 那么我们处于一种设置中, 其中一些输入完全作为缓存的一部分传递(例如, 将input_embeds作为输入)

            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            # 2 - 如果past_length小于input_ids', 那么input_ids保存所有输入标记. 我们可以根据past_length丢弃input_ids.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            # 3 - 否则(past_length >= input_ids.shape[1]), 让我们假设input_ids只有未处理的标记.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            # 如果我们即将超过最大缓存长度, 我们需要裁剪输入注意力掩码.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and past_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]
        else:
            past_key_values = HybridMambaAttentionDynamicCache(
                self.config, input_ids.shape[0], self.dtype, device=self.device
            )

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            # 为批量生成即时创建position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if not empty_past_kv:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        # 如果传递了`inputs_embeds`, 我们只想在第1代步中使用它们
        if inputs_embeds is not None and empty_past_kv:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                "num_logits_to_keep": self.config.num_logits_to_keep,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridMambaAttentionDynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: Optional[Union[int, None]] = None,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                用于计算掩码语言建模损失的标签. 索引应在 `[0, ..., config.vocab_size]` 范围内. 设置为 `-100` 的索引被忽略(掩码), 仅为标签为 `[0, ..., config.vocab_size]` 的标记计算损失.

            num_logits_to_keep (`int` or `None`, `optional`):
                计算最后 `num_logits_to_keep` 个标记的对数. 如果为 `None`, 则计算所有 `input_ids` 的对数. 仅对生成的最后一个标记的对数进行计算, 并且仅为该标记节省内存, 对于长序列来说这是非常重要的.


        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int` or `None`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `None`, calculate logits for all
                `input_ids`. Only last token logits are needed for generation, and calculating them only for that token
                can save memory, which becomes pretty significant for long sequences.
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 解码器输出由(dec_features, layer_state, dec_hidden, dec_attn)组成
        # Decoder output consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if num_logits_to_keep is None:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -num_logits_to_keep:, :])
        logits = logits.float()

    
        loss = None
        if labels is not None:
            # Shift 使得 tokens < n 预测 n
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 扁平化 the tokens
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # 开启模型并行
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)


        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # 确保在同一设备上 to ensure on the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
