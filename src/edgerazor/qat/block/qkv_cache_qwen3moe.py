# ruff: noqa: UP045

from collections.abc import Callable
from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

from ..util.quant_config import QuantConfig


class QKVCacheQwen3MoeAttention(Qwen3MoeAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_idx: int,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__(config, layer_idx)
        if quant_config is None:
            raise ValueError("quant_config must be provided for QKVCacheQwen3MoeAttention")
        
        # Small value to prevent division by zero
        self.epsilon = quant_config.function.epsilon
        
        # Quantization configuration
        ## KV Cache (State)
        self.kv_cache_quant_function = quant_config.function.kv_cache_function
        self.kv_block_size = quant_config.function.kv_block_size

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --------------------------------------------------------------------------
        # After RoPE | Before KV Cache Storing: Apply KV Cache Quantization
        if self.kv_block_size > 0:
            key_quant = self.kv_cache_quant_function(x=key_states, epsilon=self.epsilon, block_size=self.kv_block_size)
            value_quant = self.kv_cache_quant_function(x=value_states, epsilon=self.epsilon, block_size=self.kv_block_size)
        else:
            key_quant = self.kv_cache_quant_function(x=key_states, epsilon=self.epsilon)
            value_quant = self.kv_cache_quant_function(x=value_states, epsilon=self.epsilon)
        key_states = key_states + (key_quant - key_states).detach()
        value_states = value_states + (value_quant - value_states).detach()
        # --------------------------------------------------------------------------

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def copy_qwen3moeattention_to_qkvcache_qwen3moeattention(
    qwen3moe_attn: Qwen3MoeAttention,
    qkvcache_qwen3moeattention_cls: QKVCacheQwen3MoeAttention,
    quant_config: QuantConfig,
) -> QKVCacheQwen3MoeAttention:
    """Copy Qwen3MoeAttention to QKVCacheQwen3MoeAttention with quantization config"""
    if quant_config is None:
        raise ValueError("quant_config must be provided for QKVCacheQwen3MoeAttention")

    # Create quantized attention with the same config and layer_idx
    qkvcache_qwen3moe_attn = qkvcache_qwen3moeattention_cls(
        config=qwen3moe_attn.config,
        layer_idx=qwen3moe_attn.layer_idx,
        quant_config=quant_config,
    )

    # Copy all projection weights
    qkvcache_qwen3moe_attn.q_proj.weight.data = qwen3moe_attn.q_proj.weight.data.clone()
    qkvcache_qwen3moe_attn.k_proj.weight.data = qwen3moe_attn.k_proj.weight.data.clone()
    qkvcache_qwen3moe_attn.v_proj.weight.data = qwen3moe_attn.v_proj.weight.data.clone()
    qkvcache_qwen3moe_attn.o_proj.weight.data = qwen3moe_attn.o_proj.weight.data.clone()

    # Copy biases if they exist
    if qwen3moe_attn.q_proj.bias is not None:
        qkvcache_qwen3moe_attn.q_proj.bias.data = qwen3moe_attn.q_proj.bias.data.clone()
    if qwen3moe_attn.k_proj.bias is not None:
        qkvcache_qwen3moe_attn.k_proj.bias.data = qwen3moe_attn.k_proj.bias.data.clone()
    if qwen3moe_attn.v_proj.bias is not None:
        qkvcache_qwen3moe_attn.v_proj.bias.data = qwen3moe_attn.v_proj.bias.data.clone()
    if qwen3moe_attn.o_proj.bias is not None:
        qkvcache_qwen3moe_attn.o_proj.bias.data = qwen3moe_attn.o_proj.bias.data.clone()

    # Copy normalization layers
    qkvcache_qwen3moe_attn.q_norm.weight.data = qwen3moe_attn.q_norm.weight.data.clone()
    qkvcache_qwen3moe_attn.k_norm.weight.data = qwen3moe_attn.k_norm.weight.data.clone()

    return qkvcache_qwen3moe_attn
