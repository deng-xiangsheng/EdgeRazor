# ruff: noqa: UP045 N801

from collections.abc import Callable
from typing import Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniConfig,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAttention,
    apply_multimodal_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils.deprecation import deprecate_kwarg

from ..util.quant_config import QuantConfig


class QKVCacheQwen2_5OmniAttention(Qwen2_5OmniAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper with KV Cache quantization"""

    def __init__(
        self,
        config: Qwen2_5OmniConfig,
        layer_idx: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__(config, layer_idx)
        if quant_config is None:
            raise ValueError("quant_config must be provided for QKVCacheQwen2_5OmniAttention")

        # Small value to prevent division by zero
        self.epsilon = quant_config.function.epsilon

        # Quantization configuration
        ## KV Cache (State)
        self.kv_cache_quant_function = quant_config.function.kv_cache_function
        self.kv_block_size = quant_config.function.kv_block_size
        self.kv_mixed_precision_prop = quant_config.function.kv_mixed_precision_prop
        self.kv_kwargs = {'epsilon': self.epsilon}
        if self.kv_block_size > 0:
            self.kv_kwargs['block_size'] = self.kv_block_size
        if self.kv_mixed_precision_prop > 0:
            self.kv_kwargs['mixed_precision_prop'] = self.kv_mixed_precision_prop

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        # --------------------------------------------------------------------------
        # After RoPE | Before KV Cache Storing: Apply KV Cache Quantization
        key_quant = self.kv_cache_quant_function(x=key_states, **self.kv_kwargs)
        value_quant = self.kv_cache_quant_function(x=value_states, **self.kv_kwargs)
        key_states = key_states + (key_quant - key_states).detach()
        value_states = value_states + (value_quant - value_states).detach()
        # --------------------------------------------------------------------------

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
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
            sliding_window=self.sliding_window,
            position_ids=position_ids,  # pass positions for FA2
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


def copy_qwen2_5omniattention_to_qkvcache_qwen2_5omniattention(
    qwen2_5omni_attn: Qwen2_5OmniAttention,
    qkvcache_qwen2_5omniattention_cls: QKVCacheQwen2_5OmniAttention,
    quant_config: QuantConfig,
) -> QKVCacheQwen2_5OmniAttention:
    """Copy Qwen2_5OmniAttention to QKVCacheQwen2_5OmniAttention with quantization config"""
    if quant_config is None:
        raise ValueError("quant_config must be provided for QKVCacheQwen2_5OmniAttention")

    # Create quantized attention with the same config and layer_idx
    qkvcache_qwen2_5omni_attn = qkvcache_qwen2_5omniattention_cls(
        config=qwen2_5omni_attn.config,
        layer_idx=qwen2_5omni_attn.layer_idx,
        quant_config=quant_config,
    )

    # Copy all projection weights
    qkvcache_qwen2_5omni_attn.q_proj.weight.data = qwen2_5omni_attn.q_proj.weight.data.clone()
    qkvcache_qwen2_5omni_attn.k_proj.weight.data = qwen2_5omni_attn.k_proj.weight.data.clone()
    qkvcache_qwen2_5omni_attn.v_proj.weight.data = qwen2_5omni_attn.v_proj.weight.data.clone()
    qkvcache_qwen2_5omni_attn.o_proj.weight.data = qwen2_5omni_attn.o_proj.weight.data.clone()

    # Copy biases if they exist
    if qwen2_5omni_attn.q_proj.bias is not None:
        qkvcache_qwen2_5omni_attn.q_proj.bias.data = qwen2_5omni_attn.q_proj.bias.data.clone()
    if qwen2_5omni_attn.k_proj.bias is not None:
        qkvcache_qwen2_5omni_attn.k_proj.bias.data = qwen2_5omni_attn.k_proj.bias.data.clone()
    if qwen2_5omni_attn.v_proj.bias is not None:
        qkvcache_qwen2_5omni_attn.v_proj.bias.data = qwen2_5omni_attn.v_proj.bias.data.clone()
    if qwen2_5omni_attn.o_proj.bias is not None:
        qkvcache_qwen2_5omni_attn.o_proj.bias.data = qwen2_5omni_attn.o_proj.bias.data.clone()

    # Copy rotary_emb (if needed, the rotary_emb should have been initialized in __init__)
    # No need to copy rotary_emb as it's stateless
    
    # Copy state
    qkvcache_qwen2_5omni_attn.training = qwen2_5omni_attn.training

    return qkvcache_qwen2_5omni_attn
