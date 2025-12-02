"""
LLM KV Cache Quantization Implementation.

Only supply KV Cache (State) quantization during forward process. If attn uses nn.Linear
for projections, those nn.Linear layers can be weight-quantized as well. However, if attn
uses nn.Parameter for QKV projections, we need to replace the attention block with a custom
quantized attention block.

Naming convention: QKVCache + original Attention class name, e.g., QKVCacheOlmoeAttention

Supported Attention Blocks:
- OlmoeAttention
- OlmoeSdpaAttention
- OlmoeFlashAttention2
"""
# ruff: noqa: UP045

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeFlashAttention2,
    OlmoeSdpaAttention,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)
from transformers.utils.deprecation import deprecate_kwarg

from ..util.quant_config import QuantConfig


class QKVCacheOlmoeAttention(OlmoeAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OlmoeConfig,
        layer_idx: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__(config, layer_idx)
        if quant_config is None:
            raise ValueError("quant_config must be provided for QKVCacheOlmoeAttention")
        
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
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # --------------------------------------------------------------------------
        # After RoPE | Before KV Cache Storing: Apply KV Cache Quantization
        key_quant = self.kv_cache_quant_function(x=key_states, **self.kv_kwargs)
        value_quant = self.kv_cache_quant_function(x=value_states, **self.kv_kwargs)
        key_states = key_states + (key_quant - key_states).detach()
        value_states = value_states + (value_quant - value_states).detach()
        # --------------------------------------------------------------------------

        if past_key_values is not None:  # use_cache=True means past_key_values is not None
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # ❗️ This is the operation of storing and updating KV cache with quantization
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class QKVCacheOlmoeFlashAttention2(OlmoeFlashAttention2):
    """
    OLMoE flash attention module with KV Cache quantization. This module inherits from `OlmoeFlashAttention2`
    as the weights of the module stays untouched. The only required change would be on the forward pass where
    it needs to correctly call the public API of flash attention and deal with padding tokens in case the input
    contains any of them, plus apply KV Cache quantization.
    """

    def __init__(
        self,
        config: OlmoeConfig,
        layer_idx: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__(config, layer_idx)
        if quant_config is None:
            raise ValueError("quant_config must be provided for QKVCacheOlmoeFlashAttention2")
        
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
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)
        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --------------------------------------------------------------------------
        # After RoPE | Before KV Cache Storing: Apply KV Cache Quantization
        key_quant = self.kv_cache_quant_function(x=key_states, **self.kv_kwargs)
        value_quant = self.kv_cache_quant_function(x=value_states, **self.kv_kwargs)
        key_states = key_states + (key_quant - key_states).detach()
        value_states = value_states + (value_quant - value_states).detach()
        # --------------------------------------------------------------------------

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (OlmoeRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        device_type = query_states.device.type if query_states.device.type != "mps" else "cpu"
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = (
                    torch.get_autocast_dtype(device_type)
                    if hasattr(torch, "get_autocast_dtype")
                    else torch.get_autocast_gpu_dtype()
                )
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class QKVCacheOlmoeSdpaAttention(OlmoeSdpaAttention):
    """
    OLMoE attention module using torch.nn.functional.scaled_dot_product_attention with KV Cache quantization.
    This module inherits from `OlmoeSdpaAttention` as the weights of the module stays untouched.
    The only changes are on the forward pass to adapt to SDPA API and add KV Cache quantization.
    """

    def __init__(
        self,
        config: OlmoeConfig,
        layer_idx: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__(config, layer_idx)
        if quant_config is None:
            raise ValueError("quant_config must be provided for QKVCacheOlmoeSdpaAttention")
        
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

    # Adapted from OlmoeAttention.forward
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
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "OlmoeModel is using OlmoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        if self.config.clip_qkv is not None:
            query_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            key_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            value_states.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --------------------------------------------------------------------------
        # After RoPE | Before KV Cache Storing: Apply KV Cache Quantization
        key_quant = self.kv_cache_quant_function(x=key_states, **self.kv_kwargs)
        value_quant = self.kv_cache_quant_function(x=value_states, **self.kv_kwargs)
        key_states = key_states + (key_quant - key_states).detach()
        value_states = value_states + (value_quant - value_states).detach()
        # --------------------------------------------------------------------------

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        # if attention_mask is not None and cache_position is not None:
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = causal_mask is None and q_len > 1

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None


def copy_olmoeattention_qkvcache_olmoeattention(
    olmoe_attn: OlmoeAttention | OlmoeSdpaAttention | OlmoeFlashAttention2,
    qkvcache_olmoeattn_cls: QKVCacheOlmoeAttention | QKVCacheOlmoeSdpaAttention | QKVCacheOlmoeFlashAttention2,
    quant_config: QuantConfig = None
) -> QKVCacheOlmoeAttention | QKVCacheOlmoeSdpaAttention | QKVCacheOlmoeFlashAttention2:
    """Copy OlmoeAttention/OlmoeSdpaAttention/OlmoeFlashAttention2 to QKVCacheOlmoeAttention/QKVCacheOlmoeSdpaAttention/QKVCacheOlmoeFlashAttention2"""
    if quant_config is None:
        raise ValueError("quant_config must be provided for QKVCacheOlmoeAttention")
    
    # Create quantized attention with the same config and layer_idx
    qkvcache_olmoe_attn = qkvcache_olmoeattn_cls(
        config=olmoe_attn.config,
        layer_idx=olmoe_attn.layer_idx,
        quant_config=quant_config
    )
    
    # Copy all projection weights
    qkvcache_olmoe_attn.q_proj.weight.data = olmoe_attn.q_proj.weight.data.clone()
    qkvcache_olmoe_attn.k_proj.weight.data = olmoe_attn.k_proj.weight.data.clone()
    qkvcache_olmoe_attn.v_proj.weight.data = olmoe_attn.v_proj.weight.data.clone()
    qkvcache_olmoe_attn.o_proj.weight.data = olmoe_attn.o_proj.weight.data.clone()
    
    # Copy biases if they exist
    if olmoe_attn.q_proj.bias is not None:
        qkvcache_olmoe_attn.q_proj.bias.data = olmoe_attn.q_proj.bias.data.clone()
    if olmoe_attn.k_proj.bias is not None:
        qkvcache_olmoe_attn.k_proj.bias.data = olmoe_attn.k_proj.bias.data.clone()
    if olmoe_attn.v_proj.bias is not None:
        qkvcache_olmoe_attn.v_proj.bias.data = olmoe_attn.v_proj.bias.data.clone()
    if olmoe_attn.o_proj.bias is not None:
        qkvcache_olmoe_attn.o_proj.bias.data = olmoe_attn.o_proj.bias.data.clone()
    
    # Copy normalization layers
    qkvcache_olmoe_attn.q_norm.weight.data = olmoe_attn.q_norm.weight.data.clone()
    qkvcache_olmoe_attn.k_norm.weight.data = olmoe_attn.k_norm.weight.data.clone()
    
    # Copy state
    qkvcache_olmoe_attn.training = olmoe_attn.training
    
    return qkvcache_olmoe_attn
