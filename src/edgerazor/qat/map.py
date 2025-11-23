"""
Mapping of quantization functions and classes.

`str -> function/class`
"""
import torch.nn as nn
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeFlashAttention2,
    OlmoeSdpaAttention,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

# Import directly from the source module to avoid circular import
from .util.quant_function import (
    state_quant_uniform_symmetric_absmax_per_block_int2,
    state_quant_uniform_symmetric_absmax_per_block_int4,
    state_quant_uniform_symmetric_absmax_per_block_int8,
    state_quant_uniform_symmetric_absmax_per_token_int2,
    state_quant_uniform_symmetric_absmax_per_token_int4,
    state_quant_uniform_symmetric_absmax_per_token_int8,
    weight_quant_uniform_asymmetric_max_per_block_int4,
    weight_quant_uniform_asymmetric_max_per_channel_int4,
    weight_quant_uniform_asymmetric_max_per_tensor_int4,
    weight_quant_uniform_symmetric_absmax_per_block_int1_58,
    weight_quant_uniform_symmetric_absmax_per_block_int4,
    weight_quant_uniform_symmetric_absmax_per_channel_int1_58,
    weight_quant_uniform_symmetric_absmax_per_channel_int4,
    weight_quant_uniform_symmetric_absmax_per_tensor_int1_58,
    weight_quant_uniform_symmetric_absmax_per_tensor_int4,
    weight_quant_uniform_symmetric_clip_per_block_int1_58,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse,
    weight_quant_uniform_symmetric_clip_per_channel_int1_58,
    weight_quant_uniform_symmetric_clip_per_tensor_int1_58,
)

quant_function_map = {
    # Weight Quantization Functions - INT1_58 (Ternary)
    "weight_quant_uniform_symmetric_clip_per_tensor_int1_58": weight_quant_uniform_symmetric_clip_per_tensor_int1_58,
    "weight_quant_uniform_symmetric_clip_per_channel_int1_58": weight_quant_uniform_symmetric_clip_per_channel_int1_58,
    "weight_quant_uniform_symmetric_clip_per_block_int1_58": weight_quant_uniform_symmetric_clip_per_block_int1_58,
    "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic": weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic,
    "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static": weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static,
    "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse": weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse,
    "weight_quant_uniform_symmetric_absmax_per_tensor_int1_58": weight_quant_uniform_symmetric_absmax_per_tensor_int1_58,
    "weight_quant_uniform_symmetric_absmax_per_channel_int1_58": weight_quant_uniform_symmetric_absmax_per_channel_int1_58,
    "weight_quant_uniform_symmetric_absmax_per_block_int1_58": weight_quant_uniform_symmetric_absmax_per_block_int1_58,
    # Weight Quantization Functions - INT4
    "weight_quant_uniform_symmetric_absmax_per_tensor_int4": weight_quant_uniform_symmetric_absmax_per_tensor_int4,
    "weight_quant_uniform_symmetric_absmax_per_channel_int4": weight_quant_uniform_symmetric_absmax_per_channel_int4,
    "weight_quant_uniform_symmetric_absmax_per_block_int4": weight_quant_uniform_symmetric_absmax_per_block_int4,
    "weight_quant_uniform_asymmetric_max_per_tensor_int4": weight_quant_uniform_asymmetric_max_per_tensor_int4,
    "weight_quant_uniform_asymmetric_max_per_channel_int4": weight_quant_uniform_asymmetric_max_per_channel_int4,
    "weight_quant_uniform_asymmetric_max_per_block_int4": weight_quant_uniform_asymmetric_max_per_block_int4,
    # State Quantization Functions (Activation & KV Cache) - Per Token
    "state_quant_uniform_symmetric_absmax_per_token_int2": state_quant_uniform_symmetric_absmax_per_token_int2,
    "state_quant_uniform_symmetric_absmax_per_token_int4": state_quant_uniform_symmetric_absmax_per_token_int4,
    "state_quant_uniform_symmetric_absmax_per_token_int8": state_quant_uniform_symmetric_absmax_per_token_int8,
    # State Quantization Functions (Activation & KV Cache) - Per Block
    "state_quant_uniform_symmetric_absmax_per_block_int2": state_quant_uniform_symmetric_absmax_per_block_int2,
    "state_quant_uniform_symmetric_absmax_per_block_int4": state_quant_uniform_symmetric_absmax_per_block_int4,
    "state_quant_uniform_symmetric_absmax_per_block_int8": state_quant_uniform_symmetric_absmax_per_block_int8,
}


modules_map = {
    "linear": nn.Linear,
    "embedding": nn.Embedding,
    "conv1d": nn.Conv1d,
    "conv2d": nn.Conv2d,
    "conv3d": nn.Conv3d,
    "multiheadattention": nn.MultiheadAttention,
    "olmoeattention": OlmoeAttention,
    "olmoesdpaattention": OlmoeSdpaAttention,
    "olmoeflashattention2": OlmoeFlashAttention2,
    "qwen3moeattention": Qwen3MoeAttention,
    "qwen3attention": Qwen3Attention,
}
