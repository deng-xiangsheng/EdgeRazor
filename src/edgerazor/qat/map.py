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
    state_quant_uniform_symmetric_absmax_per_block_int4_nested,
    state_quant_uniform_symmetric_absmax_per_block_int8,
    state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic,
    state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic_nested,
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
    weight_quant_uniform_symmetric_clip_per_block_int1_58_nested,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_nested,
    weight_quant_uniform_symmetric_clip_per_channel_int1_58,
    weight_quant_uniform_symmetric_clip_per_tensor_int1_58,
)

# Collect all quantization functions automatically
_quant_functions = [
    # INT1_58 (Ternary) Weight Quantization - Clip Method
    weight_quant_uniform_symmetric_clip_per_tensor_int1_58,
    weight_quant_uniform_symmetric_clip_per_channel_int1_58,
    weight_quant_uniform_symmetric_clip_per_block_int1_58,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse,
    # INT1_58 (Ternary) Weight Quantization - Absmax Method
    weight_quant_uniform_symmetric_absmax_per_tensor_int1_58,
    weight_quant_uniform_symmetric_absmax_per_channel_int1_58,
    weight_quant_uniform_symmetric_absmax_per_block_int1_58,
    # INT4 Weight Quantization - Symmetric Absmax Method
    weight_quant_uniform_symmetric_absmax_per_tensor_int4,
    weight_quant_uniform_symmetric_absmax_per_channel_int4,
    weight_quant_uniform_symmetric_absmax_per_block_int4,
    # INT4 Weight Quantization - Asymmetric Max Method
    weight_quant_uniform_asymmetric_max_per_tensor_int4,
    weight_quant_uniform_asymmetric_max_per_channel_int4,
    weight_quant_uniform_asymmetric_max_per_block_int4,
    # Stepped Weight Quantization - Symmetric Method
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_nested,
    weight_quant_uniform_symmetric_clip_per_block_int1_58_nested,
    # INT2 State Quantization - Absmax Method
    state_quant_uniform_symmetric_absmax_per_token_int2,
    state_quant_uniform_symmetric_absmax_per_block_int2,
    # INT4 State Quantization - Absmax Method
    state_quant_uniform_symmetric_absmax_per_token_int4,
    state_quant_uniform_symmetric_absmax_per_block_int4,
    state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic,
    # INT8 State Quantization - Absmax Method
    state_quant_uniform_symmetric_absmax_per_token_int8,
    state_quant_uniform_symmetric_absmax_per_block_int8,
    state_quant_uniform_symmetric_absmax_per_block_int4_nested,
    state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic_nested,
]

# Build the map automatically: function_name -> function
quant_function_map = {func.__name__: func for func in _quant_functions}


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
