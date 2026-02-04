"""
Mapping of quantization functions and classes.

`str -> function/class`
"""

from collections import OrderedDict

import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeFlashAttention2,
    OlmoeSdpaAttention,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

# Import directly from the source module to avoid circular import
from .util.quant_function import (
    state_quant_uniform_symmetric_absmax_per_block_int2,
    state_quant_uniform_symmetric_absmax_per_block_int4,
    state_quant_uniform_symmetric_absmax_per_block_int4_nested,  # deprecated
    state_quant_uniform_symmetric_absmax_per_block_int8,
    state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic,  # deprecated
    state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic_nested,  # deprecated
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
    weight_quant_uniform_symmetric_clip_per_block_int1_58_nested,  # deprecated
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_column_wise,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_nested,  # deprecated
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse,
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
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_column_wise,
    weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse,
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
    "qwen2_5omniattention": Qwen2_5OmniAttention,
    "llamaattention": LlamaAttention,
}


def create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.01,
    with_activation_kv=False,
    a_block_size=256,
    kv_block_size=128,
):
    """
    Create w1_58 quantization config.

    Args:
        mp_prop: mixed precision proportion (e.g. 0.01 or 0.05)
        with_activation_kv: whether to include activation and kv_cache quantization
    """
    target_types = ["linear", "embedding"]
    if with_activation_kv:
        target_types.append("qwen3attention")
        target_types.append("qwen2_5omniattention")
        target_types.append("llamaattention")
    
    config = OrderedDict(
        [
            ("method", "QAT"),
            (
                "select",
                OrderedDict(
                    [
                        ("target_types", target_types),
                        ("target_names", []),
                        ("exclude_types", []),
                        ("exclude_names", []),
                    ]
                ),
            ),
            (
                "function",
                OrderedDict(
                    [
                        ("epsilon", 1e-05),
                        (
                            "weight_function",
                            w_func,
                        ),
                        ("w_scale_factor", 2.0),
                        ("w_block_size", 256),
                        ("w_mixed_precision_prop", mp_prop),
                        ("is_w_quantized", True),
                        ("activation_function", ""),
                        ("a_block_size", -1),
                        ("a_mixed_precision_prop", -1.0),
                        ("kv_cache_function", ""),
                        ("kv_block_size", -1),
                        ("kv_mixed_precision_prop", -1.0),
                    ]
                ),
            ),
            ("training", "all"),
        ]
    )

    if with_activation_kv:
        config["function"]["activation_function"] = (
            "state_quant_uniform_symmetric_absmax_per_block_int8"
        )
        config["function"]["a_block_size"] = a_block_size
        config["function"]["kv_cache_function"] = (
            "state_quant_uniform_symmetric_absmax_per_block_int8"
        )
        config["function"]["kv_block_size"] = kv_block_size

    return config


def create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=False,
    w_block_size=256,
    a_block_size=256,
    kv_block_size=128,
):
    """
    Create w1_58 quantization config, using int4 quantization for embedding and lm_head.

    Args:
        mp_prop: mixed precision proportion (e.g. 0.125, 0.25, or 0.50)
        with_activation_kv: whether to include activation and kv_cache quantization
    """
    target_types = ["linear", "embedding"]
    if with_activation_kv:
        target_types.append("qwen3attention")
        target_types.append("qwen2_5omniattention")
        target_types.append("llamaattention")

    config = OrderedDict(
        [
            ("method", "QAT"),
            (
                "select",
                OrderedDict(
                    [
                        ("target_types", target_types),
                        ("target_names", []),
                        ("exclude_types", []),
                        ("exclude_names", []),
                    ]
                ),
            ),
            (
                "function",
                OrderedDict(
                    [
                        ("epsilon", 1e-05),
                        (
                            "weight_function",
                            w_func,
                        ),
                        ("w_scale_factor", 2.0),
                        ("w_block_size", w_block_size),
                        ("w_mixed_precision_prop", mp_prop),
                        ("is_w_quantized", True),
                        ("activation_function", ""),
                        ("a_block_size", -1),
                        ("a_mixed_precision_prop", -1.0),
                        ("kv_cache_function", ""),
                        ("kv_block_size", -1),
                        ("kv_mixed_precision_prop", -1.0),
                    ]
                ),
            ),
            (
                "overrides",
                [
                    {
                        "name": ".*embed_tokens",
                        "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                        "w_scale_factor": -1,
                    },
                    {
                        "name": ".*lm_head",
                        "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
                        "w_scale_factor": -1,
                    },
                ],
            ),
            ("training", "all"),
        ]
    )

    if with_activation_kv:
        config["function"]["activation_function"] = (
            "state_quant_uniform_symmetric_absmax_per_block_int8"
        )
        config["function"]["a_block_size"] = a_block_size
        config["function"]["kv_cache_function"] = (
            "state_quant_uniform_symmetric_absmax_per_block_int8"
        )
        config["function"]["kv_block_size"] = kv_block_size

    return config


w4a16kv16 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_scale_factor", 2.0),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    ("activation_function", ""),
                    ("a_block_size", -1),
                    ("a_mixed_precision_prop", -1.0),
                    ("kv_cache_function", ""),
                    ("kv_block_size", -1),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w4a8kv8 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "qwen3attention", "qwen2_5omniattention", "llamaattention"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 256),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 128),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w4a8kv8_bs32 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "qwen3attention", "qwen2_5omniattention", "llamaattention"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_block_size", 32),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 32),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 32),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w4a8kv8_bs256 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "qwen3attention", "qwen2_5omniattention", "llamaattention"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 256),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 256),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w4a8kv8_omni = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "qwen2_5omniattention"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", ["thinker.audio_tower.*", "talker.*", "token2wav.*"]),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_block_size", 32),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 32),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 32),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w4a8kv8_mobilellm = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "qwen3attention", "qwen2_5omniattention", "llamaattention"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_absmax_per_block_int4",
                    ),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", -1.0),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 256),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 256),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w1_58a16kv16 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                    ),
                    ("w_scale_factor", 2.0),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", 0.05),
                    ("is_w_quantized", True),
                    ("activation_function", ""),
                    ("a_block_size", -1),
                    ("a_mixed_precision_prop", -1.0),
                    ("kv_cache_function", ""),
                    ("kv_block_size", -1),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

w1_58a8kv8 = OrderedDict(
    [
        ("method", "QAT"),
        (
            "select",
            OrderedDict(
                [
                    ("target_types", ["linear", "embedding", "qwen3attention", "qwen2_5omniattention", "llamaattention"]),
                    ("target_names", []),
                    ("exclude_types", []),
                    ("exclude_names", []),
                ]
            ),
        ),
        (
            "function",
            OrderedDict(
                [
                    ("epsilon", 1e-05),
                    (
                        "weight_function",
                        "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                    ),
                    ("w_scale_factor", 2.0),
                    ("w_block_size", 256),
                    ("w_mixed_precision_prop", 0.05),
                    ("is_w_quantized", True),
                    (
                        "activation_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("a_block_size", 256),
                    ("a_mixed_precision_prop", -1.0),
                    (
                        "kv_cache_function",
                        "state_quant_uniform_symmetric_absmax_per_block_int8",
                    ),
                    ("kv_block_size", 128),
                    ("kv_mixed_precision_prop", -1.0),
                ]
            ),
        ),
        ("training", "all"),
    ]
)

# Use function to create config - Row-wise Sparse (rws -> default, no suffix)
w1_58_mp1a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.01,
    with_activation_kv=False,
)
w1_58_mp1a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.01,
    with_activation_kv=True,
)
w1_58_mp5a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.05,
    with_activation_kv=False,
)
w1_58_mp5a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.05,
    with_activation_kv=True,
)
w1_58_mp10a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.10,
    with_activation_kv=False,
)
w1_58_mp10a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.10,
    with_activation_kv=True,
)
w1_58_mp15a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.15,
    with_activation_kv=False,
)
w1_58_mp15a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.15,
    with_activation_kv=True,
)
w1_58_mp20a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.20,
    with_activation_kv=False,
)
w1_58_mp20a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.20,
    with_activation_kv=True,
)
w1_58_mp30a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.30,
    with_activation_kv=False,
)
w1_58_mp30a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.30,
    with_activation_kv=True,
)

## Guarantee divisibility ratio
w1_58_mp50a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=False,
)
w1_58_mp50a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=True,
)
w1_58_mp25a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.25,
    with_activation_kv=False,
)
w1_58_mp25a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.25,
    with_activation_kv=True,
)
w1_58_mp12_5a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=False,
)
w1_58_mp12_5a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=True,
)
w1_58_mp6_25a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.0625,
    with_activation_kv=False,
)
w1_58_mp6_25a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.0625,
    with_activation_kv=True,
)
w1_58_mp3_125a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.03125,
    with_activation_kv=False,
)
w1_58_mp3_125a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.03125,
    with_activation_kv=True,
)
w1_58_mp1_5625a16kv16 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.015625,
    with_activation_kv=False,
)
w1_58_mp1_5625a8kv8 = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.015625,
    with_activation_kv=True,
)

# Use function to create config - Column-wise Dense (cwd suffix)
w1_58_mp1a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.01,
    with_activation_kv=False,
)
w1_58_mp1a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.01,
    with_activation_kv=True,
)
w1_58_mp5a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.05,
    with_activation_kv=False,
)
w1_58_mp5a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.05,
    with_activation_kv=True,
)
w1_58_mp10a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.10,
    with_activation_kv=False,
)
w1_58_mp10a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.10,
    with_activation_kv=True,
)
w1_58_mp15a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.15,
    with_activation_kv=False,
)
w1_58_mp15a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.15,
    with_activation_kv=True,
)
w1_58_mp20a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.20,
    with_activation_kv=False,
)
w1_58_mp20a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.20,
    with_activation_kv=True,
)
w1_58_mp30a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.30,
    with_activation_kv=False,
)
w1_58_mp30a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.30,
    with_activation_kv=True,
)

## Guarantee divisibility ratio
w1_58_mp50a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.50,
    with_activation_kv=False,
)
w1_58_mp50a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.50,
    with_activation_kv=True,
)
w1_58_mp25a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.25,
    with_activation_kv=False,
)
w1_58_mp25a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.25,
    with_activation_kv=True,
)
w1_58_mp12_5a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.125,
    with_activation_kv=False,
)
w1_58_mp12_5a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.125,
    with_activation_kv=True,
)
w1_58_mp6_25a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.0625,
    with_activation_kv=False,
)
w1_58_mp6_25a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.0625,
    with_activation_kv=True,
)
w1_58_mp3_125a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.03125,
    with_activation_kv=False,
)
w1_58_mp3_125a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.03125,
    with_activation_kv=True,
)
w1_58_mp1_5625a16kv16_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.015625,
    with_activation_kv=False,
)
w1_58_mp1_5625a8kv8_cwd = create_w1_58_config(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
    mp_prop=0.015625,
    with_activation_kv=True,
)

## Standard quantization config
w1_58a16kv16_embint4 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.00,
    with_activation_kv=False,
)
w1_58a8kv8_embint4 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.00,
    with_activation_kv=True,
)
w1_58a8kv8_embint4_bs32 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.00,
    with_activation_kv=True,
    w_block_size=32,
    a_block_size=32,
    kv_block_size=32,
)
w1_58a8kv8_embint4_bs256 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.00,
    with_activation_kv=True,
    w_block_size=256,
    a_block_size=256,
    kv_block_size=256,
)
w1_58a8kv8_embint4_mobilellm = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.00,
    with_activation_kv=True,
    w_block_size=64,
    a_block_size=64,
    kv_block_size=64,
)

w1_88a16kv16_embint4 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=False,
)
w1_88a8kv8_embint4 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=True,
)
w1_88a8kv8_embint4_bs32 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=True,
    w_block_size=32,
    a_block_size=32,
    kv_block_size=32,
)
w1_88a8kv8_embint4_bs64 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=True,
    w_block_size=64,
    a_block_size=64,
    kv_block_size=64,
)
w1_88a8kv8_embint4_mobilellm = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.125,
    with_activation_kv=True,
    w_block_size=64,
    a_block_size=64,
    kv_block_size=64,
)

w2_79a16kv16_embint4 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=False,
)
w2_79a8kv8_embint4 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=True,
)
w2_79a8kv8_embint4_bs32 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=True,
    w_block_size=32,
    a_block_size=32,
    kv_block_size=32,
)
w2_79a8kv8_embint4_bs64 = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=True,
    w_block_size=64,
    a_block_size=64,
    kv_block_size=64,
)
w2_79a8kv8_embint4_mobilellm = create_w1_58_config_embint4(
    w_func="weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse",
    mp_prop=0.50,
    with_activation_kv=True,
    w_block_size=64,
    a_block_size=64,
    kv_block_size=64,
)

# Map quant_mode string to imported config dict
quant_config_map = {
    "w4a16kv16": w4a16kv16,
    "w4a8kv8": w4a8kv8,
    "w4a8kv8_bs32": w4a8kv8_bs32,
    "w4a8kv8_bs256": w4a8kv8_bs256,
    "w4a8kv8_omni": w4a8kv8_omni,
    "w1_58a16kv16": w1_58a16kv16,
    "w1_58a8kv8": w1_58a8kv8,
    # w_func: weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_row_wise_sparse (default)
    "w1_58_mp1a16kv16": w1_58_mp1a16kv16,
    "w1_58_mp1a8kv8": w1_58_mp1a8kv8,
    "w1_58_mp5a16kv16": w1_58_mp5a16kv16,
    "w1_58_mp5a8kv8": w1_58_mp5a8kv8,
    "w1_58_mp10a16kv16": w1_58_mp10a16kv16,
    "w1_58_mp10a8kv8": w1_58_mp10a8kv8,
    "w1_58_mp15a16kv16": w1_58_mp15a16kv16,
    "w1_58_mp15a8kv8": w1_58_mp15a8kv8,
    "w1_58_mp20a16kv16": w1_58_mp20a16kv16,
    "w1_58_mp20a8kv8": w1_58_mp20a8kv8,
    "w1_58_mp30a16kv16": w1_58_mp30a16kv16,
    "w1_58_mp30a8kv8": w1_58_mp30a8kv8,
    ## Guarantee divisibility ratio
    "w1_58_mp50a16kv16": w1_58_mp50a16kv16,
    "w1_58_mp50a8kv8": w1_58_mp50a8kv8,
    "w1_58_mp25a16kv16": w1_58_mp25a16kv16,
    "w1_58_mp25a8kv8": w1_58_mp25a8kv8,
    "w1_58_mp12_5a16kv16": w1_58_mp12_5a16kv16,
    "w1_58_mp12_5a8kv8": w1_58_mp12_5a8kv8,
    "w1_58_mp6_25a16kv16": w1_58_mp6_25a16kv16,
    "w1_58_mp6_25a8kv8": w1_58_mp6_25a8kv8,
    "w1_58_mp3_125a16kv16": w1_58_mp3_125a16kv16,
    "w1_58_mp3_125a8kv8": w1_58_mp3_125a8kv8,
    "w1_58_mp1_5625a16kv16": w1_58_mp1_5625a16kv16,
    "w1_58_mp1_5625a8kv8": w1_58_mp1_5625a8kv8,
    # w_func: weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static (cwd suffix)
    "w1_58_mp1a16kv16_cwd": w1_58_mp1a16kv16_cwd,
    "w1_58_mp1a8kv8_cwd": w1_58_mp1a8kv8_cwd,
    "w1_58_mp5a16kv16_cwd": w1_58_mp5a16kv16_cwd,
    "w1_58_mp5a8kv8_cwd": w1_58_mp5a8kv8_cwd,
    "w1_58_mp10a16kv16_cwd": w1_58_mp10a16kv16_cwd,
    "w1_58_mp10a8kv8_cwd": w1_58_mp10a8kv8_cwd,
    "w1_58_mp15a16kv16_cwd": w1_58_mp15a16kv16_cwd,
    "w1_58_mp15a8kv8_cwd": w1_58_mp15a8kv8_cwd,
    "w1_58_mp20a16kv16_cwd": w1_58_mp20a16kv16_cwd,
    "w1_58_mp20a8kv8_cwd": w1_58_mp20a8kv8_cwd,
    "w1_58_mp30a16kv16_cwd": w1_58_mp30a16kv16_cwd,
    "w1_58_mp30a8kv8_cwd": w1_58_mp30a8kv8_cwd,
    ## Guarantee divisibility ratio
    "w1_58_mp50a16kv16_cwd": w1_58_mp50a16kv16_cwd,
    "w1_58_mp50a8kv8_cwd": w1_58_mp50a8kv8_cwd,
    "w1_58_mp25a16kv16_cwd": w1_58_mp25a16kv16_cwd,
    "w1_58_mp25a8kv8_cwd": w1_58_mp25a8kv8_cwd,
    "w1_58_mp12_5a16kv16_cwd": w1_58_mp12_5a16kv16_cwd,
    "w1_58_mp12_5a8kv8_cwd": w1_58_mp12_5a8kv8_cwd,
    "w1_58_mp6_25a16kv16_cwd": w1_58_mp6_25a16kv16_cwd,
    "w1_58_mp6_25a8kv8_cwd": w1_58_mp6_25a8kv8_cwd,
    "w1_58_mp3_125a16kv16_cwd": w1_58_mp3_125a16kv16_cwd,
    "w1_58_mp3_125a8kv8_cwd": w1_58_mp3_125a8kv8_cwd,
    "w1_58_mp1_5625a16kv16_cwd": w1_58_mp1_5625a16kv16_cwd,
    "w1_58_mp1_5625a8kv8_cwd": w1_58_mp1_5625a8kv8_cwd,
    # Standard quantization config with embedding int4
    "w1_58a16kv16_embint4": w1_58a16kv16_embint4,
    "w1_58a8kv8_embint4": w1_58a8kv8_embint4,
    "w1_58a8kv8_embint4_bs32": w1_58a8kv8_embint4_bs32,
    "w1_58a8kv8_embint4_bs256": w1_58a8kv8_embint4_bs256,
    "w1_88a16kv16_embint4": w1_88a16kv16_embint4,
    "w1_88a8kv8_embint4": w1_88a8kv8_embint4,
    "w1_88a8kv8_embint4_bs32": w1_88a8kv8_embint4_bs32,
    "w1_88a8kv8_embint4_bs64": w1_88a8kv8_embint4_bs64,
    "w2_79a16kv16_embint4": w2_79a16kv16_embint4,
    "w2_79a8kv8_embint4": w2_79a8kv8_embint4,
    "w2_79a8kv8_embint4_bs32": w2_79a8kv8_embint4_bs32,
    "w2_79a8kv8_embint4_bs64": w2_79a8kv8_embint4_bs64,
    # MobileLLM-specific configs with embedding int4
    "w4a8kv8_mobilellm": w4a8kv8_mobilellm,
    "w2_79a8kv8_embint4_mobilellm": w2_79a8kv8_embint4_mobilellm,
    "w1_88a8kv8_embint4_mobilellm": w1_88a8kv8_embint4_mobilellm,
    "w1_58a8kv8_embint4_mobilellm": w1_58a8kv8_embint4_mobilellm,
}
