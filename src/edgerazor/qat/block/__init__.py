"""
Implementation of Quantization Blocks/Modules/Components
"""
# ruff: noqa: F401 I001

# Weight and Activation (State) Quantized Modules
from .qattn import QMultiheadAttention, copy_multiheadattention_to_qmultiheadattention

# KV Cache (State) Quantized Modules
from .qkv_cache_olmoe import (
    QKVCacheOlmoeAttention,
    QKVCacheOlmoeFlashAttention2,
    QKVCacheOlmoeSdpaAttention,
    copy_olmoeattention_qkvcache_olmoeattention,
)
from .qkv_cache_qwen2_5omni import (
    QKVCacheQwen2_5OmniAttention,
    copy_qwen2_5omniattention_to_qkvcache_qwen2_5omniattention,
)
from .qkv_cache_qwen3 import (
    QKVCacheQwen3Attention,
    copy_qwen3attention_to_qkvcache_qwen3attention,
)
from .qkv_cache_qwen3moe import (
    QKVCacheQwen3MoeAttention,
    copy_qwen3moeattention_to_qkvcache_qwen3moeattention,
)
from .qkv_cache_llama import (
    QKVCacheLlamaAttention,
    copy_llamaattention_to_qkvcache_llamaattention,
)
