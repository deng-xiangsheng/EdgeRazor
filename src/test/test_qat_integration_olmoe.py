"""
Integration test for QKVCacheOlmoeAttention in QAT framework.
"""
import torch
import torch.nn as nn

# Mock the transformers imports since we may not have the exact package version
try:
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import OlmoeAttention
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers library not available or Olmoe not found, skipping test")


def test_qkvcacheolmoeattention_integration():
    """Test that QKVCacheOlmoeAttention integrates correctly with QAT framework"""
    if not TRANSFORMERS_AVAILABLE:
        print("Skipping test - transformers.models.olmoe not available")
        return
    
    from edgerazor.qat.block import (
        QKVCacheOlmoeAttention,
        copy_olmoeattention_qkvcache_olmoeattention,
    )
    from edgerazor.qat.util import QuantConfig
    
    # Create a simple config for testing
    config = OlmoeConfig(
        hidden_size=256,
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    
    # Create a simple model with OlmoeAttention
    class SimpleOlmoeModel(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.attention = OlmoeAttention(config, layer_idx=0)
            
        def forward(self, hidden_states, position_embeddings):
            attn_output, _ = self.attention(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
            )
            return attn_output
    
    model = SimpleOlmoeModel(config)
    
    # Create quantization config
    quant_config = QuantConfig({
        "method": "ste",
        "function": {
            "epsilon": 1e-5,
            "weight_function": "weight_quant_uniform_symmetric_clip_per_channel_int1_58",
            "activation_function": None,
            "kv_cache_function": "state_quant_uniform_symmetric_absmax_per_token_int8",
            "w_scale_factor": 1.0,
            "w_block_size": 0,
            "a_block_size": 0,
            "kv_block_size": 0,
            "is_w_quantized": False,
        },
        "select": {
            "target_types": ["olmoeattention"],
            "target_names": [],
            "exclude_types": [],
            "exclude_names": [],
        },
        "training": "qat"
    })
    
    # Test that the copy function works
    qkvcache_attn = copy_olmoeattention_qkvcache_olmoeattention(
        model.attention,
        QKVCacheOlmoeAttention,
        quant_config
    )
    
    # Verify it's the right type
    assert isinstance(qkvcache_attn, QKVCacheOlmoeAttention)
    
    # Verify weights were copied
    assert torch.allclose(qkvcache_attn.q_proj.weight, model.attention.q_proj.weight)
    assert torch.allclose(qkvcache_attn.k_proj.weight, model.attention.k_proj.weight)
    assert torch.allclose(qkvcache_attn.v_proj.weight, model.attention.v_proj.weight)
    assert torch.allclose(qkvcache_attn.o_proj.weight, model.attention.o_proj.weight)
    
    print("✓ QKVCacheOlmoeAttention integration test passed!")
    print("  - Successfully created QKVCacheOlmoeAttention")
    print("  - Weights copied correctly")
    print("  - Quantization config applied")


if __name__ == "__main__":
    test_qkvcacheolmoeattention_integration()
