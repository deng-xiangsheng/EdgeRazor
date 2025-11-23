"""
Test for QMultiheadAttention module to ensure it works consistently with QLinear/QConv2d.
"""
import pytest
import torch
import torch.nn as nn

from edgerazor.qat.block.qattn import (
    QMultiheadAttention,
    copy_multiheadattention_to_qmultiheadattention,
)
from edgerazor.qat.util.quant_config import QuantConfig


@pytest.fixture
def simple_quant_config():
    """Create a simple quantization config for testing"""
    config_dict = {
        "method": "QAT",
        "function": {
            "epsilon": 1e-5,
            "weight_function": "weight_quant_uniform_symmetric_clip_per_channel_int1_58",
            "w_scale_factor": 2.0,
            "w_block_size": -1,
            "is_w_quantized": False,
            "activation_function": None,
            "a_block_size": -1,
            "kv_cache_function": None,
            "kv_block_size": -1
        }
    }
    return QuantConfig(config_dict)


@pytest.fixture
def quant_config_with_activation():
    """Create a quantization config with activation quantization"""
    config_dict = {
        "method": "QAT",
        "function": {
            "epsilon": 1e-5,
            "weight_function": "weight_quant_uniform_symmetric_clip_per_channel_int1_58",
            "w_scale_factor": 2.0,
            "w_block_size": -1,
            "is_w_quantized": False,
            "activation_function": "state_quant_uniform_symmetric_absmax_per_token_int8",
            "a_block_size": -1,
            "kv_cache_function": None,
            "kv_block_size": -1
        }
    }
    return QuantConfig(config_dict)


def test_qmultiheadattention_initialization(simple_quant_config):
    """Test that QMultiheadAttention can be initialized with QuantConfig"""
    print("\n" + "="*80)
    print("Test 1: QMultiheadAttention Initialization")
    print("="*80)
    
    embed_dim = 128
    num_heads = 8
    
    qmha = QMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,
        quant_config=simple_quant_config
    )
    
    assert qmha.embed_dim == embed_dim
    assert qmha.num_heads == num_heads
    assert qmha.epsilon == 1e-5
    assert qmha.w_scale_factor == 2.0
    assert not qmha.is_w_quantized
    
    print(f"✓ QMultiheadAttention initialized successfully")
    print(f"  embed_dim: {embed_dim}, num_heads: {num_heads}")
    print(f"  w_scale_factor: {qmha.w_scale_factor}")


def test_qmultiheadattention_forward_training(simple_quant_config):
    """Test forward pass in training mode"""
    print("\n" + "="*80)
    print("Test 2: QMultiheadAttention Forward Pass (Training Mode)")
    print("="*80)
    
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    qmha = QMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,
        quant_config=simple_quant_config
    )
    qmha.train()
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass with average_attn_weights=False to get per-head attention
    output, attn_weights = qmha(x, x, x, average_attn_weights=False)
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    print(f"✓ Forward pass in training mode successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")


def test_qmultiheadattention_forward_eval(simple_quant_config):
    """Test forward pass in evaluation mode"""
    print("\n" + "="*80)
    print("Test 3: QMultiheadAttention Forward Pass (Evaluation Mode)")
    print("="*80)
    
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    qmha = QMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,
        quant_config=simple_quant_config
    )
    qmha.eval()
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass with average_attn_weights=False to get per-head attention
    with torch.no_grad():
        output, attn_weights = qmha(x, x, x, average_attn_weights=False)
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    print(f"✓ Forward pass in evaluation mode successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")


def test_qmultiheadattention_with_activation_quant(quant_config_with_activation):
    """Test forward pass with activation quantization"""
    print("\n" + "="*80)
    print("Test 4: QMultiheadAttention with Activation Quantization")
    print("="*80)
    
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    qmha = QMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,
        quant_config=quant_config_with_activation
    )
    qmha.train()
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass with average_attn_weights=False to get per-head attention
    output, attn_weights = qmha(x, x, x, average_attn_weights=False)
    
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    print(f"✓ Forward pass with activation quantization successful")
    print(f"  Activation function: {qmha.a_quant_function}")


def test_copy_multiheadattention_to_qmultiheadattention(simple_quant_config):
    """Test copying weights from standard MHA to QMHA"""
    print("\n" + "="*80)
    print("Test 5: Copy MultiheadAttention to QMultiheadAttention")
    print("="*80)
    
    embed_dim = 64
    num_heads = 4
    
    # Create standard MHA
    mha = nn.MultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True
    )
    
    # Copy to QMHA
    qmha = copy_multiheadattention_to_qmultiheadattention(
        mha, quant_config=simple_quant_config
    )
    
    # Check that weights are copied correctly
    if mha._qkv_same_embed_dim:
        assert torch.allclose(qmha.in_proj_weight, mha.in_proj_weight)
    else:
        assert torch.allclose(qmha.q_proj_weight, mha.q_proj_weight)
        assert torch.allclose(qmha.k_proj_weight, mha.k_proj_weight)
        assert torch.allclose(qmha.v_proj_weight, mha.v_proj_weight)
    
    assert torch.allclose(qmha.out_proj.weight, mha.out_proj.weight)
    
    print(f"✓ Weights copied successfully")
    print(f"  _qkv_same_embed_dim: {mha._qkv_same_embed_dim}")


def test_qmultiheadattention_weight_quantization(simple_quant_config):
    """Test weight quantization methods"""
    print("\n" + "="*80)
    print("Test 6: Weight Quantization Methods")
    print("="*80)
    
    embed_dim = 64
    num_heads = 4
    
    qmha = QMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,
        quant_config=simple_quant_config
    )
    
    # Test in_proj weight quantization
    if qmha._qkv_same_embed_dim:
        w_quant = qmha._in_proj_weight_quant(replace_self=False)
        assert w_quant.shape == qmha.in_proj_weight.shape
        print(f"✓ in_proj weight quantization successful")
        print(f"  Original shape: {qmha.in_proj_weight.shape}")
        print(f"  Quantized shape: {w_quant.shape}")
    else:
        q_w, k_w, v_w = qmha._qkv_proj_weight_quant(replace_self=False)
        assert q_w.shape == qmha.q_proj_weight.shape
        assert k_w.shape == qmha.k_proj_weight.shape
        assert v_w.shape == qmha.v_proj_weight.shape
        print(f"✓ QKV projection weight quantization successful")
    
    # Test out_proj weight quantization
    out_w_quant = qmha._out_proj_weight_quant(replace_self=False)
    assert out_w_quant.shape == qmha.out_proj.weight.shape
    print(f"✓ out_proj weight quantization successful")


def test_qmultiheadattention_backward(simple_quant_config):
    """Test backward pass to ensure gradients flow correctly"""
    print("\n" + "="*80)
    print("Test 7: Backward Pass (Gradient Flow)")
    print("="*80)
    
    embed_dim = 64
    num_heads = 4
    batch_size = 2
    seq_len = 10
    
    qmha = QMultiheadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        batch_first=True,
        quant_config=simple_quant_config
    )
    qmha.train()
    
    # Create dummy input with gradients
    x = torch.randn(batch_size, seq_len, embed_dim, requires_grad=True)
    
    # Forward pass
    output, _ = qmha(x, x, x, need_weights=False)
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    if qmha._qkv_same_embed_dim:
        assert qmha.in_proj_weight.grad is not None
    else:
        assert qmha.q_proj_weight.grad is not None
        assert qmha.k_proj_weight.grad is not None
        assert qmha.v_proj_weight.grad is not None
    assert qmha.out_proj.weight.grad is not None
    
    print(f"✓ Backward pass successful, gradients computed")
    print(f"  Input gradient shape: {x.grad.shape}")
    print(f"  out_proj weight gradient shape: {qmha.out_proj.weight.grad.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
