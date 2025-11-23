"""
Test wildcard selector functionality for target_types and exclude_types
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgerazor.qat.util import QuantConfig


def test_wildcard_all():
    """Test using '.*' to select all module types"""
    print("=" * 80)
    print("Test 1: Wildcard '.*' to select all module types")
    print("=" * 80)
    
    config_yaml = """
method: QAT

select:
  target_types:
    - ".*"
  target_names: []
  exclude_types: []
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_channel_int1_58
  w_scale_factor: 2.0
  w_block_size: -1
  is_w_quantized: false
  activation_function: ""
  a_block_size: -1
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
    
    config = QuantConfig.from_yaml_string(config_yaml)
    
    print(f"Target types selected: {len(config.select.target_types)}")
    for module_type in config.select.target_types:
        print(f"  - {module_type.__name__}")
    
    # Should include: Linear, Embedding, Conv1d, Conv2d, Conv3d, MultiheadAttention,
    # OlmoeAttention, OlmoeSdpaAttention, OlmoeFlashAttention2, Qwen3Attention, Qwen3MoeAttention
    assert len(config.select.target_types) == 11, f"Expected 11 types, got {len(config.select.target_types)}"
    print("✓ Test passed: All module types selected\n")


def test_wildcard_conv():
    """Test using 'conv.*' to select all conv types"""
    print("=" * 80)
    print("Test 2: Wildcard 'conv.*' to select all conv types")
    print("=" * 80)
    
    config_yaml = """
method: QAT

select:
  target_types:
    - "conv.*"
  target_names: []
  exclude_types: []
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_channel_int1_58
  w_scale_factor: 2.0
  w_block_size: -1
  is_w_quantized: false
  activation_function: ""
  a_block_size: -1
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
    
    config = QuantConfig.from_yaml_string(config_yaml)
    
    print(f"Target types selected: {len(config.select.target_types)}")
    for module_type in config.select.target_types:
        print(f"  - {module_type.__name__}")
    
    # Should include: Conv1d, Conv2d, Conv3d
    assert len(config.select.target_types) == 3, f"Expected 3 types, got {len(config.select.target_types)}"
    print("✓ Test passed: All conv types selected\n")


def test_wildcard_mixed():
    """Test mixing exact match and wildcard"""
    print("=" * 80)
    print("Test 3: Mixed exact match and wildcard")
    print("=" * 80)
    
    config_yaml = """
method: QAT

select:
  target_types:
    - "linear"
    - "conv.*"
  target_names: []
  exclude_types: []
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_channel_int1_58
  w_scale_factor: 2.0
  w_block_size: -1
  is_w_quantized: false
  activation_function: ""
  a_block_size: -1
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
    
    config = QuantConfig.from_yaml_string(config_yaml)
    
    print(f"Target types selected: {len(config.select.target_types)}")
    for module_type in config.select.target_types:
        print(f"  - {module_type.__name__}")
    
    # Should include: Linear, Conv1d, Conv2d, Conv3d
    assert len(config.select.target_types) == 4, f"Expected 4 types, got {len(config.select.target_types)}"
    print("✓ Test passed: Mixed selection works\n")


def test_wildcard_exclude():
    """Test using wildcard in exclude_types"""
    print("=" * 80)
    print("Test 4: Wildcard in exclude_types")
    print("=" * 80)
    
    config_yaml = """
method: QAT

select:
  target_types:
    - ".*"
  target_names: []
  exclude_types:
    - "conv.*"
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_channel_int1_58
  w_scale_factor: 2.0
  w_block_size: -1
  is_w_quantized: false
  activation_function: ""
  a_block_size: -1
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
    
    config = QuantConfig.from_yaml_string(config_yaml)
    
    print(f"Target types selected: {len(config.select.target_types)}")
    for module_type in config.select.target_types:
        print(f"  - {module_type.__name__}")
    
    print(f"Exclude types selected: {len(config.select.exclude_types)}")
    for module_type in config.select.exclude_types:
        print(f"  - {module_type.__name__}")
    
    # Target should include all 11 module types, exclude should have 3 conv types
    assert len(config.select.target_types) == 11, f"Expected 11 target types, got {len(config.select.target_types)}"
    assert len(config.select.exclude_types) == 3, f"Expected 3 exclude types, got {len(config.select.exclude_types)}"
    print("✓ Test passed: Wildcard exclude works\n")


def test_wildcard_specific():
    """Test using wildcard with specific patterns"""
    print("=" * 80)
    print("Test 5: Specific wildcard patterns")
    print("=" * 80)
    
    config_yaml = """
method: QAT

select:
  target_types:
    - "conv[12]d"
  target_names: []
  exclude_types: []
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_channel_int1_58
  w_scale_factor: 2.0
  w_block_size: -1
  is_w_quantized: false
  activation_function: ""
  a_block_size: -1
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
    
    config = QuantConfig.from_yaml_string(config_yaml)
    
    print(f"Target types selected: {len(config.select.target_types)}")
    for module_type in config.select.target_types:
        print(f"  - {module_type.__name__}")
    
    # Should include: Conv1d, Conv2d (but not Conv3d)
    assert len(config.select.target_types) == 2, f"Expected 2 types, got {len(config.select.target_types)}"
    print("✓ Test passed: Specific regex pattern works\n")


def test_backward_compatibility():
    """Test that exact matches still work (backward compatibility)"""
    print("=" * 80)
    print("Test 6: Backward compatibility with exact matches")
    print("=" * 80)
    
    config_yaml = """
method: QAT

select:
  target_types:
    - linear
    - conv2d
  target_names: []
  exclude_types: []
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_channel_int1_58
  w_scale_factor: 2.0
  w_block_size: -1
  is_w_quantized: false
  activation_function: ""
  a_block_size: -1
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
    
    config = QuantConfig.from_yaml_string(config_yaml)
    
    print(f"Target types selected: {len(config.select.target_types)}")
    for module_type in config.select.target_types:
        print(f"  - {module_type.__name__}")
    
    # Should include: Linear, Conv2d
    assert len(config.select.target_types) == 2, f"Expected 2 types, got {len(config.select.target_types)}"
    print("✓ Test passed: Backward compatibility maintained\n")


if __name__ == "__main__":
    try:
        test_wildcard_all()
        test_wildcard_conv()
        test_wildcard_mixed()
        test_wildcard_exclude()
        test_wildcard_specific()
        test_backward_compatibility()
        
        print("=" * 80)
        print("✓ All tests passed!")
        print("=" * 80)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
