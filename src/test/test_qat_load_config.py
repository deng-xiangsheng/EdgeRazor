"""
Test QAT initialization with different configuration input methods.

This test demonstrates all supported ways to initialize QAT:
1. From YAML file
2. From JSON file
3. From Python dictionary
4. From QuantConfig object
"""
import tempfile
from pathlib import Path

from edgerazor.qat import QAT
from edgerazor.qat.util import QuantConfig


def test_qat_from_yaml():
    """Test QAT initialization from YAML file"""
    print("\n" + "=" * 80)
    print("Test 1: QAT from YAML File")
    print("=" * 80)
    
    # Create a temporary YAML file
    yaml_content = """
method: QAT
select:
  target_types: [linear, conv2d]
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        yaml_path = f.name
    
    try:
        # Initialize QAT from YAML file
        qat = QAT(yaml_path)
        
        print("\n✓ QAT initialized successfully from YAML file")
        print(f"  Config method: {qat.config.method}")
        print(f"  Weight function: {qat.config.function.weight_function}")
        
        # Verify QAT is correctly initialized
        assert qat is not None
        assert qat.config.method == "QAT"
    finally:
        # Clean up
        Path(yaml_path).unlink()


def test_qat_from_json():
    """Test QAT initialization from JSON file"""
    print("\n" + "=" * 80)
    print("Test 2: QAT from JSON File")
    print("=" * 80)
    
    # Create a temporary JSON file
    import json
    
    json_content = {
        "method": "QAT",
        "select": {
            "target_types": ["linear", "conv2d"],
            "target_names": [],
            "exclude_types": [],
            "exclude_names": []
        },
        "function": {
            "epsilon": 1e-5,
            "weight_function": "weight_quant_uniform_symmetric_absmax_per_channel_int4",
            "w_scale_factor": -1,
            "w_block_size": -1,
            "is_w_quantized": False,
            "activation_function": "",
            "a_block_size": -1,
            "kv_cache_function": "",
            "kv_block_size": -1
        },
        "training": "all"
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(json_content, f, indent=2)
        json_path = f.name
    
    try:
        # Initialize QAT from JSON file
        qat = QAT(json_path)
        
        print("\n✓ QAT initialized successfully from JSON file")
        print(f"  Config method: {qat.config.method}")
        print(f"  Weight function: {qat.config.function.weight_function}")
        
        # Verify QAT is correctly initialized
        assert qat is not None
        assert qat.config.method == "QAT"
    finally:
        # Clean up
        Path(json_path).unlink()


def test_qat_from_dict():
    """Test QAT initialization from Python dictionary"""
    print("\n" + "=" * 80)
    print("Test 3: QAT from Python Dictionary")
    print("=" * 80)
    
    config_dict = {
        "method": "QAT",
        "select": {
            "target_types": ["linear"],
            "target_names": [],
            "exclude_types": [],
            "exclude_names": []
        },
        "function": {
            "epsilon": 1e-5,
            "weight_function": "weight_quant_uniform_symmetric_clip_per_block_int1_58",
            "w_scale_factor": 2.0,
            "w_block_size": 128,
            "is_w_quantized": False,
            "activation_function": "",
            "a_block_size": -1,
            "kv_cache_function": "",
            "kv_block_size": -1
        },
        "training": "all"
    }
    
    # Initialize QAT from dictionary
    qat = QAT(config_dict)
    
    print("\n✓ QAT initialized successfully from Python dictionary")
    print(f"  Config method: {qat.config.method}")
    print(f"  Weight function: {qat.config.function.weight_function}")
    print(f"  Block size: {qat.config.function.w_block_size}")
    
    # Verify QAT is correctly initialized
    assert qat is not None
    assert qat.config.method == "QAT"
    assert qat.config.function.w_block_size == 128


def test_qat_from_quantconfig():
    """Test QAT initialization from QuantConfig object"""
    print("\n" + "=" * 80)
    print("Test 4: QAT from QuantConfig Object")
    print("=" * 80)
    
    # Create QuantConfig first
    config_dict = {
        "method": "QAT",
        "select": {
            "target_types": ["linear", "conv2d"],
            "target_names": [],
            "exclude_types": [],
            "exclude_names": []
        },
        "function": {
            "epsilon": 1e-5,
            "weight_function": "weight_quant_uniform_symmetric_absmax_per_channel_int4",
            "w_scale_factor": -1,
            "w_block_size": -1,
            "is_w_quantized": False,
            "activation_function": "state_quant_uniform_symmetric_absmax_per_token_int8",
            "a_block_size": -1,
            "kv_cache_function": "",
            "kv_block_size": -1
        },
        "training": "all"
    }
    
    config = QuantConfig(config_dict)
    
    # Initialize QAT from QuantConfig object
    qat = QAT(config)
    
    print("\n✓ QAT initialized successfully from QuantConfig object")
    print(f"  Config method: {qat.config.method}")
    print(f"  Weight function: {qat.config.function.weight_function}")
    print(f"  Activation function: {qat.config.function.activation_function}")
    
    # Verify QAT is correctly initialized
    assert qat is not None
    assert qat.config.method == "QAT"
    assert qat.config.function.activation_function is not None


def test_qat_path_object():
    """Test QAT initialization with Path object"""
    print("\n" + "=" * 80)
    print("Test 5: QAT from Path Object")
    print("=" * 80)
    
    # Create a temporary YAML file
    yaml_content = """
method: QAT
select:
  target_types: [linear]
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        f.write(yaml_content)
        yaml_path = Path(f.name)  # Use Path object
    
    try:
        # Initialize QAT with Path object
        qat = QAT(yaml_path)
        
        print("\n✓ QAT initialized successfully from Path object")
        print(f"  Config method: {qat.config.method}")
        print(f"  Weight function: {qat.config.function.weight_function}")
        
        # Verify QAT is correctly initialized
        assert qat is not None
        assert qat.config.method == "QAT"
    finally:
        # Clean up
        yaml_path.unlink()


def test_error_handling():
    """Test error handling for invalid inputs"""
    print("\n" + "=" * 80)
    print("Test 6: Error Handling")
    print("=" * 80)
    
    # Test 1: Invalid file extension
    print("\n[Test 6.1] Invalid file extension:")
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            txt_path = f.name
        _ = QAT(txt_path)
        print("  ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"  ✓ Correctly raised ValueError: {e}")
    finally:
        Path(txt_path).unlink()
    
    # Test 2: Invalid type
    print("\n[Test 6.2] Invalid configuration type:")
    try:
        _ = QAT(12345)  # Invalid type
        print("  ✗ Should have raised TypeError")
    except TypeError as e:
        print(f"  ✓ Correctly raised TypeError: {e}")
    
    # Test 3: Non-existent file
    print("\n[Test 6.3] Non-existent file:")
    try:
        _ = QAT("non_existent_config.yaml")
        print("  ✗ Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        print(f"  ✓ Correctly raised FileNotFoundError: {e}")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("QAT CONFIGURATION LOADING TESTS")
    print("=" * 80)
    print("\nThis script demonstrates all supported methods for initializing QAT:")
    print("1. From YAML file (.yaml/.yml)")
    print("2. From JSON file (.json)")
    print("3. From Python dictionary")
    print("4. From QuantConfig object")
    print("5. From Path object")
    print("6. Error handling for invalid inputs")
    
    try:
        # Run all tests
        test_qat_from_yaml()
        test_qat_from_json()
        test_qat_from_dict()
        test_qat_from_quantconfig()
        test_qat_path_object()
        test_error_handling()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"TEST FAILED: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
