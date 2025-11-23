"""
Test various ways to load QuantConfig objects.
- Python dict
- YAML file and string
- JSON file and string

This test demonstrates all the supported methods for loading and saving QuantConfig.
"""
import json
import tempfile
from pathlib import Path

from edgerazor.qat.util import QuantConfig


def test_from_dict():
    """Test creating QuantConfig from Python dictionary"""
    print("\n" + "=" * 80)
    print("Test 1: Creating QuantConfig from Python Dictionary")
    print("=" * 80)
    
    # Define configuration as a Python dictionary
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
            "weight_function": "weight_quant_uniform_symmetric_clip_per_channel_int1_58",
            "w_scale_factor": 2.0,
            "w_block_size": -1,
            "is_w_quantized": False,
            "activation_function": "",
            "a_block_size": -1,
            "kv_cache_function": "",
            "kv_block_size": -1
        },
        "training": "all"
    }
    
    # Create QuantConfig from dictionary
    config = QuantConfig(config_dict)
    
    print("\n✓ QuantConfig created successfully from dictionary")
    print(f"\nMethod: {config.method}")
    print(f"Target Types: {config.select.target_types}")
    print(f"Weight Function: {config.function.weight_function}")
    
    # Verify configuration is correctly created
    assert config is not None
    assert config.method == "QAT"
    assert len(config.select.target_types) == 2


def test_from_yaml_file():
    """Test creating QuantConfig from YAML file"""
    print("\n" + "=" * 80)
    print("Test 2: Creating QuantConfig from YAML File")
    print("=" * 80)
    
    # Create a temporary YAML file
    yaml_content = """
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
  weight_function: weight_quant_uniform_symmetric_clip_per_block_int1_58
  w_scale_factor: 2.0
  w_block_size: 128
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
        # Load from YAML file
        config = QuantConfig.from_yaml(yaml_path)
        
        print(f"\n✓ QuantConfig loaded successfully from YAML file: {yaml_path}")
        print(f"\nMethod: {config.method}")
        print(f"Weight Function: {config.function.weight_function}")
        print(f"Block Size: {config.function.w_block_size}")
        
        # Verify configuration is correctly loaded
        assert config is not None
        assert config.method == "QAT"
        assert config.function.w_block_size == 128
    finally:
        # Clean up temporary file
        Path(yaml_path).unlink()


def test_from_yaml_string():
    """Test creating QuantConfig from YAML string"""
    print("\n" + "=" * 80)
    print("Test 3: Creating QuantConfig from YAML String")
    print("=" * 80)
    
    yaml_string = """
method: QAT
select:
  target_types: [linear, conv2d]
  target_names: []
  exclude_types: []
  exclude_names: []
function:
  epsilon: 1e-05
  weight_function: weight_quant_uniform_symmetric_absmax_per_channel_int4
  w_scale_factor: -1
  w_block_size: -1
  is_w_quantized: false
  activation_function: state_quant_uniform_symmetric_absmax_per_token_int8
  a_block_size: -1
  kv_cache_function: ""
  kv_block_size: -1
training: all
"""
    
    # Load from YAML string
    config = QuantConfig.from_yaml_string(yaml_string)
    
    print("\n✓ QuantConfig loaded successfully from YAML string")
    print(f"\nMethod: {config.method}")
    print(f"Weight Function: {config.function.weight_function}")
    print(f"Activation Function: {config.function.activation_function}")
    
    # Verify configuration is correctly loaded
    assert config is not None
    assert config.method == "QAT"
    assert config.function.activation_function is not None


def test_from_json_file():
    """Test creating QuantConfig from JSON file"""
    print("\n" + "=" * 80)
    print("Test 4: Creating QuantConfig from JSON File")
    print("=" * 80)
    
    # Create a temporary JSON file
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(json_content, f, indent=2)
        json_path = f.name
    
    try:
        # Load from JSON file
        config = QuantConfig.from_json(json_path)
        
        print(f"\n✓ QuantConfig loaded successfully from JSON file: {json_path}")
        print(f"\nMethod: {config.method}")
        print(f"Weight Function: {config.function.weight_function}")
        print(f"Block Size: {config.function.w_block_size}")
        
        # Verify configuration is correctly loaded
        assert config is not None
        assert config.method == "QAT"
        assert config.function.w_block_size == 128
    finally:
        # Clean up temporary file
        Path(json_path).unlink()


def test_from_json_string():
    """Test creating QuantConfig from JSON string"""
    print("\n" + "=" * 80)
    print("Test 5: Creating QuantConfig from JSON String")
    print("=" * 80)
    
    json_string = """
{
  "method": "QAT",
  "select": {
    "target_types": ["linear", "conv2d"],
    "target_names": [],
    "exclude_types": [],
    "exclude_names": []
  },
  "function": {
    "epsilon": 1e-05,
    "weight_function": "weight_quant_uniform_symmetric_absmax_per_channel_int4",
    "w_scale_factor": -1,
    "w_block_size": -1,
    "is_w_quantized": false,
    "activation_function": "state_quant_uniform_symmetric_absmax_per_token_int8",
    "a_block_size": -1,
    "kv_cache_function": "",
    "kv_block_size": -1
  },
  "training": "all"
}
"""
    
    # Load from JSON string
    config = QuantConfig.from_json_string(json_string)
    
    print("\n✓ QuantConfig loaded successfully from JSON string")
    print(f"\nMethod: {config.method}")
    print(f"Weight Function: {config.function.weight_function}")
    print(f"Activation Function: {config.function.activation_function}")
    
    # Verify configuration is correctly loaded
    assert config is not None
    assert config.method == "QAT"
    assert config.function.activation_function is not None


def test_save_yaml():
    """Test saving QuantConfig to YAML file"""
    print("\n" + "=" * 80)
    print("Test 6: Saving QuantConfig to YAML File")
    print("=" * 80)
    
    # Create a QuantConfig
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
            "weight_function": "weight_quant_uniform_symmetric_clip_per_channel_int1_58",
            "w_scale_factor": 2.0,
            "w_block_size": -1,
            "is_w_quantized": False,
            "activation_function": "",
            "a_block_size": -1,
            "kv_cache_function": "",
            "kv_block_size": -1
        },
        "training": "all"
    }
    config = QuantConfig(config_dict)
    
    # Save to temporary YAML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml_path = f.name
    
    try:
        config.to_yaml(yaml_path)
        print(f"\n✓ QuantConfig saved successfully to YAML file: {yaml_path}")
        
        # Verify by reading back
        with open(yaml_path, encoding='utf-8') as f:
            content = f.read()
            print("\nSaved YAML content:")
            print(content)
        
        # Load it back to verify
        _ = QuantConfig.from_yaml(yaml_path)
        print("✓ Successfully loaded back the saved YAML file")
        
    finally:
        # Clean up temporary file
        Path(yaml_path).unlink()


def test_save_json():
    """Test saving QuantConfig to JSON file"""
    print("\n" + "=" * 80)
    print("Test 7: Saving QuantConfig to JSON File")
    print("=" * 80)
    
    # Create a QuantConfig
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
            "weight_function": "weight_quant_uniform_symmetric_clip_per_channel_int1_58",
            "w_scale_factor": 2.0,
            "w_block_size": -1,
            "is_w_quantized": False,
            "activation_function": "",
            "a_block_size": -1,
            "kv_cache_function": "",
            "kv_block_size": -1
        },
        "training": "all"
    }
    config = QuantConfig(config_dict)
    
    # Save to temporary JSON file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        config.to_json(json_path)
        print(f"\n✓ QuantConfig saved successfully to JSON file: {json_path}")
        
        # Verify by reading back
        with open(json_path, encoding='utf-8') as f:
            content = f.read()
            print("\nSaved JSON content:")
            print(content)
        
        # Load it back to verify
        _ = QuantConfig.from_json(json_path)
        print("✓ Successfully loaded back the saved JSON file")
        
    finally:
        # Clean up temporary file
        Path(json_path).unlink()


def test_with_overrides():
    """Test QuantConfig with per-layer overrides"""
    print("\n" + "=" * 80)
    print("Test 8: QuantConfig with Per-Layer Overrides")
    print("=" * 80)
    
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
            "weight_function": "weight_quant_uniform_symmetric_clip_per_channel_int1_58",
            "w_scale_factor": 2.0,
            "w_block_size": -1,
            "is_w_quantized": False,
            "activation_function": "",
            "a_block_size": -1,
            "kv_cache_function": "",
            "kv_block_size": -1
        },
        "overrides": [
            {
                "type": "linear",
                "weight_function": "weight_quant_uniform_symmetric_absmax_per_channel_int4",
                "w_scale_factor": -1
            },
            {
                "name": "layer4.*",
                "weight_function": "weight_quant_uniform_symmetric_clip_per_block_int1_58",
                "w_block_size": 128
            }
        ],
        "training": "all"
    }
    
    config = QuantConfig(config_dict)
    
    # Test saving and loading with overrides
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_path = f.name
    
    try:
        config.to_json(json_path)
        # Verify by loading back
        _ = QuantConfig.from_json(json_path)
        print("\n✓ QuantConfig with overrides saved and loaded successfully")
    finally:
        # Clean up temporary file
        Path(json_path).unlink()
    
    print("\n✓ QuantConfig with overrides created successfully")
    print(f"\nGlobal Weight Function: {config.function.weight_function}")
    print(f"Number of overrides: {len(config.overrides)}")
    print(f"\nOverride 1: type={config.overrides[0].module_type}, overrides={config.overrides[0].overrides}")
    print(f"Override 2: name={config.overrides[1].module_name}, overrides={config.overrides[1].overrides}")
    
    # Verify configuration with overrides is correctly created
    assert config is not None
    assert len(config.overrides) == 2
    assert config.overrides[0].module_type == "linear"


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE QUANTCONFIG LOADING TESTS")
    print("=" * 80)
    print("\nThis script demonstrates all supported methods for loading and saving QuantConfig:")
    print("1. From Python dictionary")
    print("2. From YAML file")
    print("3. From YAML string")
    print("4. From JSON file (NEW)")
    print("5. From JSON string (NEW)")
    print("6. Save to YAML file")
    print("7. Save to JSON file (NEW)")
    print("8. With per-layer overrides")
    
    try:
        # Run all tests
        test_from_dict()
        test_from_yaml_file()
        test_from_yaml_string()
        test_from_json_file()
        test_from_json_string()
        test_save_yaml()
        test_save_json()
        test_with_overrides()
        
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

