"""
Comprehensive test suite for EdgeRazorConfig unified configuration.

This module tests the ability to load both QAT and KD configurations from:
- Single unified YAML/JSON files
- Separate QAT and KD files
- Dictionary objects
- Individual QuantConfig and DistillConfig objects
"""

import json
from pathlib import Path

import pytest
import yaml

from edgerazor import EdgeRazorConfig
from edgerazor.kd.util import DistillConfig
from edgerazor.qat.util import QuantConfig


class TestUnifiedConfigLoading:
    """Test loading unified configurations from various sources."""

    def test_load_from_unified_yaml(self, tmp_path):
        """Test loading both QAT and KD from a single unified YAML file."""
        yaml_content = """
# QAT Configuration
method: QAT
qat_configuration:
  select:
    target_types: [linear]
    target_names: []
    exclude_types: []
    exclude_names: []
  
  function:
    epsilon: 0.00001
    weight_function: weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static
    w_scale_factor: 2.0
    w_block_size: 128
    is_w_quantized: false
    activation_function: state_quant_uniform_symmetric_absmax_per_block_int8
    a_block_size: 128
    kv_cache_function: ""
    kv_block_size: -1
  
  training: all

# KD Configuration
method: KD
kd_configuration:
  loss_task_alpha: 1.0
  
  loss_1:
    loss_type: logits
    loss_function: kldc
    alpha: 0.7
    temperature: 2.0
    use_entropy: true
    padding_id: -100
    is_router_logits: false
    reduction: batch_mean
  
  loss_2:
    loss_type: hidden_states
    loss_function: fd
    alpha: 0.5
    padding_id: -100
    reduction: batch_mean
"""
        yaml_path = tmp_path / "unified_config.yaml"
        yaml_path.write_text(yaml_content)
        
        # Load configuration
        config = EdgeRazorConfig.from_yaml(yaml_path)
        
        # Verify QAT config is loaded
        assert config.qat_config is not None
        assert isinstance(config.qat_config, QuantConfig)
        # target_types gets converted to actual module classes
        assert len(config.qat_config.select.target_types) == 1
        assert config.qat_config.function.epsilon == 0.00001
        
        # Verify KD config is loaded
        assert config.kd_config is not None
        assert isinstance(config.kd_config, DistillConfig)
        assert config.kd_config.loss_task_alpha == 1.0
        assert len(config.kd_config.losses) == 2
        assert "loss_1" in config.kd_config.losses
        assert "loss_2" in config.kd_config.losses
        
        # Verify loss configurations
        loss_1 = config.kd_config.losses["loss_1"]
        assert loss_1.loss_function == "kldc"
        assert loss_1.alpha == 0.7
        assert loss_1.temperature == 2.0
        
        loss_2 = config.kd_config.losses["loss_2"]
        assert loss_2.loss_function == "fd"
        assert loss_2.alpha == 0.5

    def test_load_from_unified_json(self, tmp_path):
        """Test loading both QAT and KD from a single unified JSON file."""
        json_content = {
            "qat_configuration": {
                "select": {
                    "target_types": ["linear"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": []
                },
                "function": {
                    "epsilon": 0.00001,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                    "w_scale_factor": 2.0,
                    "w_block_size": 128,
                    "is_w_quantized": False,
                    "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                    "a_block_size": 128,
                    "kv_cache_function": "",
                    "kv_block_size": -1
                },
                "training": "all"
            },
            "kd_configuration": {
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "kldf",
                    "alpha": 0.5,
                    "temperature": 2.0,
                    "use_entropy": True,
                    "padding_id": -100,
                    "is_router_logits": False,
                    "reduction": "batch_mean"
                }
            }
        }
        
        json_path = tmp_path / "unified_config.json"
        with open(json_path, 'w') as f:
            json.dump(json_content, f)
        
        # Load configuration
        config = EdgeRazorConfig.from_json(json_path)
        
        # Verify both configs are loaded
        assert config.qat_config is not None
        assert config.kd_config is not None
        assert len(config.kd_config.losses) == 1

    def test_load_from_separate_yaml_files(self, tmp_path):
        """Test loading QAT and KD from separate YAML files."""
        qat_yaml = """
method: QAT
select:
  target_types: [linear]
  target_names: []
  exclude_types: []
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static
  w_scale_factor: 2.0
  w_block_size: 128
  is_w_quantized: false
  activation_function: state_quant_uniform_symmetric_absmax_per_block_int8
  a_block_size: 128
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
        
        kd_yaml = """
method: KD
loss_task_alpha: 1.0

loss_1:
  loss_type: logits
  loss_function: kldc
  alpha: 0.7
  temperature: 2.0
  use_entropy: true
  padding_id: -100
  is_router_logits: false
  reduction: batch_mean
"""
        
        qat_path = tmp_path / "qat_config.yaml"
        kd_path = tmp_path / "kd_config.yaml"
        qat_path.write_text(qat_yaml)
        kd_path.write_text(kd_yaml)
        
        # Load from separate files
        config = EdgeRazorConfig.from_yaml(qat_yaml=qat_path, kd_yaml=kd_path)
        
        # Verify both configs are loaded
        assert config.qat_config is not None
        assert config.kd_config is not None
        assert config.kd_config.loss_task_alpha == 1.0

    def test_load_qat_only(self, tmp_path):
        """Test loading only QAT configuration."""
        yaml_content = """
method: QAT
select:
  target_types: [linear]
  target_names: []
  exclude_types: []
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static
  w_scale_factor: 2.0
  w_block_size: 128
  is_w_quantized: false
  activation_function: state_quant_uniform_symmetric_absmax_per_block_int8
  a_block_size: 128
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
        yaml_path = tmp_path / "qat_only.yaml"
        yaml_path.write_text(yaml_content)
        
        config = EdgeRazorConfig.from_yaml(qat_yaml=yaml_path)
        
        # Verify QAT is loaded, KD is None
        assert config.qat_config is not None
        assert config.kd_config is None

    def test_load_kd_only(self, tmp_path):
        """Test loading only KD configuration."""
        yaml_content = """
method: KD
loss_task_alpha: 1.0

loss_1:
  loss_type: logits
  loss_function: kldf
  alpha: 0.5
  temperature: 2.0
  use_entropy: true
  padding_id: -100
  is_router_logits: false
  reduction: batch_mean
"""
        yaml_path = tmp_path / "kd_only.yaml"
        yaml_path.write_text(yaml_content)
        
        config = EdgeRazorConfig.from_yaml(kd_yaml=yaml_path)
        
        # Verify KD is loaded, QAT is None
        assert config.qat_config is None
        assert config.kd_config is not None

    def test_load_from_dict(self):
        """Test loading from dictionary."""
        config_dict = {
            "qat_configuration": {
                "select": {
                    "target_types": ["linear"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": []
                },
                "function": {
                    "epsilon": 0.00001,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                    "w_scale_factor": 2.0,
                    "w_block_size": 128,
                    "is_w_quantized": False,
                    "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                    "a_block_size": 128,
                    "kv_cache_function": "",
                    "kv_block_size": -1
                },
                "training": "all"
            },
            "kd_configuration": {
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "kldf",
                    "alpha": 0.5,
                    "temperature": 2.0,
                    "use_entropy": True,
                    "padding_id": -100,
                    "is_router_logits": False,
                    "reduction": "batch_mean"
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        
        # Verify both configs are loaded
        assert config.qat_config is not None
        assert config.kd_config is not None

    def test_load_from_config_objects(self):
        """Test creating EdgeRazorConfig from existing QuantConfig and DistillConfig objects."""
        # Create QAT config
        qat_dict = {
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": []
            },
            "function": {
                "epsilon": 0.00001,
                "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                "w_scale_factor": 2.0,
                "w_block_size": 128,
                "is_w_quantized": False,
                "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                "a_block_size": 128,
                "kv_cache_function": "",
                "kv_block_size": -1
            },
            "training": "all"
        }
        qat_config = QuantConfig(qat_dict)
        
        # Create KD config
        kd_dict = {
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "kldf",
                "alpha": 0.5,
                "temperature": 2.0,
                "use_entropy": True,
                "padding_id": -100,
                "is_router_logits": False,
                "reduction": "batch_mean"
            }
        }
        kd_config = DistillConfig.from_dict(kd_dict)
        
        # Create unified config
        config = EdgeRazorConfig(qat_config=qat_config, kd_config=kd_config)
        
        # Verify
        assert config.qat_config is qat_config
        assert config.kd_config is kd_config


class TestUnifiedConfigSaving:
    """Test saving unified configurations to files."""

    def test_save_to_yaml(self, tmp_path):
        """Test saving unified config to YAML file."""
        # Create config
        config_dict = {
            "qat_configuration": {
                "select": {
                    "target_types": ["linear"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": []
                },
                "function": {
                    "epsilon": 0.00001,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                    "w_scale_factor": 2.0,
                    "w_block_size": 128,
                    "is_w_quantized": False,
                    "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                    "a_block_size": 128,
                    "kv_cache_function": "",
                    "kv_block_size": -1
                },
                "training": "all"
            },
            "kd_configuration": {
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "kldf",
                    "alpha": 0.5,
                    "temperature": 2.0,
                    "use_entropy": True,
                    "padding_id": -100,
                    "is_router_logits": False,
                    "reduction": "batch_mean"
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        
        # Save to YAML
        yaml_path = tmp_path / "saved_config.yaml"
        config.to_yaml(yaml_path)
        
        # Verify file exists
        assert yaml_path.exists()
        
        # Load and verify
        loaded_config = EdgeRazorConfig.from_yaml(yaml_path)
        assert loaded_config.qat_config is not None
        assert loaded_config.kd_config is not None
        assert loaded_config.kd_config.loss_task_alpha == 1.0

    def test_save_to_json(self, tmp_path):
        """Test saving unified config to JSON file."""
        config_dict = {
            "qat_configuration": {
                "select": {
                    "target_types": ["linear"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": []
                },
                "function": {
                    "epsilon": 0.00001,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                    "w_scale_factor": 2.0,
                    "w_block_size": 128,
                    "is_w_quantized": False,
                    "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                    "a_block_size": 128,
                    "kv_cache_function": "",
                    "kv_block_size": -1
                },
                "training": "all"
            },
            "kd_configuration": {
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "kldf",
                    "alpha": 0.5,
                    "temperature": 2.0,
                    "use_entropy": True,
                    "padding_id": -100,
                    "is_router_logits": False,
                    "reduction": "batch_mean"
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        
        # Save to JSON
        json_path = tmp_path / "saved_config.json"
        config.to_json(json_path)
        
        # Verify file exists
        assert json_path.exists()
        
        # Load and verify
        loaded_config = EdgeRazorConfig.from_json(json_path)
        assert loaded_config.qat_config is not None
        assert loaded_config.kd_config is not None

    def test_save_qat_only(self, tmp_path):
        """Test saving when only QAT config is present."""
        qat_dict = {
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": []
            },
            "function": {
                "epsilon": 0.00001,
                "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                "w_scale_factor": 2.0,
                "w_block_size": 128,
                "is_w_quantized": False,
                "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                "a_block_size": 128,
                "kv_cache_function": "",
                "kv_block_size": -1
            },
            "training": "all"
        }
        qat_config = QuantConfig(qat_dict)
        config = EdgeRazorConfig(qat_config=qat_config, kd_config=None)
        
        yaml_path = tmp_path / "qat_only.yaml"
        config.to_yaml(yaml_path)
        
        # Load and verify
        loaded_config = EdgeRazorConfig.from_yaml(yaml_path)
        assert loaded_config.qat_config is not None
        assert loaded_config.kd_config is None

    def test_save_kd_only(self, tmp_path):
        """Test saving when only KD config is present."""
        kd_dict = {
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "kldf",
                "alpha": 0.5,
                "temperature": 2.0,
                "use_entropy": True,
                "padding_id": -100,
                "is_router_logits": False,
                "reduction": "batch_mean"
            }
        }
        kd_config = DistillConfig.from_dict(kd_dict)
        config = EdgeRazorConfig(qat_config=None, kd_config=kd_config)
        
        yaml_path = tmp_path / "kd_only.yaml"
        config.to_yaml(yaml_path)
        
        # Load and verify
        loaded_config = EdgeRazorConfig.from_yaml(yaml_path)
        assert loaded_config.qat_config is None
        assert loaded_config.kd_config is not None


class TestUnifiedConfigConversion:
    """Test conversion between different configuration formats."""

    def test_to_dict(self):
        """Test converting EdgeRazorConfig to dictionary."""
        config_dict = {
            "qat_configuration": {
                "select": {
                    "target_types": ["linear"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": []
                },
                "function": {
                    "epsilon": 0.00001,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                    "w_scale_factor": 2.0,
                    "w_block_size": 128,
                    "is_w_quantized": False,
                    "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                    "a_block_size": 128,
                    "kv_cache_function": "",
                    "kv_block_size": -1
                },
                "training": "all"
            },
            "kd_configuration": {
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "kldf",
                    "alpha": 0.5,
                    "temperature": 2.0,
                    "use_entropy": True,
                    "padding_id": -100,
                    "is_router_logits": False,
                    "reduction": "batch_mean"
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        result_dict = config.to_dict()
        
        # Verify structure
        assert "qat_configuration" in result_dict
        assert "kd_configuration" in result_dict

    def test_roundtrip_yaml(self, tmp_path):
        """Test loading and saving maintains configuration integrity."""
        original_dict = {
            "qat_configuration": {
                "select": {
                    "target_types": ["linear"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": []
                },
                "function": {
                    "epsilon": 0.00001,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                    "w_scale_factor": 2.0,
                    "w_block_size": 128,
                    "is_w_quantized": False,
                    "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                    "a_block_size": 128,
                    "kv_cache_function": "",
                    "kv_block_size": -1
                },
                "training": "all"
            },
            "kd_configuration": {
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "kldf",
                    "alpha": 0.5,
                    "temperature": 2.0,
                    "use_entropy": True,
                    "padding_id": -100,
                    "is_router_logits": False,
                    "reduction": "batch_mean"
                }
            }
        }
        
        # Load, save, and reload
        config1 = EdgeRazorConfig.from_dict(original_dict)
        yaml_path = tmp_path / "roundtrip.yaml"
        config1.to_yaml(yaml_path)
        config2 = EdgeRazorConfig.from_yaml(yaml_path)
        
        # Verify key parameters are preserved
        assert config2.qat_config.function.epsilon == 0.00001
        assert config2.kd_config.loss_task_alpha == 1.0
        assert len(config2.kd_config.losses) == 1


class TestUnifiedConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_config(self):
        """Test creating config with neither QAT nor KD raises error."""
        with pytest.raises(ValueError, match="At least one of qat_config or kd_config must be provided"):
            EdgeRazorConfig()

    def test_invalid_yaml_path(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            EdgeRazorConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_invalid_json_path(self):
        """Test loading from non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            EdgeRazorConfig.from_json("/nonexistent/path/config.json")

    def test_malformed_yaml(self, tmp_path):
        """Test loading from malformed YAML."""
        yaml_path = tmp_path / "malformed.yaml"
        yaml_path.write_text("this is not: valid: yaml: content:")
        
        with pytest.raises(yaml.YAMLError):
            EdgeRazorConfig.from_yaml(yaml_path)

    def test_malformed_json(self, tmp_path):
        """Test loading from malformed JSON."""
        json_path = tmp_path / "malformed.json"
        json_path.write_text("{ this is not valid json }")
        
        with pytest.raises(json.JSONDecodeError):
            EdgeRazorConfig.from_json(json_path)

    def test_str_representation(self):
        """Test string representation of config."""
        qat_dict = {
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": []
            },
            "function": {
                "epsilon": 0.00001,
                "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                "w_scale_factor": 2.0,
                "w_block_size": 128,
                "is_w_quantized": False,
                "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                "a_block_size": 128,
                "kv_cache_function": "",
                "kv_block_size": -1
            },
            "training": "all"
        }
        qat_config = QuantConfig(qat_dict)
        config = EdgeRazorConfig(qat_config=qat_config)
        str_repr = str(config)
        assert "EdgeRazorConfig" in str_repr

    def test_repr_representation(self):
        """Test repr representation of config."""
        qat_dict = {
            "select": {
                "target_types": ["linear"],
                "target_names": [],
                "exclude_types": [],
                "exclude_names": []
            },
            "function": {
                "epsilon": 0.00001,
                "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                "w_scale_factor": 2.0,
                "w_block_size": 128,
                "is_w_quantized": False,
                "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                "a_block_size": 128,
                "kv_cache_function": "",
                "kv_block_size": -1
            },
            "training": "all"
        }
        qat_config = QuantConfig(qat_dict)
        config = EdgeRazorConfig(qat_config=qat_config)
        repr_str = repr(config)
        assert "EdgeRazorConfig" in repr_str


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_load_example_config(self):
        """Test loading the actual example config file."""
        config_path = Path("example/configs/qad/qat_w1.58mp4_a8_kd_kldc_fd.yaml")
        
        if config_path.exists():
            config = EdgeRazorConfig.from_yaml(config_path)
            
            # Verify QAT configuration
            assert config.qat_config is not None
            # target_types gets converted to actual module classes
            assert len(config.qat_config.select.target_types) == 1
            
            # Verify KD configuration
            assert config.kd_config is not None
            assert config.kd_config.loss_task_alpha == 1.0
            assert len(config.kd_config.losses) == 2
            
            # Verify specific loss functions
            loss_1 = config.kd_config.losses["loss_1"]
            assert loss_1.loss_function == "compute_kld_confidence"
            assert loss_1.alpha == 0.7
            
            loss_2 = config.kd_config.losses["loss_2"]
            assert loss_2.loss_function == "compute_fd"
            assert loss_2.alpha == 0.5

    def test_modify_and_save(self, tmp_path):
        """Test loading, modifying, and saving config."""
        config_dict = {
            "qat_configuration": {
                "select": {
                    "target_types": ["linear"],
                    "target_names": [],
                    "exclude_types": [],
                    "exclude_names": []
                },
                "function": {
                    "epsilon": 0.00001,
                    "weight_function": "weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static",
                    "w_scale_factor": 2.0,
                    "w_block_size": 128,
                    "is_w_quantized": False,
                    "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
                    "a_block_size": 128,
                    "kv_cache_function": "",
                    "kv_block_size": -1
                },
                "training": "all"
            },
            "kd_configuration": {
                "loss_task_alpha": 1.0,
                "loss_1": {
                    "loss_type": "logits",
                    "loss_function": "kldf",
                    "alpha": 0.5,
                    "temperature": 2.0,
                    "use_entropy": True,
                    "padding_id": -100,
                    "is_router_logits": False,
                    "reduction": "batch_mean"
                }
            }
        }
        
        # Load config
        config = EdgeRazorConfig.from_dict(config_dict)
        
        # Modify KD config
        config.kd_config.loss_task_alpha = 1.5
        config.kd_config.losses["loss_1"].alpha = 0.8
        
        # Save
        yaml_path = tmp_path / "modified_config.yaml"
        config.to_yaml(yaml_path)
        
        # Reload and verify modifications
        loaded_config = EdgeRazorConfig.from_yaml(yaml_path)
        assert loaded_config.kd_config.loss_task_alpha == 1.5
        assert loaded_config.kd_config.losses["loss_1"].alpha == 0.8

    def test_progressive_loading(self, tmp_path):
        """Test loading QAT first, then adding KD configuration."""
        # First load QAT only
        qat_yaml = """
method: QAT
select:
  target_types: [linear]
  target_names: []
  exclude_types: []
  exclude_names: []

function:
  epsilon: 0.00001
  weight_function: weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static
  w_scale_factor: 2.0
  w_block_size: 128
  is_w_quantized: false
  activation_function: state_quant_uniform_symmetric_absmax_per_block_int8
  a_block_size: 128
  kv_cache_function: ""
  kv_block_size: -1

training: all
"""
        qat_path = tmp_path / "qat.yaml"
        qat_path.write_text(qat_yaml)
        
        config = EdgeRazorConfig.from_yaml(qat_yaml=qat_path)
        assert config.qat_config is not None
        assert config.kd_config is None
        
        # Add KD configuration
        kd_dict = {
            "loss_task_alpha": 1.0,
            "loss_1": {
                "loss_type": "logits",
                "loss_function": "kldf",
                "alpha": 0.5,
                "temperature": 2.0,
                "use_entropy": True,
                "padding_id": -100,
                "is_router_logits": False,
                "reduction": "batch_mean"
            }
        }
        config.kd_config = DistillConfig.from_dict(kd_dict)
        
        # Save combined config
        combined_path = tmp_path / "combined.yaml"
        config.to_yaml(combined_path)
        
        # Reload and verify
        combined_config = EdgeRazorConfig.from_yaml(combined_path)
        assert combined_config.qat_config is not None
        assert combined_config.kd_config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

