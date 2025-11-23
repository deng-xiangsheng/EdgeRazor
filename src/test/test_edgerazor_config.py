"""
Test suite for EdgeRazorConfig unified configuration loading.

This module tests the ability to load both QAT and KD configurations from a
single unified configuration file or from separate sources.
"""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from edgerazor import EdgeRazorConfig


class TestEdgeRazorConfigUnified:
    """Test unified configuration loading from single file"""
    
    def test_load_from_dict_both_configs(self):
        """Test loading both QAT and KD configs from dictionary"""
        config_dict = {
            'qat_configuration': {
                'method': 'QAT',
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 1e-5,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'w_scale_factor': -1,
                    'w_block_size': -1,
                    'is_w_quantized': False,
                    'activation_function': None,
                    'a_block_size': -1,
                    'kv_cache_function': None,
                    'kv_block_size': -1
                },
                'training': 'all'
            },
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'kldf',
                    'alpha': 0.5,
                    'temperature': 2.0,
                    'reduction': 'batch_mean'
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        
        assert config.has_qat
        assert config.has_kd
        assert config.qat_config is not None
        assert config.kd_config is not None
        assert config.kd_config.loss_task_alpha == 1.0
        assert 'loss_1' in config.kd_config.losses
    
    def test_load_from_dict_qat_only(self):
        """Test loading QAT config only"""
        config_dict = {
            'qat_configuration': {
                'method': 'QAT',
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 1e-5,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'w_scale_factor': -1,
                    'w_block_size': -1,
                    'is_w_quantized': False,
                    'activation_function': None,
                    'a_block_size': -1,
                    'kv_cache_function': None,
                    'kv_block_size': -1
                },
                'training': 'all'
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        
        assert config.has_qat
        assert not config.has_kd
        assert config.qat_config is not None
        assert config.kd_config is None
    
    def test_load_from_dict_kd_only(self):
        """Test loading KD config only"""
        config_dict = {
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'kldf',
                    'alpha': 0.5,
                    'temperature': 2.0
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        
        assert not config.has_qat
        assert config.has_kd
        assert config.qat_config is None
        assert config.kd_config is not None
    
    def test_load_from_yaml_unified(self):
        """Test loading from unified YAML file"""
        config_dict = {
            'qat_configuration': {
                'method': 'QAT',
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 1e-5,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'w_scale_factor': -1,
                    'w_block_size': -1,
                    'is_w_quantized': False,
                    'activation_function': '',
                    'a_block_size': -1,
                    'kv_cache_function': '',
                    'kv_block_size': -1
                },
                'training': 'all'
            },
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'kldf',
                    'alpha': 0.7,
                    'temperature': 2.0,
                    'reduction': 'batch_mean'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            yaml_path = f.name
        
        try:
            config = EdgeRazorConfig.from_yaml(yaml_path)
            
            assert config.has_qat
            assert config.has_kd
            assert config.qat_config is not None
            assert config.kd_config is not None
            assert config.kd_config.loss_task_alpha == 1.0
            assert 'loss_1' in config.kd_config.losses
            assert config.kd_config.losses['loss_1'].alpha == 0.7
        finally:
            Path(yaml_path).unlink()
    
    def test_load_from_json_unified(self):
        """Test loading from unified JSON file"""
        config_dict = {
            'qat_configuration': {
                'method': 'QAT',
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 1e-5,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'w_scale_factor': -1,
                    'w_block_size': -1,
                    'is_w_quantized': False,
                    'activation_function': None,
                    'a_block_size': -1,
                    'kv_cache_function': None,
                    'kv_block_size': -1
                },
                'training': 'all'
            },
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'kldf',
                    'alpha': 0.8,
                    'temperature': 2.0
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            json_path = f.name
        
        try:
            config = EdgeRazorConfig.from_json(json_path)
            
            assert config.has_qat
            assert config.has_kd
            assert config.qat_config is not None
            assert config.kd_config is not None
            assert config.kd_config.losses['loss_1'].alpha == 0.8
        finally:
            Path(json_path).unlink()


class TestEdgeRazorConfigSeparate:
    """Test loading from separate configuration files"""
    
    def test_load_from_separate_yaml_files(self):
        """Test loading QAT and KD from separate YAML files"""
        qat_dict = {
            'method': 'QAT',
            'select': {
                'target_types': ['linear'],
                'target_names': [],
                'exclude_types': [],
                'exclude_names': []
            },
            'function': {
                'epsilon': 1e-5,
                'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                'w_scale_factor': -1,
                'w_block_size': -1,
                'is_w_quantized': False,
                'activation_function': '',
                'a_block_size': -1,
                'kv_cache_function': '',
                'kv_block_size': -1
            },
            'training': 'all'
        }
        
        kd_dict = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.5,
                'temperature': 2.0
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(qat_dict, f)
            qat_yaml_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(kd_dict, f)
            kd_yaml_path = f.name
        
        try:
            config = EdgeRazorConfig.from_yaml(
                qat_yaml=qat_yaml_path,
                kd_yaml=kd_yaml_path
            )
            
            assert config.has_qat
            assert config.has_kd
            assert config.qat_config is not None
            assert config.kd_config is not None
        finally:
            Path(qat_yaml_path).unlink()
            Path(kd_yaml_path).unlink()
    
    def test_load_qat_only_from_separate_yaml(self):
        """Test loading only QAT from separate YAML file"""
        qat_dict = {
            'method': 'QAT',
            'select': {
                'target_types': ['linear'],
                'target_names': [],
                'exclude_types': [],
                'exclude_names': []
            },
            'function': {
                'epsilon': 1e-5,
                'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                'w_scale_factor': -1,
                'w_block_size': -1,
                'is_w_quantized': False,
                'activation_function': '',
                'a_block_size': -1,
                'kv_cache_function': '',
                'kv_block_size': -1
            },
            'training': 'all'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(qat_dict, f)
            qat_yaml_path = f.name
        
        try:
            config = EdgeRazorConfig.from_yaml(qat_yaml=qat_yaml_path)
            
            assert config.has_qat
            assert not config.has_kd
            assert config.qat_config is not None
            assert config.kd_config is None
        finally:
            Path(qat_yaml_path).unlink()


class TestEdgeRazorConfigConversion:
    """Test configuration conversion methods"""
    
    def test_to_dict(self):
        """Test converting EdgeRazorConfig to dictionary"""
        config_dict = {
            'qat_configuration': {
                'method': 'QAT',
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 1e-5,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'w_scale_factor': -1,
                    'w_block_size': -1,
                    'is_w_quantized': False,
                    'activation_function': None,
                    'a_block_size': -1,
                    'kv_cache_function': None,
                    'kv_block_size': -1
                },
                'training': 'all'
            },
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'kldf',
                    'alpha': 0.5,
                    'temperature': 2.0
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        result_dict = config.to_dict()
        
        assert 'qat_configuration' in result_dict
        assert 'kd_configuration' in result_dict
    
    def test_to_yaml(self):
        """Test saving EdgeRazorConfig to YAML file"""
        config_dict = {
            'qat_configuration': {
                'method': 'QAT',
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 1e-5,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'w_scale_factor': -1,
                    'w_block_size': -1,
                    'is_w_quantized': False,
                    'activation_function': None,
                    'a_block_size': -1,
                    'kv_cache_function': None,
                    'kv_block_size': -1
                },
                'training': 'all'
            },
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'kldf',
                    'alpha': 0.5,
                    'temperature': 2.0
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            config.to_yaml(yaml_path)
            
            # Load back and verify
            loaded_config = EdgeRazorConfig.from_yaml(yaml_path)
            assert loaded_config.has_qat
            assert loaded_config.has_kd
        finally:
            Path(yaml_path).unlink()


class TestEdgeRazorConfigErrors:
    """Test error handling in EdgeRazorConfig"""
    
    def test_error_both_configs_missing(self):
        """Test error when both QAT and KD configs are missing"""
        with pytest.raises(ValueError, match="At least one of qat_config or kd_config"):
            EdgeRazorConfig(qat_config=None, kd_config=None)
    
    def test_error_invalid_yaml_path(self):
        """Test error when YAML file doesn't exist"""
        with pytest.raises(FileNotFoundError):
            EdgeRazorConfig.from_yaml("nonexistent_file.yaml")
    
    def test_error_no_path_provided(self):
        """Test error when no path is provided to from_yaml"""
        with pytest.raises(ValueError, match="Must provide either"):
            EdgeRazorConfig.from_yaml()


class TestEdgeRazorConfigProperties:
    """Test EdgeRazorConfig properties"""
    
    def test_repr_both_enabled(self):
        """Test string representation with both QAT and KD enabled"""
        config_dict = {
            'qat_configuration': {
                'method': 'QAT',
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 1e-5,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'w_scale_factor': -1,
                    'w_block_size': -1,
                    'is_w_quantized': False,
                    'activation_function': None,
                    'a_block_size': -1,
                    'kv_cache_function': None,
                    'kv_block_size': -1
                },
                'training': 'all'
            },
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'kldf',
                    'alpha': 0.5
                }
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        repr_str = repr(config)
        
        assert 'QAT=enabled' in repr_str
        assert 'KD=enabled' in repr_str
    
    def test_repr_qat_only(self):
        """Test string representation with only QAT enabled"""
        config_dict = {
            'qat_configuration': {
                'method': 'QAT',
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 1e-5,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'w_scale_factor': -1,
                    'w_block_size': -1,
                    'is_w_quantized': False,
                    'activation_function': None,
                    'a_block_size': -1,
                    'kv_cache_function': None,
                    'kv_block_size': -1
                },
                'training': 'all'
            }
        }
        
        config = EdgeRazorConfig.from_dict(config_dict)
        repr_str = repr(config)
        
        assert 'QAT=enabled' in repr_str
        assert 'KD=disabled' in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
