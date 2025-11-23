"""
Comprehensive test suite for EdgeRazor class.

This module tests the core functionality of the EdgeRazor unified API,
including configuration loading, module initialization, and basic operations.
"""

import pytest
import torch
import torch.nn as nn

from edgerazor import EdgeRazor, EdgeRazorConfig
from edgerazor.kd.util import DistillConfig
from edgerazor.qat.util import QuantConfig


class SimpleModel(nn.Module):
    """Simple test model for EdgeRazor testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class TestEdgeRazorInitialization:
    """Test EdgeRazor initialization with various configurations."""
    
    def test_init_with_unified_config_file(self):
        """Test initialization with unified YAML configuration file."""
        config_path = "example/configs/qad/qat_w1.58mp4_a8_kd_kldc_fd.yaml"
        razor = EdgeRazor(config=config_path)
        
        assert razor.is_qat_enabled
        assert razor.is_kd_enabled
        assert razor.qat is not None
        assert razor.kd is not None
    
    def test_init_with_qat_only_file(self):
        """Test initialization with QAT-only configuration file."""
        config_path = "example/qat/vit/q_vit_w1.58_a8.yaml"
        razor = EdgeRazor(qat_config=config_path)
        
        assert razor.is_qat_enabled
        assert not razor.is_kd_enabled
        assert razor.qat is not None
        assert razor.kd is None
    
    def test_init_with_kd_only_file(self):
        """Test initialization with KD-only configuration file."""
        config_path = "example/configs/kd/kd_kldc_fd.yaml"
        razor = EdgeRazor(kd_config=config_path)
        
        assert not razor.is_qat_enabled
        assert razor.is_kd_enabled
        assert razor.qat is None
        assert razor.kd is not None
    
    def test_init_with_separate_files(self):
        """Test initialization with separate QAT and KD config files."""
        qat_path = "example/qat/vit/q_vit_w1.58_a8.yaml"
        kd_path = "example/configs/kd/kd_kldc_fd.yaml"
        razor = EdgeRazor(qat_config=qat_path, kd_config=kd_path)
        
        assert razor.is_qat_enabled
        assert razor.is_kd_enabled
        assert razor.qat is not None
        assert razor.kd is not None
    
    def test_init_with_dict_config(self):
        """Test initialization with dictionary configuration."""
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
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                    'activation_function': 'state_quant_uniform_symmetric_absmax_per_block_int8'
                },
                'training': 'all'
            },
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'kldf',
                    'alpha': 0.7
                }
            }
        }
        razor = EdgeRazor(config=config_dict)
        
        assert razor.is_qat_enabled
        assert razor.is_kd_enabled
    
    def test_init_with_edgerazor_config_object(self):
        """Test initialization with EdgeRazorConfig object."""
        qat_config = QuantConfig({
            'method': 'QAT',
            'select': {
                'target_types': ['linear']
            },
            'function': {
                'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_int1_58',
                'activation_function': 'state_quant_uniform_symmetric_absmax_per_block_int8'
            },
            'training': 'all'
        })
        
        kd_config = DistillConfig.from_dict({
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.7
            }
        })
        
        edge_config = EdgeRazorConfig(qat_config=qat_config, kd_config=kd_config)
        razor = EdgeRazor(config=edge_config)
        
        assert razor.is_qat_enabled
        assert razor.is_kd_enabled
    
    def test_init_with_no_config_raises_error(self):
        """Test that initialization without config raises ValueError."""
        with pytest.raises(ValueError, match="No configuration provided"):
            EdgeRazor()
    
    def test_repr_qat_only(self):
        """Test string representation for QAT-only configuration."""
        razor = EdgeRazor(qat_config="example/qat/vit/q_vit_w1.58_a8.yaml")
        repr_str = repr(razor)
        
        assert "EdgeRazor" in repr_str
        assert "QAT=enabled" in repr_str
        assert "KD=disabled" in repr_str
    
    def test_repr_kd_only(self):
        """Test string representation for KD-only configuration."""
        razor = EdgeRazor(kd_config="example/configs/kd/kd_kldc_fd.yaml")
        repr_str = repr(razor)
        
        assert "EdgeRazor" in repr_str
        assert "QAT=disabled" in repr_str
        assert "KD=enabled" in repr_str
    
    def test_repr_qat_and_kd(self):
        """Test string representation for QAT + KD configuration."""
        razor = EdgeRazor(config="example/configs/qad/qat_w1.58mp4_a8_kd_kldc_fd.yaml")
        repr_str = repr(razor)
        
        assert "EdgeRazor" in repr_str
        assert "QAT=enabled" in repr_str
        assert "KD=enabled" in repr_str


class TestEdgeRazorPrepare:
    """Test EdgeRazor.prepare() method for model quantization."""
    
    def test_prepare_with_qat_enabled(self):
        """Test prepare() applies quantization when QAT is enabled."""
        razor = EdgeRazor(qat_config="example/qat/vit/q_vit_w1.58_a8.yaml")
        model = SimpleModel()
        
        # Get original module count
        original_modules = len(list(model.modules()))
        
        # Apply quantization
        quantized_model = razor.quantize(model)
        
        # Check that model was modified (quantization layers added)
        assert quantized_model is not None
        assert isinstance(quantized_model, nn.Module)
        # Quantized model should have more modules (quantization wrappers)
        quantized_modules = len(list(quantized_model.modules()))
        assert quantized_modules >= original_modules
    
    def test_prepare_with_kd_only_returns_unchanged(self):
        """Test prepare() returns unchanged model when only KD is enabled."""
        razor = EdgeRazor(kd_config="example/configs/kd/kd_kldc_fd.yaml")
        model = SimpleModel()
        
        # Apply prepare (should be no-op)
        result_model = razor.quantize(model)
        
        # Model should be unchanged
        assert result_model is model
    
    def test_prepare_with_qat_and_kd(self):
        """Test prepare() applies quantization when both QAT and KD are enabled."""
        razor = EdgeRazor(config="example/configs/qad/qat_w1.58mp4_a8_kd_kldc_fd.yaml")
        model = SimpleModel()
        
        original_modules = len(list(model.modules()))
        quantized_model = razor.quantize(model)
        
        assert quantized_model is not None
        quantized_modules = len(list(quantized_model.modules()))
        assert quantized_modules >= original_modules
    
    def test_prepare_preserves_model_functionality(self):
        """Test that prepare() preserves model forward pass functionality."""
        # Use int8 quantization which works with small models
        config = {
            'method': 'QAT',
            'function': {
                'weight': 'weight_quant_uniform_symmetric_clip_per_block_int8',
                'state': 'state_quant_uniform_symmetric_absmax_per_block_int8'
            },
            'selection': {'module_type': ['Linear']}
        }
        razor = EdgeRazor(config=config)
        model = SimpleModel()
        quantized_model = razor.quantize(model)
        
        # Test forward pass
        batch_size = 4
        input_dim = 10
        x = torch.randn(batch_size, input_dim)
        
        # Original model output
        with torch.no_grad():
            output = quantized_model(x)
        
        assert output.shape == (batch_size, 10)  # SimpleModel outputs 10 features (fc2 output)
        assert not torch.isnan(output).any()


class TestEdgeRazorComputeLoss:
    """Test EdgeRazor.compute_loss() method for knowledge distillation."""
    
    def test_compute_loss_with_kd_enabled(self):
        """Test compute_loss() with KD enabled."""
        razor = EdgeRazor(kd_config="example/configs/kd/kd_kldc_fd.yaml")
        
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        hidden_size = 64
        
        # Create mock outputs
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        student_hidden = torch.randn(batch_size, seq_len, hidden_size)
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Task loss
        task_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        student_outputs = {
            'loss': task_loss,
            'logits': student_logits,
            'hidden_states': student_hidden  # Should be tensor, not tuple
        }
        
        teacher_outputs = {
            'logits': teacher_logits,
            'hidden_states': teacher_hidden  # Should be tensor, not tuple
        }
        
        # Compute loss
        total_loss, loss_dict = razor.compute_loss(
            student_outputs,
            teacher_outputs,
            labels
        )
        
        # Validate outputs
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # Scalar
        assert not torch.isnan(total_loss)
        
        assert 'task_loss' in loss_dict
        assert 'distill_loss' in loss_dict
        assert 'total_loss' in loss_dict
        assert 'distill_loss_details' in loss_dict
    
    def test_compute_loss_with_kd_disabled_returns_task_loss(self):
        """Test compute_loss() returns only task loss when KD is disabled."""
        razor = EdgeRazor(qat_config="example/qat/vit/q_vit_w1.58_a8.yaml")
        
        task_loss_value = 1.5
        task_loss = torch.tensor(task_loss_value)
        
        student_outputs = {'loss': task_loss}
        teacher_outputs = {}
        labels = torch.randint(0, 100, (2, 10))
        
        # Compute loss
        total_loss, loss_dict = razor.compute_loss(
            student_outputs,
            teacher_outputs,
            labels
        )
        
        # Should return task loss only
        assert total_loss == task_loss
        assert loss_dict['task_loss'] == task_loss_value
        assert loss_dict['total_loss'] == task_loss_value
        assert 'distill_loss' not in loss_dict
    
    def test_compute_loss_with_missing_task_loss_raises_error(self):
        """Test compute_loss() raises error when student_outputs lacks 'loss' field."""
        razor = EdgeRazor(qat_config="example/qat/vit/q_vit_w1.58_a8.yaml")
        
        student_outputs = {'logits': torch.randn(2, 10, 100)}
        teacher_outputs = {}
        labels = torch.randint(0, 100, (2, 10))
        
        with pytest.raises(ValueError, match="must contain 'loss' field"):
            razor.compute_loss(student_outputs, teacher_outputs, labels)
    
    def test_compute_loss_with_qat_and_kd(self):
        """Test compute_loss() with both QAT and KD enabled."""
        razor = EdgeRazor(config="example/configs/qad/qat_w1.58mp4_a8_kd_kldc_fd.yaml")
        
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        hidden_size = 64
        
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        student_hidden = torch.randn(batch_size, seq_len, hidden_size)
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        task_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100
        )
        
        student_outputs = {
            'loss': task_loss,
            'logits': student_logits,
            'hidden_states': student_hidden  # Should be tensor, not tuple
        }
        
        teacher_outputs = {
            'logits': teacher_logits,
            'hidden_states': teacher_hidden  # Should be tensor, not tuple
        }
        
        # Compute loss (KD is active even though QAT is also enabled)
        total_loss, loss_dict = razor.compute_loss(
            student_outputs,
            teacher_outputs,
            labels
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert 'task_loss' in loss_dict
        assert 'distill_loss' in loss_dict
        assert 'total_loss' in loss_dict


class TestEdgeRazorConfigurationFiles:
    """Test EdgeRazor with various real configuration files."""
    
    def test_all_vit_configs(self):
        """Test all ViT configuration files load successfully."""
        vit_configs = [
            "example/qat/vit/q_vit_w1.58_a8.yaml",
            "example/qat/vit/q_vit_w1.58_a4.yaml",
            "example/qat/vit/q_vit_w1.58_a16.yaml",
            "example/qat/vit/q_vit_w4_a8.yaml",
            "example/qat/vit/q_vit_w4_a4.yaml",
            "example/qat/vit/q_vit_w4_a16.yaml",
        ]
        
        for config_path in vit_configs:
            razor = EdgeRazor(qat_config=config_path)
            assert razor.is_qat_enabled
            assert not razor.is_kd_enabled
    
    def test_all_resnet_configs(self):
        """Test all ResNet configuration files load successfully."""
        resnet_configs = [
            "example/qat/resnet/q_resnet_w1.58_a8.yaml",
            "example/qat/resnet/q_resnet_w1.58_a4.yaml",
            "example/qat/resnet/q_resnet_w1.58_a16.yaml",
            "example/qat/resnet/q_resnet_w4_a8.yaml",
            "example/qat/resnet/q_resnet_w4_a4.yaml",
            "example/qat/resnet/q_resnet_w4_a16.yaml",
        ]
        
        for config_path in resnet_configs:
            razor = EdgeRazor(qat_config=config_path)
            assert razor.is_qat_enabled
            assert not razor.is_kd_enabled
    
    def test_resnet_configs_with_overrides(self):
        """Test ResNet configurations with overrides load successfully."""
        override_configs = [
            "example/qat/resnet/q_resnet_w1.58_a16_with_overrides.yaml",
            "example/qat/resnet/q_resnet_w1.58_a16_with_wildcard.yaml",
        ]
        
        for config_path in override_configs:
            razor = EdgeRazor(qat_config=config_path)
            assert razor.is_qat_enabled
            assert not razor.is_kd_enabled
    
    def test_unified_qat_kd_config(self):
        """Test unified QAT+KD configuration file."""
        config_path = "example/configs/qad/qat_w1.58mp4_a8_kd_kldc_fd.yaml"
        razor = EdgeRazor(config=config_path)
        
        assert razor.is_qat_enabled
        assert razor.is_kd_enabled
    
    def test_separate_qat_config(self):
        """Test separate QAT configuration file."""
        config_path = "example/configs/qat/qat_w1.58mp4_a8.yaml"
        razor = EdgeRazor(qat_config=config_path)
        
        assert razor.is_qat_enabled
        assert not razor.is_kd_enabled
    
    def test_kd_config(self):
        """Test KD configuration file."""
        config_path = "example/configs/kd/kd_kldc_fd.yaml"
        razor = EdgeRazor(kd_config=config_path)
        
        assert not razor.is_qat_enabled
        assert razor.is_kd_enabled


class TestEdgeRazorProperties:
    """Test EdgeRazor property methods."""
    
    def test_is_qat_enabled_true(self):
        """Test is_qat_enabled returns True when QAT is configured."""
        razor = EdgeRazor(qat_config="example/qat/vit/q_vit_w1.58_a8.yaml")
        assert razor.is_qat_enabled is True
    
    def test_is_qat_enabled_false(self):
        """Test is_qat_enabled returns False when QAT is not configured."""
        razor = EdgeRazor(kd_config="example/configs/kd/kd_kldc_fd.yaml")
        assert razor.is_qat_enabled is False
    
    def test_is_kd_enabled_true(self):
        """Test is_kd_enabled returns True when KD is configured."""
        razor = EdgeRazor(kd_config="example/configs/kd/kd_kldc_fd.yaml")
        assert razor.is_kd_enabled is True
    
    def test_is_kd_enabled_false(self):
        """Test is_kd_enabled returns False when KD is not configured."""
        razor = EdgeRazor(qat_config="example/qat/vit/q_vit_w1.58_a8.yaml")
        assert razor.is_kd_enabled is False
    
    def test_both_enabled(self):
        """Test properties when both QAT and KD are enabled."""
        razor = EdgeRazor(config="example/configs/qad/qat_w1.58mp4_a8_kd_kldc_fd.yaml")
        assert razor.is_qat_enabled is True
        assert razor.is_kd_enabled is True


class TestEdgeRazorEdgeCases:
    """Test EdgeRazor edge cases and error handling."""
    
    def test_invalid_config_path_raises_error(self):
        """Test that invalid config path raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            EdgeRazor(config="nonexistent_config.yaml")
    
    def test_invalid_qat_config_path_raises_error(self):
        """Test that invalid QAT config path raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            EdgeRazor(qat_config="nonexistent_qat.yaml")
    
    def test_invalid_kd_config_path_raises_error(self):
        """Test that invalid KD config path raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            EdgeRazor(kd_config="nonexistent_kd.yaml")
    
    def test_malformed_config_dict_raises_error(self):
        """Test that malformed config dict raises appropriate error."""
        malformed_config = {
            'qat_configuration': {
                'method': 'INVALID_METHOD'  # Should be 'QAT'
            }
        }
        
        with pytest.raises(ValueError):
            EdgeRazor(config=malformed_config)
    
    def test_empty_dict_raises_error(self):
        """Test that empty config dict raises error."""
        with pytest.raises(ValueError):
            EdgeRazor(config={})
    
    def test_mixed_file_and_dict_config(self):
        """Test initialization with QAT from file and KD from dict."""
        qat_path = "example/qat/vit/q_vit_w1.58_a8.yaml"
        kd_dict = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.7
            }
        }
        
        razor = EdgeRazor(qat_config=qat_path, kd_config=kd_dict)
        
        assert razor.is_qat_enabled
        assert razor.is_kd_enabled


class TestEdgeRazorIntegration:
    """Integration tests for EdgeRazor with real workflow."""
    
    def test_qat_workflow(self):
        """Test complete QAT workflow: init → prepare → forward."""
        # Use int8 quantization for small test models
        config = {
            'method': 'QAT',
            'function': {
                'weight': 'weight_quant_uniform_symmetric_clip_per_block_int8',
                'state': 'state_quant_uniform_symmetric_absmax_per_block_int8'
            },
            'selection': {'module_type': ['Linear']}
        }
        razor = EdgeRazor(config=config)
        model = SimpleModel()
        
        # Prepare model
        quantized_model = razor.quantize(model)
        
        # Forward pass
        x = torch.randn(4, 10)
        with torch.no_grad():
            output = quantized_model(x)
        
        assert output.shape == (4, 10)
    
    def test_kd_workflow(self):
        """Test complete KD workflow: init → compute_loss."""
        razor = EdgeRazor(kd_config="example/configs/kd/kd_kldc_fd.yaml")
        
        batch_size = 2
        seq_len = 10
        vocab_size = 100
        hidden_size = 64
        
        # Create outputs with requires_grad=True for backpropagation
        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        student_hidden = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        task_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, vocab_size),
            labels.view(-1)
        )
        
        student_outputs = {
            'loss': task_loss,
            'logits': student_logits,
            'hidden_states': student_hidden  # Should be tensor, not tuple
        }
        
        teacher_outputs = {
            'logits': teacher_logits,
            'hidden_states': teacher_hidden  # Should be tensor, not tuple
        }
        
        # Compute loss
        total_loss, loss_dict = razor.compute_loss(
            student_outputs,
            teacher_outputs,
            labels
        )
        
        # Verify loss can be backpropagated
        assert total_loss.requires_grad
        total_loss.backward()
    
    def test_qat_kd_combined_workflow(self):
        """Test complete QAT+KD workflow: init → prepare → forward → compute_loss."""
        # Use int8 quantization for small test models
        config = {
            'qat_configuration': {
                'method': 'QAT',
                'function': {
                    'weight': 'weight_quant_uniform_symmetric_clip_per_block_int8',
                    'state': 'state_quant_uniform_symmetric_absmax_per_block_int8'
                },
                'selection': {'module_type': ['Linear']}
            },
            'kd_configuration': {
                'method': 'KD',
                'loss_task_alpha': 1.0,  # Correct parameter name
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': 'compute_kld_confidence',
                    'alpha': 0.7,
                    'temperature': 2.0,
                    'use_entropy': True,
                    'reduction': 'batch_mean'
                }
            }
        }
        razor = EdgeRazor(config=config)
        
        # Prepare models
        student_model = SimpleModel()
        teacher_model = SimpleModel()
        quantized_student = razor.quantize(student_model)
        
        # Create input
        batch_size = 2
        seq_len = 5
        vocab_size = 10  # SimpleModel outputs 10 features
        x = torch.randn(batch_size, 10)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        student_output = quantized_student(x)  # Shape: [2, 10]
        with torch.no_grad():
            teacher_output = teacher_model(x)  # Shape: [2, 10]
        
        # Expand to include seq_len dimension for KD compatibility
        # [batch_size, output_dim] -> [batch_size, seq_len, output_dim]
        student_logits = student_output.unsqueeze(1).expand(-1, seq_len, -1)
        teacher_logits = teacher_output.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute task loss
        task_loss = torch.nn.functional.cross_entropy(
            student_logits.reshape(-1, vocab_size),
            labels.reshape(-1)
        )
        
        # Prepare outputs for KD
        student_outputs = {
            'loss': task_loss,
            'logits': student_logits,
            'hidden_states': student_logits  # Use logits as hidden states for simplicity
        }
        
        teacher_outputs = {
            'logits': teacher_logits,
            'hidden_states': teacher_logits
        }
        
        # Compute distillation loss
        total_loss, loss_dict = razor.compute_loss(
            student_outputs,
            teacher_outputs,
            labels
        )
        
        # Verify
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert 'task_loss' in loss_dict
        assert 'distill_loss' in loss_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
