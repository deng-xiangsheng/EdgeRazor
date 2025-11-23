"""
Test suite for Knowledge Distillation (KD) module

Tests:
1. Configuration loading (YAML, JSON, dict, DistillConfig object)
2. Loss computation correctness
3. Multi-loss configuration
4. loss_task_alpha functionality
5. Edge cases and error handling
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from edgerazor.kd import KD, DistillConfig, LossConfig


class TestKDConfiguration:
    """Test KD configuration loading from different sources"""
    
    def test_load_from_dict(self):
        """Test loading configuration from Python dictionary"""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.5,
                'temperature': 2.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            }
        }
        
        kd = KD(config)
        
        assert kd.config.method == 'KD'
        assert kd.config.loss_task_alpha == 1.0
        assert len(kd.config.losses) == 1
        assert 'loss_1' in kd.config.losses
        assert kd.config.losses['loss_1'].alpha == 0.5
        assert kd.config.losses['loss_1'].temperature == 2.0
        print("✓ Test load_from_dict passed")
    
    def test_load_from_yaml(self):
        """Test loading configuration from YAML file"""
        config_dict = {
            'method': 'KD',
            'loss_task_alpha': 1.5,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldc',
                'alpha': 0.7,
                'temperature': 2.5,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            }
        }
        
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            yaml_path = f.name
        
        try:
            kd = KD(yaml_path)
            
            assert kd.config.method == 'KD'
            assert kd.config.loss_task_alpha == 1.5
            assert len(kd.config.losses) == 1
            assert kd.config.losses['loss_1'].loss_function == 'kldc'
            print("✓ Test load_from_yaml passed")
        finally:
            Path(yaml_path).unlink()
    
    def test_load_from_json(self):
        """Test loading configuration from JSON file"""
        config_dict = {
            'method': 'KD',
            'loss_task_alpha': 2.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldr',
                'alpha': 0.6,
                'temperature': 3.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'mean',
                'use_entropy': False
            }
        }
        
        # Create temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_dict, f)
            json_path = f.name
        
        try:
            kd = KD(json_path)
            
            assert kd.config.method == 'KD'
            assert kd.config.loss_task_alpha == 2.0
            assert kd.config.losses['loss_1'].reduction == 'mean'
            print("✓ Test load_from_json passed")
        finally:
            Path(json_path).unlink()
    
    def test_load_from_distill_config(self):
        """Test loading from DistillConfig object"""
        loss_config = LossConfig(
            loss_type='logits',
            loss_function='kldf',
            alpha=0.8,
            temperature=2.0
        )
        
        distill_config = DistillConfig(
            method='KD',
            loss_task_alpha=1.0,
            losses={'loss_1': loss_config}
        )
        
        kd = KD(distill_config)
        
        assert kd.config.method == 'KD'
        assert kd.config.loss_task_alpha == 1.0
        assert kd.config.losses['loss_1'].alpha == 0.8
        print("✓ Test load_from_distill_config passed")
    
    def test_multi_loss_configuration(self):
        """Test configuration with multiple losses"""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldc',
                'alpha': 0.7,
                'temperature': 2.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            },
            'loss_2': {
                'loss_type': 'hidden_states',
                'loss_function': 'fd',
                'alpha': 0.5,
                'padding_id': -100,
                'reduction': 'batch_mean',
                'normalize': False
            }
        }
        
        kd = KD(config)
        
        assert len(kd.config.losses) == 2
        assert 'loss_1' in kd.config.losses
        assert 'loss_2' in kd.config.losses
        assert kd.config.losses['loss_1'].loss_type == 'logits'
        assert kd.config.losses['loss_2'].loss_type == 'hidden_states'
        print("✓ Test multi_loss_configuration passed")


class TestKDLossComputation:
    """Test KD loss computation correctness"""
    
    def test_single_logits_loss(self):
        """Test single logits-based distillation loss"""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.5,
                'temperature': 2.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            }
        }
        
        kd = KD(config)
        
        # Create test data
        batch_size, seq_len, vocab_size = 2, 10, 100
        task_loss = torch.tensor(2.0)
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        student_outputs = {'loss': task_loss, 'logits': student_logits}
        teacher_outputs = {'logits': teacher_logits}
        
        total_loss, loss_dict = kd.compute_loss(student_outputs, teacher_outputs, labels)
        
        # Verify loss dict structure
        assert 'task_loss' in loss_dict
        assert 'distill_loss' in loss_dict
        assert 'distill_loss_details' in loss_dict
        assert 'total_loss' in loss_dict
        assert 'loss_1' in loss_dict['distill_loss_details']
        
        # Verify loss computation: total_loss = loss_task_alpha * task_loss + distill_loss
        expected_task_loss_weighted = 1.0 * task_loss.item()
        expected_total = expected_task_loss_weighted + loss_dict['distill_loss']
        assert abs(loss_dict['total_loss'] - expected_total) < 1e-5
        
        # Verify distill_loss is weighted by alpha
        raw_loss_1 = loss_dict['distill_loss_details']['loss_1']
        expected_distill_loss = 0.5 * raw_loss_1
        assert abs(loss_dict['distill_loss'] - expected_distill_loss) < 1e-5
        
        print("✓ Test single_logits_loss passed")
    
    def test_multi_loss_computation(self):
        """Test multiple losses computation"""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.7,
                'temperature': 2.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            },
            'loss_2': {
                'loss_type': 'hidden_states',
                'loss_function': 'fd',
                'alpha': 0.3,
                'padding_id': -100,
                'reduction': 'batch_mean',
                'normalize': False
            }
        }
        
        kd = KD(config)
        
        # Create test data
        batch_size, seq_len, vocab_size, hidden_size = 2, 10, 100, 128
        task_loss = torch.tensor(2.5)
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        student_hidden = torch.randn(batch_size, seq_len, hidden_size)
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        student_outputs = {
            'loss': task_loss,
            'logits': student_logits,
            'hidden_states': student_hidden
        }
        teacher_outputs = {
            'logits': teacher_logits,
            'hidden_states': teacher_hidden
        }
        
        total_loss, loss_dict = kd.compute_loss(student_outputs, teacher_outputs, labels)
        
        # Verify both losses are computed
        assert 'loss_1' in loss_dict['distill_loss_details']
        assert 'loss_2' in loss_dict['distill_loss_details']
        
        # Verify distill_loss is sum of weighted individual losses
        loss_1_raw = loss_dict['distill_loss_details']['loss_1']
        loss_2_raw = loss_dict['distill_loss_details']['loss_2']
        expected_distill_loss = 0.7 * loss_1_raw + 0.3 * loss_2_raw
        assert abs(loss_dict['distill_loss'] - expected_distill_loss) < 1e-5
        
        # Verify total loss
        expected_total = 1.0 * task_loss.item() + expected_distill_loss
        assert abs(loss_dict['total_loss'] - expected_total) < 1e-5
        
        print("✓ Test multi_loss_computation passed")
    
    def test_loss_task_alpha(self):
        """Test loss_task_alpha weight functionality"""
        config = {
            'method': 'KD',
            'loss_task_alpha': 2.0,  # Task loss weight = 2.0
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.5,
                'temperature': 2.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            }
        }
        
        kd = KD(config)
        
        # Create test data
        batch_size, seq_len, vocab_size = 2, 10, 100
        task_loss = torch.tensor(1.5)
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        student_outputs = {'loss': task_loss, 'logits': student_logits}
        teacher_outputs = {'logits': teacher_logits}
        
        total_loss, loss_dict = kd.compute_loss(student_outputs, teacher_outputs, labels)
        
        # Verify task_loss is weighted by loss_task_alpha
        expected_task_loss_weighted = 2.0 * task_loss.item()
        expected_total = expected_task_loss_weighted + loss_dict['distill_loss']
        assert abs(loss_dict['total_loss'] - expected_total) < 1e-5
        
        # Verify task_loss in loss_dict is the original value
        assert abs(loss_dict['task_loss'] - task_loss.item()) < 1e-5
        
        print("✓ Test loss_task_alpha passed")
    
    def test_different_kld_functions(self):
        """Test different KLD loss functions (kldf, kldr, kldc)"""
        kld_functions = ['kldf', 'kldr', 'kldc']
        
        for kld_func in kld_functions:
            config = {
                'method': 'KD',
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'loss_function': kld_func,
                    'alpha': 0.5,
                    'temperature': 2.0,
                    'padding_id': -100,
                    'is_router_logits': False,
                    'reduction': 'batch_mean',
                    'use_entropy': True
                }
            }
            
            kd = KD(config)
            
            # Create test data
            batch_size, seq_len, vocab_size = 2, 10, 100
            task_loss = torch.tensor(2.0)
            student_logits = torch.randn(batch_size, seq_len, vocab_size)
            teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len))
            
            student_outputs = {'loss': task_loss, 'logits': student_logits}
            teacher_outputs = {'logits': teacher_logits}
            
            total_loss, loss_dict = kd.compute_loss(student_outputs, teacher_outputs, labels)
            
            # Verify loss computation works
            assert 'loss_1' in loss_dict['distill_loss_details']
            assert loss_dict['distill_loss'] > 0
            assert total_loss > 0
        
        print("✓ Test different_kld_functions passed")


class TestKDEdgeCases:
    """Test edge cases and error handling"""
    
    def test_missing_task_loss(self):
        """Test error when task_loss is missing from student_outputs"""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.5,
                'temperature': 2.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            }
        }
        
        kd = KD(config)
        
        batch_size, seq_len, vocab_size = 2, 10, 100
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Missing 'loss' field
        student_outputs = {'logits': student_logits}
        teacher_outputs = {'logits': teacher_logits}
        
        with pytest.raises(ValueError, match="task_loss not found"):
            kd.compute_loss(student_outputs, teacher_outputs, labels)
        
        print("✓ Test missing_task_loss passed")
    
    def test_missing_logits(self):
        """Test warning when logits are missing"""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.5,
                'temperature': 2.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            }
        }
        
        kd = KD(config)
        
        task_loss = torch.tensor(2.0)
        labels = torch.randint(0, 100, (2, 10))
        
        # Missing logits
        student_outputs = {'loss': task_loss}
        teacher_outputs = {}
        
        total_loss, loss_dict = kd.compute_loss(student_outputs, teacher_outputs, labels)
        
        # Should still return task_loss
        assert loss_dict['task_loss'] == task_loss.item()
        assert loss_dict['distill_loss'] == 0.0
        assert loss_dict['total_loss'] == task_loss.item()
        
        print("✓ Test missing_logits passed")
    
    def test_invalid_loss_type(self):
        """Test error with invalid loss_type"""
        with pytest.raises(ValueError, match="loss_type must be one of"):
            LossConfig(
                loss_type='invalid_type',
                loss_function='kldf',
                alpha=0.5
            )
        
        print("✓ Test invalid_loss_type passed")
    
    def test_invalid_reduction(self):
        """Test error with invalid reduction mode"""
        with pytest.raises(ValueError, match="reduction must be one of"):
            LossConfig(
                loss_type='logits',
                loss_function='kldf',
                alpha=0.5,
                reduction='invalid_reduction'
            )
        
        print("✓ Test invalid_reduction passed")
    
    def test_model_output_format(self):
        """Test with transformers ModelOutput-like object"""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'logits',
                'loss_function': 'kldf',
                'alpha': 0.5,
                'temperature': 2.0,
                'padding_id': -100,
                'is_router_logits': False,
                'reduction': 'batch_mean',
                'use_entropy': True
            }
        }
        
        kd = KD(config)
        
        # Create mock ModelOutput object
        class ModelOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        batch_size, seq_len, vocab_size = 2, 10, 100
        task_loss = torch.tensor(2.0)
        student_logits = torch.randn(batch_size, seq_len, vocab_size)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        student_outputs = ModelOutput(loss=task_loss, logits=student_logits)
        teacher_outputs = ModelOutput(loss=None, logits=teacher_logits)
        
        total_loss, loss_dict = kd.compute_loss(student_outputs, teacher_outputs, labels)
        
        # Verify it works with ModelOutput format
        assert 'task_loss' in loss_dict
        assert 'distill_loss' in loss_dict
        assert loss_dict['task_loss'] == task_loss.item()
        
        print("✓ Test model_output_format passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Running KD Module Tests")
    print("=" * 80)
    
    # Configuration tests
    print("\n[Configuration Tests]")
    config_tests = TestKDConfiguration()
    config_tests.test_load_from_dict()
    config_tests.test_load_from_yaml()
    config_tests.test_load_from_json()
    config_tests.test_load_from_distill_config()
    config_tests.test_multi_loss_configuration()
    
    # Loss computation tests
    print("\n[Loss Computation Tests]")
    loss_tests = TestKDLossComputation()
    loss_tests.test_single_logits_loss()
    loss_tests.test_multi_loss_computation()
    loss_tests.test_loss_task_alpha()
    loss_tests.test_different_kld_functions()
    
    # Edge case tests
    print("\n[Edge Case Tests]")
    edge_tests = TestKDEdgeCases()
    edge_tests.test_missing_task_loss()
    edge_tests.test_missing_logits()
    edge_tests.test_invalid_loss_type()
    edge_tests.test_invalid_reduction()
    edge_tests.test_model_output_format()
    
    print("\n" + "=" * 80)
    print("All tests passed successfully! ✓")
    print("=" * 80)


if __name__ == '__main__':
    run_all_tests()
