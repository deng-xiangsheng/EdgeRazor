"""
Test layer selection feature for hidden_states distillation in KD module.
"""

import torch

from edgerazor.kd import KD


class TestKDLayerSelection:
    """Test layer selection for hidden_states distillation."""
    
    def test_single_layer_index(self):
        """Test distillation with single layer index."""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'hidden_states',
                'loss_function': 'compute_fd',
                'alpha': 0.5,
                'layer_index': 1,  # Select layer 1
                'reduction': 'batch_mean'
            }
        }
        
        kd = KD(config)
        
        # Create outputs with tuple of hidden_states
        batch_size, seq_len, hidden_size = 2, 10, 64
        num_layers = 4
        
        task_loss = torch.tensor(2.5, requires_grad=True)
        student_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
            for _ in range(num_layers)
        ])
        teacher_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        
        student_outputs = {
            'loss': task_loss,
            'hidden_states': student_hidden
        }
        teacher_outputs = {
            'hidden_states': teacher_hidden
        }
        labels = torch.randint(0, 100, (batch_size, seq_len))
        
        # Compute loss
        total_loss, loss_dict = kd.compute_loss(
            student_outputs, teacher_outputs, labels
        )
        
        # Verify
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert 'distill_loss' in loss_dict
        assert loss_dict['distill_loss'] > 0
    
    def test_multiple_layer_indices(self):
        """Test distillation with multiple layer indices."""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'hidden_states',
                'loss_function': 'compute_fd',
                'alpha': 0.5,
                'layer_index': [0, 2, -1],  # Select first, third, and last layers
                'reduction': 'batch_mean'
            }
        }
        
        kd = KD(config)
        
        # Create outputs with tuple of hidden_states
        batch_size, seq_len, hidden_size = 2, 10, 64
        num_layers = 5
        
        task_loss = torch.tensor(2.5, requires_grad=True)
        student_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
            for _ in range(num_layers)
        ])
        teacher_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        
        student_outputs = {
            'loss': task_loss,
            'hidden_states': student_hidden
        }
        teacher_outputs = {
            'hidden_states': teacher_hidden
        }
        labels = torch.randint(0, 100, (batch_size, seq_len))
        
        # Compute loss
        total_loss, loss_dict = kd.compute_loss(
            student_outputs, teacher_outputs, labels
        )
        
        # Verify
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert 'distill_loss' in loss_dict
        assert loss_dict['distill_loss'] > 0
    
    def test_negative_layer_index(self):
        """Test distillation with negative layer index."""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'hidden_states',
                'loss_function': 'compute_fd',
                'alpha': 0.5,
                'layer_index': -1,  # Last layer
                'reduction': 'batch_mean'
            }
        }
        
        kd = KD(config)
        
        # Create outputs with tuple of hidden_states
        batch_size, seq_len, hidden_size = 2, 10, 64
        num_layers = 4
        
        task_loss = torch.tensor(2.5, requires_grad=True)
        student_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
            for _ in range(num_layers)
        ])
        teacher_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        
        student_outputs = {
            'loss': task_loss,
            'hidden_states': student_hidden
        }
        teacher_outputs = {
            'hidden_states': teacher_hidden
        }
        labels = torch.randint(0, 100, (batch_size, seq_len))
        
        # Compute loss
        total_loss, loss_dict = kd.compute_loss(
            student_outputs, teacher_outputs, labels
        )
        
        # Verify
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert 'distill_loss' in loss_dict
        assert loss_dict['distill_loss'] > 0
    
    def test_no_layer_index_uses_all_features(self):
        """Test that no layer_index uses all features (backward compatibility)."""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'hidden_states',
                'loss_function': 'compute_fd',
                'alpha': 0.5,
                # No layer_index specified
                'reduction': 'batch_mean'
            }
        }
        
        kd = KD(config)
        
        # Create outputs with single tensor hidden_states
        batch_size, seq_len, hidden_size = 2, 10, 64
        
        task_loss = torch.tensor(2.5, requires_grad=True)
        student_hidden = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_size)
        
        student_outputs = {
            'loss': task_loss,
            'hidden_states': student_hidden
        }
        teacher_outputs = {
            'hidden_states': teacher_hidden
        }
        labels = torch.randint(0, 100, (batch_size, seq_len))
        
        # Compute loss
        total_loss, loss_dict = kd.compute_loss(
            student_outputs, teacher_outputs, labels
        )
        
        # Verify
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert 'distill_loss' in loss_dict
        assert loss_dict['distill_loss'] > 0
    
    def test_layer_index_with_single_tensor_warns(self):
        """Test that layer_index with single tensor gives warning but still works."""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'hidden_states',
                'loss_function': 'compute_fd',
                'alpha': 0.5,
                'layer_index': 1,  # Specified but hidden_states is not tuple
                'reduction': 'batch_mean'
            }
        }
        
        kd = KD(config)
        
        # Create outputs with single tensor (not tuple)
        batch_size, seq_len, hidden_size = 2, 10, 64
        
        task_loss = torch.tensor(2.5, requires_grad=True)
        student_hidden = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
        teacher_hidden = torch.randn(batch_size, seq_len, hidden_size)
        
        student_outputs = {
            'loss': task_loss,
            'hidden_states': student_hidden
        }
        teacher_outputs = {
            'hidden_states': teacher_hidden
        }
        labels = torch.randint(0, 100, (batch_size, seq_len))
        
        # Compute loss (should work but warn)
        total_loss, loss_dict = kd.compute_loss(
            student_outputs, teacher_outputs, labels
        )
        
        # Verify
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert 'distill_loss' in loss_dict
        assert loss_dict['distill_loss'] > 0
    
    def test_out_of_range_layer_index_skips(self):
        """Test that out of range layer index is skipped with warning."""
        config = {
            'method': 'KD',
            'loss_task_alpha': 1.0,
            'loss_1': {
                'loss_type': 'hidden_states',
                'loss_function': 'compute_fd',
                'alpha': 0.5,
                'layer_index': [1, 10],  # 10 is out of range
                'reduction': 'batch_mean'
            }
        }
        
        kd = KD(config)
        
        # Create outputs with tuple of hidden_states
        batch_size, seq_len, hidden_size = 2, 10, 64
        num_layers = 4  # Only 4 layers, so index 10 is out of range
        
        task_loss = torch.tensor(2.5, requires_grad=True)
        student_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
            for _ in range(num_layers)
        ])
        teacher_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        
        student_outputs = {
            'loss': task_loss,
            'hidden_states': student_hidden
        }
        teacher_outputs = {
            'hidden_states': teacher_hidden
        }
        labels = torch.randint(0, 100, (batch_size, seq_len))
        
        # Compute loss (should only use layer 1, skip layer 10)
        total_loss, loss_dict = kd.compute_loss(
            student_outputs, teacher_outputs, labels
        )
        
        # Verify - should still compute loss with valid layer
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert 'distill_loss' in loss_dict
        assert loss_dict['distill_loss'] > 0
    
    def test_layer_index_from_yaml_config(self):
        """Test loading layer_index from YAML configuration file."""
        kd = KD("example/configs/kd/kd_kldc_fd.yaml")
        
        # Verify layer_index is loaded
        assert 'loss_2' in kd.config.losses
        loss_2_config = kd.config.losses['loss_2']
        assert loss_2_config.layer_index == [1, -1]  # From YAML file
        
        # Test with actual data
        batch_size, seq_len, hidden_size = 2, 10, 64
        vocab_size = 100
        num_layers = 4
        
        task_loss = torch.tensor(2.5, requires_grad=True)
        student_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
        student_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size, requires_grad=True)
            for _ in range(num_layers)
        ])
        teacher_hidden = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
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
        
        # Compute loss
        total_loss, loss_dict = kd.compute_loss(
            student_outputs, teacher_outputs, labels
        )
        
        # Verify
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad
        assert 'distill_loss' in loss_dict
        assert loss_dict['distill_loss'] > 0
        assert 'distill_loss_details' in loss_dict
        assert len(loss_dict['distill_loss_details']) == 2  # loss_1 and loss_2
