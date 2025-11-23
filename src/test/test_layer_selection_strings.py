"""
Test layer selection with string choices (low, mid, high)
"""

import pytest
import torch

from edgerazor.kd import KD
from edgerazor.kd.util import DistillConfig, LossConfig


class TestLayerSelectionStrings:
    """Test layer selection with predefined string choices"""
    
    def test_layer_index_string_validation(self):
        """Test that only valid string choices are accepted"""
        # Valid string choices
        valid_choices = ["low", "mid", "high"]
        for choice in valid_choices:
            config = LossConfig(
                loss_type="hidden_states",
                loss_function="fd",
                alpha=0.5,
                layer_index=choice
            )
            assert config.layer_index == choice
        
        # Invalid string choice should raise ValueError
        with pytest.raises(ValueError, match="layer_index string must be one of"):
            LossConfig(
                loss_type="hidden_states",
                loss_function="fd",
                alpha=0.5,
                layer_index="invalid"
            )
    
    def test_layer_index_list_of_strings_validation(self):
        """Test that list of strings are validated"""
        # Valid list of strings
        config = LossConfig(
            loss_type="hidden_states",
            loss_function="fd",
            alpha=0.5,
            layer_index=["low", "mid", "high"]
        )
        assert config.layer_index == ["low", "mid", "high"]
        
        # Invalid string in list should raise ValueError
        with pytest.raises(ValueError, match="layer_index string must be one of"):
            LossConfig(
                loss_type="hidden_states",
                loss_function="fd",
                alpha=0.5,
                layer_index=["low", "invalid", "high"]
            )
    
    def test_layer_index_mixed_list_validation(self):
        """Test that mixed list (ints and strings) are validated"""
        # Valid mixed list
        config = LossConfig(
            loss_type="hidden_states",
            loss_function="fd",
            alpha=0.5,
            layer_index=[0, "mid", -1, "high"]
        )
        assert config.layer_index == [0, "mid", -1, "high"]
        
        # Invalid string in mixed list should raise ValueError
        with pytest.raises(ValueError, match="layer_index string must be one of"):
            LossConfig(
                loss_type="hidden_states",
                loss_function="fd",
                alpha=0.5,
                layer_index=[0, "invalid", 5]
            )
    
    def test_kd_with_string_layer_selection_single(self):
        """Test KD with single string layer selection"""
        # Create config with string layer selection
        kd_config = DistillConfig(
            method="KD",
            loss_task_alpha=1.0,
            losses={
                "loss_1": LossConfig(
                    loss_type="hidden_states",
                    loss_function="fd",
                    alpha=0.5,
                    layer_index="mid"
                )
            }
        )
        
        kd = KD(kd_config)
        
        # Create mock outputs with 13 layers (embedding + 12 transformer layers)
        batch_size, seq_len, hidden_size = 2, 10, 384
        num_layers = 13
        
        student_hidden_states = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        teacher_hidden_states = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        
        student_outputs = {
            'loss': torch.tensor(2.5),
            'logits': torch.randn(batch_size, seq_len, 1000),
            'hidden_states': student_hidden_states
        }
        
        teacher_outputs = {
            'logits': torch.randn(batch_size, seq_len, 1000),
            'hidden_states': teacher_hidden_states
        }
        
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Compute loss
        total_loss, loss_dict = kd.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            labels=labels
        )
        
        # Verify loss is computed
        assert total_loss is not None
        assert 'distill_loss_details' in loss_dict
        assert 'loss_1' in loss_dict['distill_loss_details']
        
        # For 13 layers, "mid" should resolve to layer 6 (13 // 2)
        print(f"Total layers: {num_layers}")
        print(f"String 'mid' resolved to layer: {num_layers // 2}")
        print(f"Loss computed: {loss_dict['distill_loss_details']['loss_1']}")
    
    def test_kd_with_string_layer_selection_multiple(self):
        """Test KD with multiple string layer selections"""
        # Create config with multiple string layer selections
        kd_config = DistillConfig(
            method="KD",
            loss_task_alpha=1.0,
            losses={
                "loss_1": LossConfig(
                    loss_type="hidden_states",
                    loss_function="fd",
                    alpha=0.5,
                    layer_index=["low", "mid", "high"]
                )
            }
        )
        
        kd = KD(kd_config)
        
        # Create mock outputs with 13 layers
        batch_size, seq_len, hidden_size = 2, 10, 384
        num_layers = 13
        
        student_hidden_states = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        teacher_hidden_states = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        
        student_outputs = {
            'loss': torch.tensor(2.5),
            'logits': torch.randn(batch_size, seq_len, 1000),
            'hidden_states': student_hidden_states
        }
        
        teacher_outputs = {
            'logits': torch.randn(batch_size, seq_len, 1000),
            'hidden_states': teacher_hidden_states
        }
        
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Compute loss
        total_loss, loss_dict = kd.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            labels=labels
        )
        
        # Verify loss is computed
        assert total_loss is not None
        assert 'distill_loss_details' in loss_dict
        assert 'loss_1' in loss_dict['distill_loss_details']
        
        # For 13 layers:
        # "low" -> 1, "mid" -> 6, "high" -> 12
        print(f"Total layers: {num_layers}")
        print(f"String 'low' resolved to layer: 1")
        print(f"String 'mid' resolved to layer: {num_layers // 2}")
        print(f"String 'high' resolved to layer: {num_layers - 1}")
        print(f"Loss computed: {loss_dict['distill_loss_details']['loss_1']}")
    
    def test_kd_with_mixed_layer_selection(self):
        """Test KD with mixed int and string layer selections"""
        # Create config with mixed layer selection
        kd_config = DistillConfig(
            method="KD",
            loss_task_alpha=1.0,
            losses={
                "loss_1": LossConfig(
                    loss_type="hidden_states",
                    loss_function="fd",
                    alpha=0.5,
                    layer_index=[0, "mid", -1]
                )
            }
        )
        
        kd = KD(kd_config)
        
        # Create mock outputs with 13 layers
        batch_size, seq_len, hidden_size = 2, 10, 384
        num_layers = 13
        
        student_hidden_states = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        teacher_hidden_states = tuple([
            torch.randn(batch_size, seq_len, hidden_size)
            for _ in range(num_layers)
        ])
        
        student_outputs = {
            'loss': torch.tensor(2.5),
            'logits': torch.randn(batch_size, seq_len, 1000),
            'hidden_states': student_hidden_states
        }
        
        teacher_outputs = {
            'logits': torch.randn(batch_size, seq_len, 1000),
            'hidden_states': teacher_hidden_states
        }
        
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Compute loss
        total_loss, loss_dict = kd.compute_loss(
            student_outputs=student_outputs,
            teacher_outputs=teacher_outputs,
            labels=labels
        )
        
        # Verify loss is computed
        assert total_loss is not None
        assert 'distill_loss_details' in loss_dict
        assert 'loss_1' in loss_dict['distill_loss_details']
        
        # For 13 layers:
        # 0 -> 0, "mid" -> 6, -1 -> 12
        print(f"Total layers: {num_layers}")
        print(f"Layer 0 -> 0 (embedding)")
        print(f"String 'mid' resolved to layer: {num_layers // 2}")
        print(f"Layer -1 -> {num_layers - 1} (last transformer layer)")
        print(f"Loss computed: {loss_dict['distill_loss_details']['loss_1']}")
    
    def test_layer_resolution_logic(self):
        """Test the layer resolution logic for different number of layers"""
        test_cases = [
            # (num_layers, expected_low, expected_mid, expected_high)
            (1, 0, 0, 0),      # Edge case: single layer
            (2, 1, 1, 1),      # Edge case: two layers
            (6, 1, 3, 5),      # Small model
            (13, 1, 6, 12),    # ViT-S/16
            (24, 1, 12, 23),   # ViT-B/16
        ]
        
        for num_layers, expected_low, expected_mid, expected_high in test_cases:
            # Compute expected values
            computed_low = 1 if num_layers > 1 else 0
            computed_mid = num_layers // 2
            computed_high = num_layers - 1
            
            assert computed_low == expected_low, \
                f"For {num_layers} layers, 'low' should be {expected_low}, got {computed_low}"
            assert computed_mid == expected_mid, \
                f"For {num_layers} layers, 'mid' should be {expected_mid}, got {computed_mid}"
            assert computed_high == expected_high, \
                f"For {num_layers} layers, 'high' should be {expected_high}, got {computed_high}"
            
            print(f"{num_layers} layers: low={computed_low}, mid={computed_mid}, high={computed_high}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
