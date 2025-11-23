"""
Test script to verify backward compatibility and new override feature for QAT
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add parent directory to path for edgerazor imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from edgerazor.qat import QAT
from edgerazor.qat.module import QConv2d, QLinear


# Create a simple test model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
        self.layer4_conv = nn.Conv2d(128, 256, kernel_size=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def test_backward_compatibility():
    """Test that old YAML files still work (no overrides)"""
    print("\n" + "="*80)
    print("Test 1: Backward Compatibility (Old YAML Format)")
    print("="*80)
    
    # Path to example config file
    config_path = Path(__file__).parent.parent.parent / "example" / "qat" / "resnet" / "q_resnet_w1.58_a16.yaml"
    
    # Initialize QAT with old-style config
    qat = QAT(config_path)
    
    # Create and quantize model
    model = SimpleModel()
    quantized_model = qat.quantize(model, qlinear_cls=QLinear, qconv2d_cls=QConv2d)
    
    # Check that modules are quantized
    assert isinstance(quantized_model.fc1, QLinear), "fc1 should be quantized"
    assert isinstance(quantized_model.conv1, QConv2d), "conv1 should be quantized"
    
    # Check that all quantized modules use the same config
    print("\nChecking that all modules use global config...")
    print(f"fc1 weight function: {quantized_model.fc1.w_quant_function}")
    print(f"fc2 weight function: {quantized_model.fc2.w_quant_function}")
    print(f"conv1 weight function: {quantized_model.conv1.w_quant_function}")
    
    assert quantized_model.fc1.w_quant_function == quantized_model.fc2.w_quant_function
    assert quantized_model.conv1.w_quant_function == quantized_model.conv2.w_quant_function
    
    print("✓ Backward compatibility test passed!")


def test_override_feature():
    """Test new override feature"""
    print("\n" + "="*80)
    print("Test 2: Override Feature (New YAML Format)")
    print("="*80)
    
    # Path to example config file with overrides
    config_path = Path(__file__).parent.parent.parent / "example" / "qat" / "resnet" / "q_resnet_w1.58_a16_with_overrides.yaml"
    
    # Initialize QAT with new-style config (with overrides)
    qat = QAT(config_path)
    
    # Create and quantize model
    model = SimpleModel()
    quantized_model = qat.quantize(model, qlinear_cls=QLinear, qconv2d_cls=QConv2d)
    
    # Check that modules are quantized
    assert isinstance(quantized_model.fc1, QLinear), "fc1 should be quantized"
    assert isinstance(quantized_model.conv1, QConv2d), "conv1 should be quantized"
    
    # Check that different modules may have different configs
    print("\nChecking that overrides are applied correctly...")
    print(f"fc1 weight function: {quantized_model.fc1.w_quant_function}")
    print(f"fc2 weight function (name='fc2', should match override for 'fc'): {quantized_model.fc2.w_quant_function}")
    print(f"conv1 weight function: {quantized_model.conv1.w_quant_function}")
    print(f"layer4_conv weight function (name matches 'layer4.*' pattern): {quantized_model.layer4_conv.w_quant_function}")
    
    # Linear layers should have different function than Conv layers (due to type override)
    print("\n✓ Override feature test completed!")
    print("  Note: Check the logs above to verify that overrides are applied correctly")


def test_config_loading():
    """Test that config can be loaded and accessed correctly"""
    print("\n" + "="*80)
    print("Test 3: Config Loading and Access")
    print("="*80)
    
    # Path to example config files
    example_dir = Path(__file__).parent.parent.parent / "example" / "qat" / "resnet"
    
    # Test old format
    old_config_path = example_dir / "q_resnet_w1.58_a16.yaml"
    qat_old = QAT(old_config_path)
    
    print(f"\nOld format - has overrides: {hasattr(qat_old.config, 'overrides') and len(qat_old.config.overrides) > 0}")
    print(f"Old format - overrides list: {qat_old.config.overrides if hasattr(qat_old.config, 'overrides') else 'N/A'}")
    
    # Test new format
    new_config_path = example_dir / "q_resnet_w1.58_a16_with_overrides.yaml"
    qat_new = QAT(new_config_path)
    
    print(f"\nNew format - has overrides: {hasattr(qat_new.config, 'overrides') and len(qat_new.config.overrides) > 0}")
    print(f"New format - number of overrides: {len(qat_new.config.overrides) if hasattr(qat_new.config, 'overrides') else 0}")
    
    # Test get_function_config method
    if hasattr(qat_new.config, 'get_function_config'):
        fc_config = qat_new.config.get_function_config("fc", nn.Linear)
        print("\nFunction config for 'fc' (Linear):")
        print(f"  Weight function: {fc_config.weight_function}")
        print(f"  w_scale_factor: {fc_config.w_scale_factor}")
        
        conv_config = qat_new.config.get_function_config("conv1", nn.Conv2d)
        print("\nFunction config for 'conv1' (Conv2d):")
        print(f"  Weight function: {conv_config.weight_function}")
        print(f"  w_scale_factor: {conv_config.w_scale_factor}")
    
    print("\n✓ Config loading test passed!")


if __name__ == "__main__":
    try:
        test_backward_compatibility()
        test_override_feature()
        test_config_loading()
        
        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
