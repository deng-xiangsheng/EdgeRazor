"""
Inspect specific layers in ResNet-18 model.

This script shows the detailed information about specific layers.
"""

import torch
from torchvision.models import resnet18


def inspect_layer(model, layer_name):
    """Get detailed information about a specific layer."""
    # Navigate to the layer
    parts = layer_name.split('.')
    module = model
    
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    
    return module


def main():
    print("=" * 80)
    print("ResNet-18 Layer Inspection")
    print("=" * 80)
    print()
    
    # Create ResNet-18 model
    model = resnet18(weights=None)
    
    # Replace final layer for MNIST (10 classes)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 10)
    
    # Inspect specific layers
    layers_to_inspect = [
        "layer4.0.downsample.0",
        "layer2.0.conv1",
        "layer2.0.conv2",
        "conv1",
        "layer1.0.conv1",
        "layer3.0.downsample.0",
    ]
    
    for layer_name in layers_to_inspect:
        print(f"Layer: {layer_name}")
        print("-" * 80)
        
        layer = inspect_layer(model, layer_name)
        
        print(f"  Class: {type(layer).__name__}")
        print(f"  Module: {type(layer).__module__}")
        print(f"  Details: {layer}")
        
        # Count parameters
        params = sum(p.numel() for p in layer.parameters())
        print(f"  Parameters: {params:,}")
        
        print()
    
    # Show the structure of a BasicBlock
    print("=" * 80)
    print("BasicBlock Structure (layer2.0)")
    print("=" * 80)
    print()
    print(model.layer2[0])
    print()
    
    # Show downsample structure
    print("=" * 80)
    print("Downsample Structure (layer4.0.downsample)")
    print("=" * 80)
    print()
    if hasattr(model.layer4[0], 'downsample') and model.layer4[0].downsample is not None:
        print(model.layer4[0].downsample)
    else:
        print("No downsample in this block")
    print()
    
    # List all Conv2d layers
    print("=" * 80)
    print("All Conv2d Layers in ResNet-18")
    print("=" * 80)
    print()
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(f"{name:40} {module}")
    
    print()


if __name__ == "__main__":
    main()
