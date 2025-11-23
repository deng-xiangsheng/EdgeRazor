"""
Count parameters in ResNet-18 model.

This script instantiates ResNet-18 and counts:
- Total parameters
- Trainable parameters
- Parameters by layer type (Conv2d, BatchNorm2d, Linear)
"""

import torch
from torchvision.models import resnet18


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_parameters_by_type(model):
    """Count parameters by module type."""
    type_counts = {}
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # Skip container modules
        if module_type in ['ResNet', 'Sequential', 'BasicBlock']:
            continue
        
        # Count parameters in this module
        params = sum(p.numel() for p in module.parameters(recurse=False))
        
        if params > 0:
            if module_type not in type_counts:
                type_counts[module_type] = {'count': 0, 'params': 0, 'modules': []}
            
            type_counts[module_type]['count'] += 1
            type_counts[module_type]['params'] += params
            type_counts[module_type]['modules'].append((name, params))
    
    return type_counts


def main():
    print("=" * 80)
    print("ResNet-18 Parameter Count")
    print("=" * 80)
    print()
    
    # Create ResNet-18 model
    model = resnet18(weights=None)
    
    # Replace final layer for MNIST (10 classes)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 10)
    
    print(f"Model: ResNet-18 (modified for MNIST with 10 classes)")
    print(f"Final layer: Linear({in_features}, 10)")
    print()
    
    # Count total parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,}")
    print()
    
    # Count by type
    print("=" * 80)
    print("Parameters by Layer Type")
    print("=" * 80)
    print()
    
    type_counts = count_parameters_by_type(model)
    
    # Sort by parameter count (descending)
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1]['params'], reverse=True)
    
    print(f"{'Layer Type':<15} {'Count':>8} {'Parameters':>15} {'Percentage':>12}")
    print("-" * 80)
    
    for module_type, info in sorted_types:
        percentage = (info['params'] / total_params) * 100
        print(f"{module_type:<15} {info['count']:>8} {info['params']:>15,} {percentage:>11.2f}%")
    
    print("-" * 80)
    print(f"{'Total':<15} {sum(t[1]['count'] for t in sorted_types):>8} {total_params:>15,} {'100.00%':>12}")
    print()
    
    # Verify total parameters from type counts
    counted_params = sum(t[1]['params'] for t in sorted_types)
    if counted_params != total_params:
        print(f"⚠️  Warning: Counted parameters ({counted_params:,}) != Total parameters ({total_params:,})")
        print(f"   Difference: {total_params - counted_params:,} parameters unaccounted for")
        print()
    
    # Detailed breakdown for all layer types with parameters
    print("=" * 80)
    print("Detailed Layer Information")
    print("=" * 80)
    print()
    
    for module_type, info in sorted_types:
        print(f"{module_type} Layers ({info['count']} layers, {info['params']:,} params):")
        print("-" * 80)
        
        # Sort by parameters
        sorted_modules = sorted(info['modules'], key=lambda x: x[1], reverse=True)
        
        for idx, (name, params) in enumerate(sorted_modules, 1):
            percentage = (params / total_params) * 100
            print(f"  {idx:2d}. {name:<60} {params:>12,} ({percentage:>6.2f}%)")
        
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
