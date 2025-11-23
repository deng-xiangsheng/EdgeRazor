"""
Count parameters in ViT-S/16 model.

This script instantiates ViT-S/16 and counts:
- Total parameters
- Trainable parameters
- Parameters by layer type (Linear, Embedding, LayerNorm)
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


def load_vit_config(config_path):
    """Load ViT configuration from JSON file."""
    with open(config_path) as f:
        vision_cfg = json.load(f)

    config = ViTConfig(
        hidden_size=vision_cfg.get("embed_dim", 384),
        num_hidden_layers=vision_cfg.get("layers", 12),
        num_attention_heads=vision_cfg.get("heads", 6),
        intermediate_size=vision_cfg.get("mlp_ratio", 4) * vision_cfg.get("embed_dim", 384),
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        image_size=vision_cfg.get("image_size", 224),
        patch_size=vision_cfg.get("patch_size", 16),
        num_channels=3,
        num_labels=10,  # MNIST classes
    )

    return config


class ViTForMNIST(nn.Module):
    """ViT model adapted for MNIST classification."""

    def __init__(self, config):
        super().__init__()
        self.vit = ViTModel(config)  # Default: add_pooling_layer=True
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # Use CLS token representation for classification
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        return logits


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_parameters_by_type(model):
    """Count parameters by module type and standalone parameters."""
    type_counts = {}
    counted_param_ids = set()
    
    # First pass: count parameters from leaf modules
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        # Skip container modules
        if module_type in ['ViTForMNIST', 'ViTModel', 'ViTEncoder', 'ViTEmbeddings',
                           'ViTLayer', 'ViTAttention', 'ViTSelfAttention', 'ViTSelfOutput',
                           'ViTIntermediate', 'ViTOutput', 'ModuleList', 'Sequential']:
            continue
        
        # Count parameters in this module (only direct parameters, not from submodules)
        module_params = []
        for param_name, param in module.named_parameters(recurse=False):
            param_id = id(param)
            if param_id not in counted_param_ids:
                counted_param_ids.add(param_id)
                module_params.append((f"{name}.{param_name}" if name else param_name, param.numel()))
        
        if module_params:
            total_params = sum(p[1] for p in module_params)
            if module_type not in type_counts:
                type_counts[module_type] = {'count': 0, 'params': 0, 'modules': []}
            
            type_counts[module_type]['count'] += 1
            type_counts[module_type]['params'] += total_params
            type_counts[module_type]['modules'].append((name, total_params))
    
    # Second pass: collect uncounted parameters (like cls_token, position_embeddings)
    standalone_params = []
    for name, param in model.named_parameters():
        param_id = id(param)
        if param_id not in counted_param_ids:
            counted_param_ids.add(param_id)
            standalone_params.append((name, param.numel()))
    
    # Add standalone parameters if any
    if standalone_params:
        type_counts['Parameter'] = {
            'count': len(standalone_params),
            'params': sum(p[1] for p in standalone_params),
            'modules': standalone_params
        }
    
    return type_counts


def main():
    print("=" * 80)
    print("ViT-S/16 Parameter Count")
    print("=" * 80)
    print()
    
    # Load config
    config_path = Path("src/ViT-S-16.json")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please run this script from the vit project directory")
        sys.exit(1)
    
    # Create ViT model
    vit_config = load_vit_config(config_path)
    model = ViTForMNIST(vit_config)
    
    print(f"Model: ViT-S/16 (Small variant with 16x16 patches)")
    print(f"Hidden size: {vit_config.hidden_size}")
    print(f"Layers: {vit_config.num_hidden_layers}")
    print(f"Attention heads: {vit_config.num_attention_heads}")
    print(f"Intermediate size: {vit_config.intermediate_size}")
    print(f"Image size: {vit_config.image_size}x{vit_config.image_size}")
    print(f"Patch size: {vit_config.patch_size}x{vit_config.patch_size}")
    print(f"Number of patches: {(vit_config.image_size // vit_config.patch_size) ** 2}")
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
    
    print(f"{'Layer Type':<20} {'Count':>8} {'Parameters':>15} {'Percentage':>12}")
    print("-" * 80)
    
    for module_type, info in sorted_types:
        percentage = (info['params'] / total_params) * 100
        print(f"{module_type:<20} {info['count']:>8} {info['params']:>15,} {percentage:>11.2f}%")
    
    print("-" * 80)
    print(f"{'Total':<20} {sum(t[1]['count'] for t in sorted_types):>8} {total_params:>15,} {'100.00%':>12}")
    print()
    
    # Verify total parameters from type counts
    counted_params = sum(t[1]['params'] for t in sorted_types)
    if counted_params != total_params:
        print(f"⚠️  Warning: Counted parameters ({counted_params:,}) != Total parameters ({total_params:,})")
        print(f"   Difference: {total_params - counted_params:,} parameters unaccounted for")
        print()
    
    # Detailed breakdown for all layer types
    print("=" * 80)
    print("Detailed Layer Information by Type")
    print("=" * 80)
    print()
    
    for module_type, info in sorted_types:
        print(f"{module_type} Layers ({info['count']} layers, {info['params']:,} params):")
        print("-" * 80)
        
        # Sort by parameters
        sorted_modules = sorted(info['modules'], key=lambda x: x[1], reverse=True)
        
        # For Linear layers, show only top 20 (too many otherwise)
        if module_type == 'Linear' and len(sorted_modules) > 20:
            display_modules = sorted_modules[:20]
            print(f"Top 20 {module_type} layers by parameter count:")
            print()
        else:
            display_modules = sorted_modules
        
        for idx, (name, params) in enumerate(display_modules, 1):
            percentage = (params / total_params) * 100
            print(f"  {idx:2d}. {name:<60} {params:>10,} ({percentage:>5.2f}%)")
        
        if module_type == 'Linear' and len(sorted_modules) > 20:
            print()
            print(f"  ... and {len(sorted_modules) - 20} more {module_type} layers")
        
        print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
