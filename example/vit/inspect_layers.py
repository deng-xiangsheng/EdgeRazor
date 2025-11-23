"""
Inspect specific layers in ViT-S/16 model.

This script shows the detailed information about specific layers.
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
    print("ViT-S/16 Layer Inspection")
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
    print("Model:")
    print(model)
    
    # Inspect specific layers
    layers_to_inspect = [
        "vit.embeddings.patch_embeddings.projection",
        "vit.embeddings.position_embeddings",
        "vit.embeddings.cls_token",
        "vit.encoder.layer.0.attention.attention.query",
        "vit.encoder.layer.0.attention.attention.key",
        "vit.encoder.layer.0.attention.attention.value",
        "vit.encoder.layer.0.attention.output.dense",
        "vit.encoder.layer.0.intermediate.dense",
        "vit.encoder.layer.0.output.dense",
        "classifier",
    ]
    
    for layer_name in layers_to_inspect:
        print(f"Layer: {layer_name}")
        print("-" * 80)
        
        try:
            layer = inspect_layer(model, layer_name)
            
            print(f"  Class: {type(layer).__name__}")
            print(f"  Module: {type(layer).__module__}")
            print(f"  Details: {layer}")
            
            # Count parameters
            if isinstance(layer, nn.Parameter):
                params = layer.numel()
                print(f"  Shape: {layer.shape}")
            else:
                params = sum(p.numel() for p in layer.parameters())
            print(f"  Parameters: {params:,}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    # Show the structure of first encoder layer
    print("=" * 80)
    print("First Encoder Layer Structure (vit.encoder.layer.0)")
    print("=" * 80)
    print()
    print(model.vit.encoder.layer[0])
    print()
    
    # Show attention structure
    print("=" * 80)
    print("Attention Structure (vit.encoder.layer.0.attention)")
    print("=" * 80)
    print()
    print(model.vit.encoder.layer[0].attention)
    print()
    
    # Show embeddings structure
    print("=" * 80)
    print("Embeddings Structure (vit.embeddings)")
    print("=" * 80)
    print()
    print(model.vit.embeddings)
    print()
    
    # List all Linear layers
    print("=" * 80)
    print("All Linear Layers in ViT-S/16 (First 30)")
    print("=" * 80)
    print()
    
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append((name, module))
    
    for idx, (name, module) in enumerate(linear_layers[:30], 1):
        params = sum(p.numel() for p in module.parameters())
        print(f"{idx:2d}. {name:<70} {params:>10,} params")
        print(f"    {module}")
        print()
    
    if len(linear_layers) > 30:
        print(f"... and {len(linear_layers) - 30} more Linear layers")
        print()
    
    print(f"Total Linear layers: {len(linear_layers)}")
    print()
    
    # Show parameter shapes for attention
    print("=" * 80)
    print("Attention QKV Parameter Shapes (Layer 0)")
    print("=" * 80)
    print()
    
    attn = model.vit.encoder.layer[0].attention.attention
    print(f"Query weight shape:  {attn.query.weight.shape}")
    print(f"Query bias shape:    {attn.query.bias.shape}")
    print(f"Key weight shape:    {attn.key.weight.shape}")
    print(f"Key bias shape:      {attn.key.bias.shape}")
    print(f"Value weight shape:  {attn.value.weight.shape}")
    print(f"Value bias shape:    {attn.value.bias.shape}")
    print()
    print(f"Hidden size:         {vit_config.hidden_size}")
    print(f"Attention heads:     {vit_config.num_attention_heads}")
    print(f"Head dimension:      {vit_config.hidden_size // vit_config.num_attention_heads}")
    print()


if __name__ == "__main__":
    main()
