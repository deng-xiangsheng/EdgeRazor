"""
Calculate quantization proportion for ViT-S/16 with QAT.

Usage: python quant_prop.py [--quant_config CONFIG.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import torch.nn as nn
from transformers import ViTConfig, ViTModel

# Add EdgeRazor to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from edgerazor.qat import QAT


def load_vit_config(config_path):
    """Load ViT configuration from JSON file. Same as train.py"""
    with open(config_path) as f:
        config_data = json.load(f)

    vision_cfg = config_data.get("vision_cfg", {})
    embed_dim = config_data.get("embed_dim", 384)

    config = ViTConfig(
        hidden_size=embed_dim,
        num_hidden_layers=vision_cfg.get("layers", 12),
        num_attention_heads=embed_dim // 64,  # Typically hidden_size/64
        intermediate_size=embed_dim * 4,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        image_size=vision_cfg.get("image_size", 224),
        patch_size=vision_cfg.get("patch_size", 16),
        num_channels=3,
        num_labels=10,  # MNIST classes
    )

    return config


class ViTForMNIST(nn.Module):
    """ViT model for MNIST."""
    def __init__(self, config):
        super().__init__()
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values):
        return self.classifier(self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :])


def main():
    parser = argparse.ArgumentParser(description="Calculate quantization proportion for ViT-S/16")
    parser.add_argument("--quant_config", type=str, default="q_vit_w1.58_a16.yaml",
                        help="Path to quantization YAML file")
    args = parser.parse_args()
    
    # Load model config and create model
    config_path = Path("src/ViT-S-16.json")
    if not config_path.exists():
        print(f"Error: {config_path} not found. Run from vit project directory.")
        sys.exit(1)
    
    vit_config = load_vit_config(config_path)
    model = ViTForMNIST(vit_config)
    
    print("=" * 80)
    print("ViT-S/16 Quantization Proportion Analysis")
    print("=" * 80)
    print(f"Model: ViT-S/16 (hidden_size={vit_config.hidden_size}, layers={vit_config.num_hidden_layers})")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Apply QAT quantization
    quant_config_path = Path(args.quant_config)
    if not quant_config_path.exists():
        print(f"Error: {quant_config_path} not found")
        sys.exit(1)
    
    print(f"Applying QAT with config: {quant_config_path}")
    qat = QAT(config=str(quant_config_path))
    model = qat.quantize(model)
    
    # Count number of blocks for each linear layer
    print()
    print("=" * 80)
    print("Number of Blocks per Linear Layer")
    print("=" * 80)
    print(f"{'Layer Name':<60} {'Weight Shape':<20} {'Num Blocks':<15}")
    print("-" * 80)
    
    from edgerazor.qat.module import QLinear
    
    total_blocks = 0
    layer_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, QLinear):
            weight_shape = module.weight.shape
            # Calculate number of blocks: (out_features, in_features // block_size)
            in_features = weight_shape[1]
            block_size = module.w_block_size if hasattr(module, 'w_block_size') else 128
            num_blocks = weight_shape[0] * ((in_features + block_size - 1) // block_size)
            
            print(f"{name:<60} {str(weight_shape):<20} {num_blocks:<15,}")
            total_blocks += num_blocks
            layer_count += 1
    
    print("-" * 80)
    print(f"{'Total':<60} {'':<20} {total_blocks:<15,}")
    print(f"\nTotal quantized linear layers: {layer_count}")
    print(f"Average blocks per layer: {total_blocks / layer_count if layer_count > 0 else 0:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
