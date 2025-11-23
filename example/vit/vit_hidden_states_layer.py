"""
Inspect and display all hidden_states layers in ViT model.

This script loads a ViT model and outputs information about all layers that produce
hidden_states, including their layer IDs, names, types, and basic properties.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import ViTConfig, ViTModel

# Add EdgeRazor to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))


def get_layer_info(module, layer_name="", depth=0):
    """
    Recursively get information about all layers in the model.
    
    Args:
        module: PyTorch module to inspect
        layer_name: Name/path of the current layer
        depth: Current depth in the module hierarchy
        
    Returns:
        List of tuples containing (layer_path, module_type, module_info)
    """
    layers_info = []
    
    # Get basic module info
    module_type = type(module).__name__
    
    # Get parameter count
    param_count = sum(p.numel() for p in module.parameters(recurse=False))
    
    # Get input/output shapes if available
    info = {
        'type': module_type,
        'params': param_count,
        'depth': depth
    }
    
    # Add specific info for common layer types
    if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
        info['shape'] = f"({module.in_features}, {module.out_features})"
    elif hasattr(module, 'hidden_size'):
        info['hidden_size'] = module.hidden_size
    elif hasattr(module, 'num_attention_heads'):
        info['num_heads'] = module.num_attention_heads
    
    # Add this layer's info
    if layer_name:
        layers_info.append((layer_name, info))
    
    # Recursively process children
    for name, child in module.named_children():
        child_name = f"{layer_name}.{name}" if layer_name else name
        layers_info.extend(get_layer_info(child, child_name, depth + 1))
    
    return layers_info


def load_vit_config(config_path):
    """
    Load ViT configuration from JSON file.
    Same function as in train.py to ensure consistency.
    """
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


def analyze_vit_hidden_states():
    """Analyze and display ViT hidden_states layer information."""
    
    print("=" * 100)
    print("ViT Model Hidden States Layer Analysis")
    print("=" * 100)
    
    # Load ViT configuration from JSON file (same as train.py)
    config_path = Path(__file__).parent / "src" / "ViT-S-16.json"
    
    if not config_path.exists():
        print(f"\nError: Configuration file not found: {config_path}")
        print("Please run this script from the vit example directory.")
        return
    
    print(f"\nLoading configuration from: {config_path}")
    config = load_vit_config(config_path)
    
    print("\nModel Configuration:")
    print(f"  Image Size: {config.image_size}x{config.image_size}")
    print(f"  Patch Size: {config.patch_size}x{config.patch_size}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Number of Transformer Layers: {config.num_hidden_layers}")
    print(f"  Number of Attention Heads: {config.num_attention_heads}")
    print(f"  Intermediate Size: {config.intermediate_size}")
    
    # Create model
    model = ViTModel(config)
    model.eval()
    
    print("\n" + "=" * 100)
    print("Testing Hidden States Output")
    print("=" * 100)
    
    # Create dummy input
    batch_size = 2
    dummy_input = torch.randn(batch_size, config.num_channels, config.image_size, config.image_size)
    
    # Forward pass with output_hidden_states=True
    with torch.no_grad():
        outputs = model(pixel_values=dummy_input, output_hidden_states=True)
    
    # Analyze hidden_states
    hidden_states = outputs.hidden_states
    
    print(f"\nTotal number of hidden_states layers: {len(hidden_states)}")
    print(f"  - Layer 0: Embeddings output")
    print(f"  - Layers 1-{len(hidden_states)-1}: Transformer block outputs")
    
    print("\n" + "=" * 100)
    print("Hidden States Layer Details")
    print("=" * 100)
    print(f"\n{'Layer ID':<10} {'Layer Name':<30} {'Shape':<30} {'Notes':<40}")
    print("-" * 110)
    
    for layer_id, hidden_state in enumerate(hidden_states):
        shape_str = f"{tuple(hidden_state.shape)}"
        
        if layer_id == 0:
            layer_name = "embeddings"
            notes = "After patch embedding + position embedding"
        else:
            layer_name = f"encoder.layer.{layer_id - 1}"
            notes = f"After transformer block {layer_id - 1}"
        
        print(f"{layer_id:<10} {layer_name:<30} {shape_str:<30} {notes:<40}")
    
    # Detailed information about key layers
    print("\n" + "=" * 100)
    print("Detailed Layer Information")
    print("=" * 100)
    
    # Embeddings
    print("\n[Layer 0] Embeddings:")
    print(f"  Output Shape: {hidden_states[0].shape}")
    print(f"  Sequence Length: {hidden_states[0].shape[1]} (includes CLS token)")
    print(f"  Hidden Dimension: {hidden_states[0].shape[2]}")
    print(f"  Components:")
    print(f"    - Patch Embeddings: Converts image patches to embeddings")
    print(f"    - CLS Token: Added at the beginning of sequence")
    print(f"    - Position Embeddings: Added to all tokens")
    
    # Transformer blocks
    print(f"\n[Layers 1-{len(hidden_states)-1}] Transformer Blocks:")
    for layer_id in range(1, len(hidden_states)):
        print(f"\n  Layer {layer_id} (encoder.layer.{layer_id - 1}):")
        print(f"    Output Shape: {hidden_states[layer_id].shape}")
        print(f"    Components:")
        print(f"      - Multi-Head Self-Attention")
        print(f"      - Layer Normalization")
        print(f"      - Feed-Forward Network (MLP)")
        print(f"      - Residual Connections")
    
    # Show model structure for reference
    print("\n" + "=" * 100)
    print("Complete Model Structure (Key Layers)")
    print("=" * 100)
    
    all_layers = get_layer_info(model)
    
    # Filter for important layers
    important_patterns = ['embeddings', 'encoder.layer', 'attention', 'intermediate', 'output', 'layernorm']
    
    print(f"\n{'Layer Path':<60} {'Type':<25} {'Parameters':<15}")
    print("-" * 100)
    
    for layer_path, info in all_layers:
        # Check if this is an important layer
        if any(pattern in layer_path.lower() for pattern in important_patterns):
            # Only show leaf layers with parameters or specific types
            if info['params'] > 0 or info['type'] in ['LayerNorm', 'Dropout']:
                params_str = f"{info['params']:,}" if info['params'] > 0 else "-"
                # Truncate long paths
                display_path = layer_path if len(layer_path) <= 58 else "..." + layer_path[-55:]
                print(f"{display_path:<60} {info['type']:<25} {params_str:<15}")
    
    # Summary for using hidden_states in knowledge distillation
    print("\n" + "=" * 100)
    print("Usage in Knowledge Distillation")
    print("=" * 100)
    
    print("\nTo use specific hidden_states layers in KD configuration:")
    print("\n1. Single layer (e.g., last layer):")
    print("   layer_index: -1  # or layer_index: 12 for a 12-layer model")
    
    print("\n2. Multiple layers (e.g., embeddings, early, middle, and last):")
    print("   layer_index: [0, 4, 8, 12]")
    
    print("\n3. All layers (default):")
    print("   # Don't specify layer_index, or set to null")
    
    print("\nExample KD configuration with layer selection:")
    print("""
kd_configuration:
  method: KD
  losses:
    - loss_type: hidden_states
      layer_index: [0, 4, 8, 12]  # Select embeddings, early, middle, and last layers
      alpha: 0.5
      distill_function: fd
      distill_function_config:
        reduction: mean
    """)
    
    # Show actual layer indices available
    print("\n" + "=" * 100)
    print("Available Layer Indices for this Model")
    print("=" * 100)
    print(f"\nValid layer_index values: 0 to {len(hidden_states) - 1}")
    print(f"Valid negative indices: -{len(hidden_states)} to -1")
    print(f"\nLayer mapping:")
    for i in range(len(hidden_states)):
        negative_idx = i - len(hidden_states)
        if i == 0:
            desc = "Embeddings output"
        else:
            desc = f"After transformer block {i-1}"
        print(f"  {i:2d} (or {negative_idx:2d}): {desc}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    analyze_vit_hidden_states()
