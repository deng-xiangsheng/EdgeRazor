"""
Analyze ViT model outputs structure

This script creates a ViT model from JSON configuration and analyzes all output fields:
- loss: Task-specific loss value
- logits: Classification predictions
- hidden_states: All layer representations
- attentions: Attention weights (if requested)

Each output shape is explained in detail.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


class ViTForMNIST(nn.Module):
    """ViT model for MNIST classification."""

    def __init__(self, config):
        super().__init__()
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, pixel_values, labels=None, output_hidden_states=False, output_attentions=False):
        """
        Forward pass with optional outputs.
        
        Args:
            pixel_values: Input images, shape (batch_size, channels, height, width)
            labels: Ground truth labels, shape (batch_size,)
            output_hidden_states: Whether to return hidden states from all layers
            output_attentions: Whether to return attention weights from all layers
        
        Returns:
            dict with keys:
                - loss (optional): CrossEntropyLoss if labels provided
                - logits: Classification predictions
                - hidden_states (optional): Tuple of hidden states from all layers
                - attentions (optional): Tuple of attention weights from all layers
        """
        # Get ViT outputs
        vit_outputs = self.vit(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        # Classification logits using CLS token (first token)
        # vit_outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
        # We take [:, 0, :] to get CLS token representation
        cls_output = vit_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        # Prepare output dictionary
        outputs = {
            'logits': logits,
        }
        
        if loss is not None:
            outputs['loss'] = loss
        
        if output_hidden_states:
            outputs['hidden_states'] = vit_outputs.hidden_states
        
        if output_attentions:
            outputs['attentions'] = vit_outputs.attentions
        
        return outputs


def load_vit_config(config_path):
    """
    Load ViT configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        ViTConfig: Transformers ViT configuration object
    """
    with open(config_path) as f:
        config_data = json.load(f)
    
    vision_cfg = config_data.get("vision_cfg", {})
    embed_dim = config_data.get("embed_dim", 384)
    
    config = ViTConfig(
        hidden_size=embed_dim,
        num_hidden_layers=vision_cfg.get("layers", 12),
        num_attention_heads=embed_dim // 64,
        intermediate_size=embed_dim * 4,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        image_size=vision_cfg.get("image_size", 224),
        patch_size=vision_cfg.get("patch_size", 16),
        num_channels=3,
        num_labels=10,
    )
    
    return config


def analyze_output_shapes(outputs, batch_size, seq_len, hidden_size, num_layers, num_heads, num_labels):
    """
    Analyze and explain output shapes.
    
    Args:
        outputs: Model output dictionary
        batch_size: Batch size used
        seq_len: Sequence length (number of patches + CLS token)
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_labels: Number of classification labels
    """
    print("=" * 80)
    print("ViT Model Outputs Analysis")
    print("=" * 80)
    print()
    
    # Analyze loss
    if 'loss' in outputs:
        loss = outputs['loss']
        print("1. Loss")
        print("-" * 80)
        print(f"   Shape: {loss.shape}")
        print(f"   Type: torch.Tensor (scalar)")
        print(f"   Value: {loss.item():.6f}")
        print()
        print("   Meaning:")
        print("   - Task-specific loss (CrossEntropyLoss for classification)")
        print("   - Computed as: -log(P(correct_class))")
        print("   - Lower is better")
        print()
    
    # Analyze logits
    if 'logits' in outputs:
        logits = outputs['logits']
        print("2. Logits")
        print("-" * 80)
        print(f"   Shape: {logits.shape}")
        print(f"   Expected: torch.Size([{batch_size}, {num_labels}])")
        print()
        print("   Meaning:")
        print(f"   - Classification scores for {num_labels} classes")
        print(f"   - Dimension 0: batch_size = {batch_size}")
        print(f"   - Dimension 1: num_labels = {num_labels} (MNIST digits 0-9)")
        print()
        print("   Usage:")
        print("   - Apply softmax to get probabilities: torch.softmax(logits, dim=-1)")
        print("   - Get predictions: torch.argmax(logits, dim=-1)")
        print()
        print(f"   Sample logits (batch 0): {logits[0].tolist()}")
        print(f"   Predicted class: {torch.argmax(logits[0]).item()}")
        print()
    
    # Analyze hidden_states
    if 'hidden_states' in outputs:
        hidden_states = outputs['hidden_states']
        print("3. Hidden States")
        print("-" * 80)
        print(f"   Type: tuple of {len(hidden_states)} tensors")
        print(f"   Expected: {num_layers + 1} layers (1 embeddings + {num_layers} transformer)")
        print()
        print("   Structure:")
        print(f"   - Layer 0 (Embeddings): shape {hidden_states[0].shape}")
        print(f"     * Input image patches converted to embeddings")
        print(f"     * Includes CLS token and position embeddings")
        print()
        
        for i in range(1, min(4, len(hidden_states))):
            print(f"   - Layer {i} (Transformer Block {i}): shape {hidden_states[i].shape}")
        
        if len(hidden_states) > 4:
            print(f"   - ... ({len(hidden_states) - 4} more layers)")
            print(f"   - Layer {len(hidden_states) - 1} (Final): shape {hidden_states[-1].shape}")
        print()
        
        print("   Shape Meaning:")
        print(f"   - Dimension 0: batch_size = {batch_size}")
        print(f"   - Dimension 1: seq_len = {seq_len}")
        print(f"     * 1 CLS token + {seq_len - 1} image patches")
        print(f"     * Number of patches = (image_size / patch_size)^2")
        print(f"     * For 224x224 image with 16x16 patches: (224/16)^2 = 196 patches")
        print(f"   - Dimension 2: hidden_size = {hidden_size}")
        print()
        
        print("   Usage for Knowledge Distillation:")
        print("   - Layer 0: Embedding layer (often used for early-stage matching)")
        print(f"   - Layer {num_layers // 2}: Middle layer (often used for mid-level features)")
        print(f"   - Layer {num_layers}: Final layer (often used for high-level features)")
        print()
        print("   Example layer selection for KD:")
        print(f"   - 'low' (layer 1): {hidden_states[1].shape}")
        print(f"   - 'mid' (layer {num_layers // 2}): {hidden_states[num_layers // 2].shape}")
        print(f"   - 'high' (layer {num_layers}): {hidden_states[num_layers].shape}")
        print()
    
    # Analyze attentions
    if 'attentions' in outputs:
        attentions = outputs['attentions']
        print("4. Attentions")
        print("-" * 80)
        print(f"   Type: tuple of {len(attentions)} tensors")
        print(f"   Expected: {num_layers} layers (one per transformer block)")
        print()
        print("   Structure:")
        for i in range(min(3, len(attentions))):
            print(f"   - Layer {i + 1}: shape {attentions[i].shape}")
        
        if len(attentions) > 3:
            print(f"   - ... ({len(attentions) - 3} more layers)")
            print(f"   - Layer {len(attentions)}: shape {attentions[-1].shape}")
        print()
        
        print("   Shape Meaning:")
        print(f"   - Dimension 0: batch_size = {batch_size}")
        print(f"   - Dimension 1: num_heads = {num_heads}")
        print(f"   - Dimension 2: query_seq_len = {seq_len}")
        print(f"   - Dimension 3: key_seq_len = {seq_len}")
        print()
        print("   Meaning:")
        print("   - Attention weights showing how each token attends to other tokens")
        print("   - Values sum to 1.0 along the last dimension (softmax applied)")
        print("   - Can visualize attention patterns for interpretability")
        print()
        print("   Usage:")
        print("   - Analyze which patches the model focuses on")
        print("   - Attention distillation in knowledge distillation")
        print("   - Model interpretability and visualization")
        print()


def main():
    """Main function to analyze ViT outputs."""
    
    # Configuration
    config_path = Path(__file__).parent / "src" / "ViT-S-16.json"
    
    print("=" * 80)
    print("Creating ViT Model from JSON Configuration")
    print("=" * 80)
    print()
    
    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_vit_config(config_path)
    
    print()
    print("Model Configuration:")
    print(f"  Image Size: {config.image_size}x{config.image_size}")
    print(f"  Patch Size: {config.patch_size}x{config.patch_size}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Number of Transformer Layers: {config.num_hidden_layers}")
    print(f"  Number of Attention Heads: {config.num_attention_heads}")
    print(f"  Intermediate Size: {config.intermediate_size}")
    print(f"  Number of Labels: {config.num_labels}")
    print()
    
    # Calculate sequence length
    num_patches = (config.image_size // config.patch_size) ** 2
    seq_len = num_patches + 1  # +1 for CLS token
    print(f"Calculated Sequence Length:")
    print(f"  Number of patches: {num_patches} = ({config.image_size}/{config.patch_size})^2")
    print(f"  Sequence length: {seq_len} = {num_patches} patches + 1 CLS token")
    print()
    
    # Create model
    print("Creating model...")
    model = ViTForMNIST(config)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Create dummy input
    batch_size = 2
    # Convert grayscale MNIST to 3-channel by repeating
    dummy_input = torch.randn(batch_size, 3, config.image_size, config.image_size)
    dummy_labels = torch.randint(0, config.num_labels, (batch_size,))
    
    print("=" * 80)
    print("Running Forward Pass")
    print("=" * 80)
    print()
    print(f"Input shape: {dummy_input.shape}")
    print(f"  - Dimension 0: batch_size = {batch_size}")
    print(f"  - Dimension 1: channels = 3 (RGB or repeated grayscale)")
    print(f"  - Dimension 2-3: height x width = {config.image_size}x{config.image_size}")
    print()
    print(f"Labels shape: {dummy_labels.shape}")
    print(f"  - Dimension 0: batch_size = {batch_size}")
    print()
    
    # Run forward pass with all outputs
    print("Computing model outputs...")
    with torch.no_grad():
        outputs = model(
            pixel_values=dummy_input,
            labels=dummy_labels,
            output_hidden_states=True,
            output_attentions=True
        )
    print()
    
    # Analyze outputs
    analyze_output_shapes(
        outputs=outputs,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=config.hidden_size,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        num_labels=config.num_labels
    )
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("Output Fields Available:")
    for key in outputs.keys():
        if key == 'loss':
            print(f"  ✓ {key}: scalar tensor")
        elif key == 'logits':
            print(f"  ✓ {key}: shape {outputs[key].shape}")
        elif key == 'hidden_states':
            print(f"  ✓ {key}: tuple of {len(outputs[key])} tensors")
        elif key == 'attentions':
            print(f"  ✓ {key}: tuple of {len(outputs[key])} tensors")
    print()
    
    print("Key Takeaways:")
    print("  1. loss: Task loss (CrossEntropyLoss), used for training")
    print(f"  2. logits: Classification scores [{batch_size}, {config.num_labels}]")
    print(f"  3. hidden_states: {config.num_hidden_layers + 1} layers of [{batch_size}, {seq_len}, {config.hidden_size}]")
    print(f"  4. attentions: {config.num_hidden_layers} layers of [{batch_size}, {config.num_attention_heads}, {seq_len}, {seq_len}]")
    print()
    
    print("For Knowledge Distillation:")
    print("  - Use 'logits' for logits-based distillation (KLD)")
    print("  - Use 'hidden_states' for feature-based distillation (FD)")
    print("  - Use 'attentions' for attention-based distillation")
    print("  - String layer selection: 'low', 'mid', 'high'")
    print(f"    * 'low' → layer 1")
    print(f"    * 'mid' → layer {config.num_hidden_layers // 2}")
    print(f"    * 'high' → layer {config.num_hidden_layers}")
    print()
    
    print("=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
