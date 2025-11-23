"""
Test trained ViT models on MNIST and generate performance table.

This script evaluates all trained models (best_model.pth) from a log directory
and generates a comprehensive performance table with metrics.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import ViTConfig, ViTModel

# Add EdgeRazor to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "src"))

# Import after adding to path
from edgerazor import EdgeRazor  # noqa: E402
from edgerazor.log import get_logger  # noqa: E402


class ViTForMNIST(nn.Module):
    """ViT model for MNIST classification."""

    def __init__(self, config):
        super().__init__()
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, pixel_values, labels=None, output_hidden_states=False, output_attentions=False, return_dict=True):
        """Forward pass."""
        vit_outputs = self.vit(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        cls_output = vit_outputs.last_hidden_state[:, 0]
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': vit_outputs.hidden_states if output_hidden_states else None,
                'attentions': vit_outputs.attentions if output_attentions else None,
            }
        return logits


def load_vit_config(config_path):
    """Load ViT configuration from JSON file."""
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


class GrayscaleToRGB:
    """Convert grayscale image to RGB by repeating channels."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)


def prepare_test_dataloader(data_root, batch_size, num_workers=4):
    """Prepare MNIST test dataloader."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        GrayscaleToRGB(),
    ])

    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return test_loader


def evaluate_model(model, test_loader, device, logger):
    """Evaluate model on test set."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, return_dict=True)
            logits = outputs['logits']

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': correct,
        'total': total
    }


def parse_config_name(config_name):
    """
    Parse configuration name to extract quantization info.
    
    Examples:
        fp_vit_w16_a16 -> {'w': 16, 'a': 16, 'distill': False, 'mp': False}
        q_vit_w4_a8 -> {'w': 4, 'a': 8, 'distill': False, 'mp': False}
        q_vit_w1.58_a16 -> {'w': 1.58, 'a': 16, 'distill': False, 'mp': False}
        q_vit_w1.58mp4_a16_kldr_fd -> {'w': '1.58+4', 'a': 16, 'distill': True, 'mp': True}
    """
    info = {
        'w': 16,
        'a': 16,
        'distill': False,
        'mp': False
    }
    
    # Check if distillation is used
    if 'kldr' in config_name or 'kldf' in config_name or 'fd' in config_name:
        info['distill'] = True
    
    # Extract weight bits
    if 'w1.58mp4' in config_name:
        info['w'] = '1.58+4'
        info['mp'] = True
    elif 'w1.58' in config_name:
        info['w'] = 1.58
    elif 'w4' in config_name:
        info['w'] = 4
    elif 'w8' in config_name:
        info['w'] = 8
    elif 'w16' in config_name or 'fp_' in config_name:
        info['w'] = 16
    
    # Extract activation bits
    if 'a4' in config_name:
        info['a'] = 4
    elif 'a8' in config_name:
        info['a'] = 8
    elif 'a16' in config_name or 'fp_' in config_name:
        info['a'] = 16
    
    return info


def calculate_compression(w_bits, total_params=21817354):
    """
    Calculate compression ratio and quantized proportion.
    
    Args:
        w_bits: Weight bit-width (can be float like 1.58 or string like '1.58+4')
        total_params: Total number of parameters
    
    Returns:
        tuple: (proportion, compression_ratio)
    """
    # Assume 98.02% parameters are quantized (based on README)
    quantized_proportion = 0.9802
    
    if w_bits == 16:
        # Full precision
        return 0.0, 1.0
    
    # Parse mixed precision
    if isinstance(w_bits, str) and '+' in w_bits:
        # Mixed precision: assume 50-50 split for simplicity
        # Actual calculation would need layer-by-layer info
        w_effective = 2.79  # Approximate for 1.58+4 mix
        quantized_proportion = 0.9802
    elif w_bits == 1.58:
        w_effective = 1.58
    elif w_bits == 4:
        w_effective = 4
    elif w_bits == 8:
        w_effective = 8
    else:
        w_effective = w_bits
    
    # Compression = 16 / ((1-prop)*16 + prop*w_effective)
    avg_bits = (1 - quantized_proportion) * 16 + quantized_proportion * w_effective
    compression_ratio = 16.0 / avg_bits
    
    return quantized_proportion, compression_ratio


def load_checkpoint_info(checkpoint_dir):
    """Load checkpoint information from directory."""
    checkpoint_path = checkpoint_dir / "best_model.pth"
    
    if not checkpoint_path.exists():
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract metadata
        info = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'accuracy': checkpoint.get('best_acc', 0.0),
            'checkpoint_path': checkpoint_path
        }
        
        # Load args if available
        if 'args' in checkpoint:
            args = checkpoint['args']
            info['config_files'] = {
                'model_config': str(args.get('model_config', '')),
                'quant_config': str(args.get('quant_config', '')) if args.get('quant_config') else None,
                'kd_config': str(args.get('kd_config', '')) if args.get('kd_config') else None,
                'edgerazor_config': str(args.get('edgerazor_config', '')) if args.get('edgerazor_config') else None,
            }
        
        return info
    except Exception as e:
        print(f"Error loading checkpoint from {checkpoint_dir}: {e}")
        return None


def test_all_models(logdir, model_config_path, data_root, device, batch_size=256):
    """Test all models in the log directory."""
    logger = get_logger("ViT-Test")
    
    logdir = Path(logdir)
    checkpoint_dir = logdir / "checkpoints"
    
    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Find all model directories
    model_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()])
    
    if not model_dirs:
        logger.error(f"No model directories found in {checkpoint_dir}")
        return
    
    logger.info(f"Found {len(model_dirs)} models to evaluate")
    logger.info("")
    
    # Load ViT configuration
    vit_config = load_vit_config(model_config_path)
    total_params = sum(p.numel() for p in ViTForMNIST(vit_config).parameters())
    
    # Prepare test dataloader
    test_loader = prepare_test_dataloader(data_root, batch_size)
    logger.info(f"Test samples: {len(test_loader.dataset):,}")
    logger.info("")
    
    # Results storage
    results = []
    
    # Evaluate each model
    for model_dir in model_dirs:
        config_name = model_dir.name
        logger.info("=" * 80)
        logger.info(f"Evaluating: {config_name}")
        logger.info("=" * 80)
        
        # Load checkpoint info
        ckpt_info = load_checkpoint_info(model_dir)
        if ckpt_info is None:
            logger.warning(f"Skipping {config_name}: checkpoint not found or invalid")
            logger.info("")
            continue
        
        # Parse configuration
        config_info = parse_config_name(config_name)
        proportion, compression = calculate_compression(config_info['w'], total_params)
        
        try:
            # Create model
            model = ViTForMNIST(vit_config)
            
            # Apply quantization if needed (BEFORE loading weights)
            edgerazor = None
            if config_info['w'] != 16 or config_info['a'] != 16:
                # Need to apply quantization
                config_files = ckpt_info.get('config_files', {})
                
                if config_files.get('edgerazor_config'):
                    edgerazor_config = Path(config_files['edgerazor_config'])
                    if edgerazor_config.exists():
                        logger.info(f"Applying EdgeRazor unified config: {edgerazor_config.name}")
                        edgerazor = EdgeRazor(config=edgerazor_config)
                        if edgerazor.is_qat_enabled:
                            model = edgerazor.quantize(model)
                            logger.info("✓ QAT applied")
                elif config_files.get('quant_config'):
                    quant_config = Path(config_files['quant_config'])
                    if quant_config.exists():
                        logger.info(f"Applying quantization config: {quant_config.name}")
                        edgerazor = EdgeRazor(qat_config=quant_config)
                        model = edgerazor.quantize(model)
                        logger.info("✓ QAT applied")
            
            # Load checkpoint and extract weights
            checkpoint = torch.load(ckpt_info['checkpoint_path'], map_location='cpu', weights_only=False)
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            logger.info("✓ Model weights loaded")
            
            # Move model to device
            model = model.to(device)
            
            # Evaluate
            metrics = evaluate_model(model, test_loader, device, logger)
            
            # Store results
            result = {
                'config_name': config_name,
                'epoch': ckpt_info['epoch'],
                'w_bits': config_info['w'],
                'a_bits': config_info['a'],
                'distill': config_info['distill'],
                'mp': config_info['mp'],
                'params': total_params,
                'proportion': proportion,
                'compression': compression,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
            }
            results.append(result)
            
            # Log results
            logger.info(f"Epoch: {ckpt_info['epoch']}")
            logger.info(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall:    {metrics['recall']:.4f}")
            logger.info(f"F1 Score:  {metrics['f1']:.4f}")
            logger.info("")
            
        except Exception as e:
            logger.error(f"Error evaluating {config_name}: {e}")
            import traceback
            traceback.print_exc()
            logger.info("")
            continue
    
    # Generate table
    print_results_table(results, logger)
    
    # Save results to JSON
    results_file = logdir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")


def print_results_table(results, logger):
    """Print results in markdown table format."""
    logger.info("=" * 80)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 80)
    logger.info("")
    
    # Table header
    header = "| Model    | Step | W      | A    | Prop   | Comp  | Accuracy | Recall | Precision | F1     |"
    separator = "| :------- | ---- | :----- | :--- | :----- | :---- | :------- | :----- | :-------- | :----- |"
    
    logger.info(header)
    logger.info(separator)
    
    # Sort results by configuration
    results_sorted = sorted(results, key=lambda x: (
        0 if x['w_bits'] == 16 else 1,
        x['w_bits'] if isinstance(x['w_bits'], (int, float)) else 99,
        x['a_bits'],
        not x['distill']
    ))
    
    # Table rows
    for r in results_sorted:
        w_str = str(r['w_bits'])
        a_str = str(r['a_bits'])
        prop_str = f"{r['proportion']*100:.2f}%" if r['proportion'] > 0 else "0%"
        comp_str = f"{r['compression']:.2f}x"

        row = (
            f"| ViT-S/16 | {r['epoch']:<4} | {w_str:<6} | {a_str:<4} | "
            f"{prop_str:<6} | {comp_str:<5} | "
            f"{r['accuracy']*100:>7.2f}% | {r['recall']*100:>6.2f}% | "
            f"{r['precision']*100:>8.2f}% | {r['f1']*100:>6.2f}% |"
        )
        logger.info(row)
    
    logger.info("")
    logger.info("=" * 80)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test trained ViT models and generate performance table"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=True,
        help="Log directory containing checkpoints (e.g., ./runs1 or ./runs2)"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="src/ViT-S-16.json",
        help="Path to ViT model configuration"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Root directory for MNIST dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for evaluation"
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Run tests
    test_all_models(
        logdir=args.logdir,
        model_config_path=args.model_config,
        data_root=args.data_root,
        device=device,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
