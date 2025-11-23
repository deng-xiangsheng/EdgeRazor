"""
Train ViT on MNIST with optional Quantization Aware Training (QAT).

This script supports both full-precision and quantized training using EdgeRazor QAT framework.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm
from transformers import ViTConfig, ViTModel

# Add EdgeRazor to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "src"))

# Import after adding to path
from edgerazor import EdgeRazor  # noqa: E402
from edgerazor.log import get_logger  # noqa: E402

from .arg import get_run_name, parse_args  # noqa: E402


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
        )
        self.current_step = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate."""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.base_lrs[i] * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            self.cosine_scheduler.step()

    def get_lr(self):
        """Get current learning rate."""
        return [group["lr"] for group in self.optimizer.param_groups]


class ViTForMNIST(nn.Module):
    """ViT model for MNIST classification."""

    def __init__(self, config):
        super().__init__()
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, pixel_values, labels=None, output_hidden_states=False, output_attentions=False, return_dict=True):
        """
        Forward pass with optional outputs for knowledge distillation.
        
        Args:
            pixel_values: Input images, shape (batch_size, channels, height, width)
            labels: Ground truth labels, shape (batch_size,)
            output_hidden_states: Whether to return hidden states from all layers
            output_attentions: Whether to return attention weights from all layers
            return_dict: Whether to return a dictionary (True) or just logits (False)
        
        Returns:
            If return_dict=True:
                dict with keys:
                    - loss (optional): CrossEntropyLoss if labels provided
                    - logits: Classification predictions
                    - hidden_states (optional): Tuple of hidden states from all layers
                    - attentions (optional): Tuple of attention weights from all layers
            If return_dict=False:
                logits tensor only (for backward compatibility)
        """
        # Get ViT outputs
        vit_outputs = self.vit(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        # Classification logits using CLS token (first token)
        cls_output = vit_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        # For backward compatibility: return logits only if return_dict=False
        if not return_dict:
            return logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        # Prepare output dictionary
        outputs = {'logits': logits}
        
        if loss is not None:
            outputs['loss'] = loss
        
        if output_hidden_states:
            outputs['hidden_states'] = vit_outputs.hidden_states
        
        if output_attentions:
            outputs['attentions'] = vit_outputs.attentions
        
        return outputs


def init_weights_kaiming(module):
    """
    Initialize model weights using Kaiming initialization.
    
    Args:
        module: PyTorch module to initialize
    """
    if isinstance(module, nn.Linear):
        # Kaiming initialization for linear layers
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
        # Kaiming initialization for convolutional layers
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        # Normal initialization for embeddings
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        # Standard initialization for normalization layers
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def load_vit_config(config_path):
    """Load ViT configuration from JSON file."""
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


def cleanup_checkpoints(checkpoint_dir, save_total_limit):
    """
    Remove old checkpoint files, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        save_total_limit: Maximum number of checkpoint files to keep (excluding best_model.pth)
    """
    if save_total_limit is None or save_total_limit <= 0:
        return
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Get all epoch checkpoint files (exclude best_model.pth)
    epoch_checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pth"),
        key=lambda x: x.stat().st_mtime,  # Sort by modification time
        reverse=True  # Newest first
    )
    
    # Remove old checkpoints if exceeding limit
    if len(epoch_checkpoints) > save_total_limit:
        for checkpoint_to_remove in epoch_checkpoints[save_total_limit:]:
            checkpoint_to_remove.unlink()


class GrayscaleToRGB:
    """Convert grayscale image to RGB by repeating channels."""
    def __call__(self, x):
        return x.repeat(3, 1, 1)


def prepare_dataloaders(data_root, batch_size, num_workers=4):
    """Prepare MNIST dataloaders with preprocessing for ViT."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to ViT input size
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            GrayscaleToRGB(),  # Convert grayscale to RGB
        ]
    )

    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, logger, writer, epoch, global_step, args, edgerazor=None, teacher_model=None):
    """Train for one epoch."""
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    
    running_loss = 0.0
    running_task_loss = 0.0
    running_distill_loss = 0.0
    correct = 0
    total = 0
    
    # Check if KD is enabled
    use_kd = edgerazor is not None and edgerazor.is_kd_enabled and teacher_model is not None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_kd:
            # Knowledge distillation mode: get full outputs
            student_outputs = model(
                pixel_values=images,
                labels=labels,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True
            )
            
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    pixel_values=images,
                    output_hidden_states=True,
                    output_attentions=False,
                    return_dict=True
                )
            
            # Compute loss using EdgeRazor (includes task loss + distillation loss)
            # Note: ViT doesn't use attention_mask (no padding in image patches) => labels=None
            loss, loss_dict = edgerazor.compute_loss(
                student_outputs,
                teacher_outputs,
                labels=None,
            )
            
            task_loss_value = loss_dict.get('task_loss', 0.0)
            distill_loss_value = loss_dict.get('distill_loss', 0.0)
            running_task_loss += task_loss_value
            running_distill_loss += distill_loss_value
            
            # Get logits for accuracy calculation
            outputs = student_outputs['logits']
        else:
            # Standard training mode
            outputs = model(pixel_values=images, return_dict=False)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        batch_correct = predicted.eq(labels).sum().item()
        batch_total = labels.size(0)
        batch_acc = batch_correct / batch_total

        running_loss += loss.item()
        correct += batch_correct
        total += batch_total

        # Update progress bar
        pbar_info = {
            "loss": f"{loss.item():.4f}",
            "acc": f"{batch_acc:.4f}",
            "lr": f"{scheduler.get_lr()[0]:.6f}",
        }
        if use_kd:
            pbar_info["task"] = f"{task_loss_value:.4f}"
            pbar_info["dist"] = f"{distill_loss_value:.4f}"
        
        pbar.set_postfix(pbar_info)

        # Log to TensorBoard
        if writer is not None and (batch_idx + 1) % args.log_interval == 0:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/batch_acc", batch_acc, global_step)
            writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)
            
            if use_kd:
                writer.add_scalar("train/batch_task_loss", task_loss_value, global_step)
                writer.add_scalar("train/batch_distill_loss", distill_loss_value, global_step)

        global_step += 1

    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

    if use_kd:
        epoch_task_loss = running_task_loss / len(train_loader)
        epoch_distill_loss = running_distill_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1:02d} Training   - Loss: {epoch_loss:.4f} "
            f"(Task: {epoch_task_loss:.4f}, Distill: {epoch_distill_loss:.4f}), "
            f"Accuracy: {epoch_acc:.4f} ({correct}/{total})"
        )
    else:
        logger.info(f"Epoch {epoch+1:02d} Training   - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f} ({correct}/{total})")

    return epoch_loss, epoch_acc, global_step


def evaluate(model, dataloader, criterion, device, logger, epoch, args, split="Test"):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [{split}]")
        for images, labels in pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # Get outputs (backward compatible with return_dict=False)
            outputs = model(pixel_values=images, return_dict=False)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})

    # Calculate metrics
    loss = running_loss / len(dataloader)
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(f"Epoch {epoch+1:02d} {split:<8} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f} ({correct}/{total})")
    logger.info(f"{'':>16}Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return loss, accuracy, precision, recall, f1


def main():
    """Main training loop."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup logger
    logger = get_logger("ViT-MNIST")
    logger.info("=" * 80)
    logger.info("ViT Training on MNIST with EdgeRazor QAT")
    logger.info("=" * 80)

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Model config: {args.model_config}")
    if args.edgerazor_config:
        logger.info(f"  EdgeRazor config: {args.edgerazor_config}")
    else:
        logger.info(f"  Quantization: {args.quant_config if args.quant_config else 'None (Full Precision)'}")
        logger.info(f"  KD config: {args.kd_config if args.kd_config else 'None'}")
    if args.teacher_pretrained_path:
        logger.info(f"  Teacher pretrained: {args.teacher_pretrained_path}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Weight decay: {args.weight_decay}")
    logger.info(f"  Warmup steps: {args.warmup_steps}")
    logger.info(f"  Data type: {args.dtype}")
    logger.info(f"  Random seed: {args.seed}")

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")
    
    # Setup dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    logger.info(f"  Data type: {args.dtype} ({dtype})")
    logger.info("")

    # Load ViT config and create model
    logger.info("Building ViT model...")
    vit_config = load_vit_config(args.model_config)
    logger.info(f"  Hidden size: {vit_config.hidden_size}")
    logger.info(f"  Layers: {vit_config.num_hidden_layers}")
    logger.info(f"  Attention heads: {vit_config.num_attention_heads}")
    logger.info(f"  Image size: {vit_config.image_size}x{vit_config.image_size}")
    logger.info(f"  Patch size: {vit_config.patch_size}x{vit_config.patch_size}")

    model = ViTForMNIST(vit_config)

    # Initialize model weights with Kaiming initialization
    logger.info("Initializing model weights with Kaiming initialization...")
    model.apply(init_weights_kaiming)
    logger.info("✓ Model weights initialized")
    logger.info("")

    # Initialize EdgeRazor (QAT and/or KD)
    edgerazor = None
    if args.edgerazor_config is not None:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Initializing EdgeRazor (Unified QAT + KD)")
        logger.info("=" * 80)
        logger.info("")
        
        edgerazor = EdgeRazor(config=args.edgerazor_config)
        
        # Apply QAT if enabled
        if edgerazor.is_qat_enabled:
            logger.info("Applying Quantization Aware Training (QAT)...")
            model = edgerazor.quantize(model)
            logger.info("✓ QAT applied")
        else:
            logger.info("QAT: disabled")
        
        # Log KD status
        if edgerazor.is_kd_enabled:
            logger.info("Knowledge Distillation (KD): enabled")
        else:
            logger.info("Knowledge Distillation (KD): disabled")
        
        logger.info("")
    elif args.quant_config is not None:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Initializing EdgeRazor (QAT only)")
        logger.info("=" * 80)
        logger.info("")
        
        edgerazor = EdgeRazor(qat_config=args.quant_config)
        model = edgerazor.quantize(model)
        
        logger.info("")
    elif args.kd_config is not None:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Initializing EdgeRazor (KD only)")
        logger.info("=" * 80)
        logger.info("")
        
        edgerazor = EdgeRazor(kd_config=args.kd_config)
        
        logger.info("")
    else:
        logger.info("Training with full precision (no quantization or distillation)")
        logger.info("")

    # Convert model to specified dtype and move to device
    model = model.to(device=device, dtype=dtype)
    
    # Create teacher model if KD is enabled
    teacher_model = None
    if edgerazor is not None and edgerazor.is_kd_enabled:
        logger.info("=" * 80)
        logger.info("Creating Teacher Model for Knowledge Distillation")
        logger.info("=" * 80)
        logger.info("")
        
        # Create teacher model with same architecture
        teacher_model = ViTForMNIST(vit_config)
        teacher_model.apply(init_weights_kaiming)
        
        # Load pretrained weights if provided
        if args.teacher_pretrained_path is not None:
            logger.info(f"Loading pretrained teacher weights from: {args.teacher_pretrained_path}")
            
            try:
                # Load checkpoint with weights_only=False to support metadata (e.g., args with Path objects)
                # This is safe when loading checkpoints from trusted sources
                checkpoint = torch.load(args.teacher_pretrained_path, map_location="cpu", weights_only=False)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model_state_dict" in checkpoint:
                        state_dict = checkpoint["model_state_dict"]
                        logger.info("  Loaded from checkpoint format (key: 'model_state_dict')")
                        
                        # Log checkpoint metadata if available
                        if "epoch" in checkpoint:
                            logger.info(f"  Checkpoint epoch: {checkpoint['epoch']}")
                        if "val_acc" in checkpoint:
                            logger.info(f"  Checkpoint validation accuracy: {checkpoint['val_acc']:.4f}")
                    else:
                        state_dict = checkpoint
                        logger.info("  Loaded from state dict format")
                else:
                    state_dict = checkpoint
                    logger.info("  Loaded from state dict format")
                
                # Load weights into teacher model
                missing_keys, unexpected_keys = teacher_model.load_state_dict(state_dict, strict=False)
                
                if missing_keys:
                    logger.warning(f"  Missing keys in teacher checkpoint: {len(missing_keys)}")
                    if len(missing_keys) <= 10:
                        for key in missing_keys:
                            logger.warning(f"    - {key}")
                    else:
                        for key in missing_keys[:5]:
                            logger.warning(f"    - {key}")
                        logger.warning(f"    ... and {len(missing_keys) - 5} more")
                
                if unexpected_keys:
                    logger.warning(f"  Unexpected keys in teacher checkpoint: {len(unexpected_keys)}")
                    if len(unexpected_keys) <= 10:
                        for key in unexpected_keys:
                            logger.warning(f"    - {key}")
                    else:
                        for key in unexpected_keys[:5]:
                            logger.warning(f"    - {key}")
                        logger.warning(f"    ... and {len(unexpected_keys) - 5} more")
                
                if not missing_keys and not unexpected_keys:
                    logger.info("  ✓ All keys matched successfully")
                
                logger.info("✓ Pretrained teacher weights loaded successfully")
                
            except FileNotFoundError:
                logger.error(f"✗ Teacher checkpoint not found: {args.teacher_pretrained_path}")
                logger.error("  Training will continue with randomly initialized teacher model")
            except Exception as e:
                logger.error(f"✗ Error loading teacher checkpoint: {e}")
                logger.error("  Training will continue with randomly initialized teacher model")
        else:
            logger.info("No pretrained teacher weights provided (--teacher_pretrained_path not set)")
            logger.info("Teacher model initialized with random weights")
        
        teacher_model = teacher_model.to(device=device, dtype=dtype)
        teacher_model.eval()
        
        # Count teacher parameters
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        logger.info("")
        logger.info("Teacher model summary:")
        logger.info(f"  Total parameters: {teacher_params:,}")
        logger.info("")

    # Display actual weight dtype
    sample_param = next(model.parameters())
    logger.info("Model dtype information:")
    logger.info(f"  Weight dtype: {sample_param.dtype}")
    logger.info(f"  Weight device: {sample_param.device}")
    logger.info("")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info("")

    # Prepare data
    logger.info("Preparing MNIST dataloaders...")
    train_loader, test_loader = prepare_dataloaders(args.data_root, args.batch_size, args.num_workers)
    logger.info(f"  Training samples: {len(train_loader.dataset):,}")
    logger.info(f"  Test samples: {len(test_loader.dataset):,}")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    logger.info("")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=args.warmup_steps, total_steps=total_steps, min_lr=args.min_lr
    )

    logger.info("Training setup:")
    logger.info("  Optimizer: AdamW")
    logger.info("  Loss function: CrossEntropyLoss")
    logger.info("  Scheduler: Warmup + Cosine Annealing")
    logger.info(f"  Total steps: {total_steps:,}")
    logger.info("")

    # Generate run name once for consistent directory naming
    run_name = get_run_name(args)

    # Setup TensorBoard
    writer = None
    if not args.no_tensorboard:
        log_dir = args.output_dir / "logs" / run_name
        writer = SummaryWriter(log_dir=log_dir)
        logger.info("TensorBoard logging enabled:")
        logger.info(f"  Log directory: {log_dir}")
        logger.info(f"  Run name: {run_name}")
        logger.info("")

    # Training loop
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    logger.info("")

    global_step = 0
    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0  # Early stopping counter

    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, logger, writer, epoch, global_step, args,
            edgerazor=edgerazor, teacher_model=teacher_model
        )

        # Evaluate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model, test_loader, criterion, device, logger, epoch, args, split="Test"
        )

        logger.info("")

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/train_acc", train_acc, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_acc", val_acc, epoch)
            writer.add_scalar("epoch/val_precision", val_precision, epoch)
            writer.add_scalar("epoch/val_recall", val_recall, epoch)
            writer.add_scalar("epoch/val_f1", val_f1, epoch)

        # Check for improvement and update early stopping counter
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % args.save_freq == 0 or is_best:
            checkpoint_dir = args.output_dir / "checkpoints" / run_name
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Build checkpoint dictionary
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "args": vars(args),
            }

            # Optionally save optimizer and scheduler states
            if args.save_optimizer_state:
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint["scheduler_state"] = {
                    "current_step": scheduler.current_step,
                    "base_lrs": scheduler.base_lrs,
                }

            if is_best:
                checkpoint_path = checkpoint_dir / "best_model.pth"
                logger.info(f"✓ New best accuracy: {best_acc:.4f}, saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)
            
            checkpoint_path = checkpoint_dir / f"epoch_{epoch+1:02d}.pth"
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)
            
            cleanup_checkpoints(checkpoint_dir, args.save_total_limit)
            
            logger.info("")

        # Early stopping check
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            logger.info("=" * 80)
            logger.info(f"Early stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
            logger.info(f"Best validation accuracy: {best_acc:.4f} (Epoch {best_epoch})")
            logger.info("=" * 80)
            break

    # Final summary
    logger.info("=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    logger.info(f"Best validation accuracy: {best_acc:.4f} (Epoch {best_epoch})")
    logger.info("=" * 80)

    # Close TensorBoard writer
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
