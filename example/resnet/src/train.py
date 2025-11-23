"""
ResNet-18 Training Script with EdgeRazor QAT

Train ResNet-18 on MNIST dataset with optional quantization-aware training (QAT).
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm

from edgerazor.log import get_logger
from edgerazor.qat import QAT

from .arg import get_run_name, parse_args


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine annealing."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr_scale = self.current_step / self.warmup_steps
            lrs = [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + np.cos(np.pi * progress))
            lrs = [self.min_lr + (base_lr - self.min_lr) * lr_scale for base_lr in self.base_lrs]

        for param_group, lr in zip(self.optimizer.param_groups, lrs, strict=True):
            param_group["lr"] = lr

    def get_lr(self):
        """Return current learning rates."""
        return [group["lr"] for group in self.optimizer.param_groups]


class ResNet18ForMNIST(nn.Module):
    """ResNet-18 model adapted for MNIST classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        
        # Load pretrained ResNet-18 and modify for MNIST
        self.resnet = resnet18(weights=None)
        
        # Replace the final fully connected layer for MNIST (10 classes)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def init_weights_kaiming(module):
    """
    Initialize model weights using Kaiming (He) initialization.
    
    Kaiming initialization is designed for layers with ReLU activations:
    - For Linear layers: uses fan_in mode (input connections) to preserve variance during forward pass
    - For Conv2d layers: uses fan_out mode (output connections) for proper gradient flow
    
    This helps maintain stable activations and gradients during training.
    """
    if isinstance(module, nn.Linear):
        # fan_in mode: variance is preserved with respect to the number of input connections
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        # fan_out mode: commonly used for conv layers to maintain gradient variance
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def load_resnet_config(config_path):
    """Load ResNet configuration from JSON file."""
    with open(config_path) as f:
        config = json.load(f)
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
    """Prepare MNIST dataloaders with preprocessing for ResNet."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to ResNet input size
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


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, logger, writer, epoch, global_step, args):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Get model dtype for input conversion
    model_dtype = next(model.parameters()).dtype

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        # Convert inputs to match model dtype
        images = images.to(device=device, dtype=model_dtype, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
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
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{batch_acc:.4f}",
                "lr": f"{scheduler.get_lr()[0]:.6f}",
            }
        )

        # Log to TensorBoard
        if writer is not None and (batch_idx + 1) % args.log_interval == 0:
            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            writer.add_scalar("train/batch_acc", batch_acc, global_step)
            writer.add_scalar("train/lr", scheduler.get_lr()[0], global_step)

        global_step += 1

    # Epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total

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
    
    # Get model dtype for input conversion
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d} [{split}]")
        for images, labels in pbar:
            # Convert inputs to match model dtype
            images = images.to(device=device, dtype=model_dtype, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
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
    logger = get_logger("ResNet-MNIST")
    logger.info("=" * 80)
    logger.info("ResNet-18 Training on MNIST with EdgeRazor QAT")
    logger.info("=" * 80)

    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Model config: {args.model_config}")
    logger.info(f"  Quantization: {args.quant_config if args.quant_config else 'None (Full Precision)'}")
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

    # Load ResNet config and create model
    logger.info("Building ResNet-18 model...")
    resnet_config = load_resnet_config(args.model_config)
    num_classes = resnet_config.get("num_classes", 10)
    logger.info(f"  Model: {resnet_config.get('model', 'resnet18')}")
    logger.info(f"  Number of classes: {num_classes}")
    logger.info(f"  Input channels: {resnet_config.get('input_channels', 3)}")
    logger.info(f"  Image size: {resnet_config.get('image_size', 224)}x{resnet_config.get('image_size', 224)}")

    model = ResNet18ForMNIST(num_classes=num_classes)

    # Initialize model weights with Kaiming initialization
    logger.info("Initializing model weights with Kaiming initialization...")
    model.apply(init_weights_kaiming)
    logger.info("✓ Model weights initialized")
    logger.info("")

    # Apply quantization if config provided
    if args.quant_config is not None:
        logger.info("")
        logger.info("=" * 80)
        logger.info("Applying Quantization Aware Training (QAT)")
        logger.info("=" * 80)
        logger.info("")

        qat = QAT(config_path=args.quant_config)
        model = qat.quantize(model)

        logger.info("")
    else:
        logger.info("Training with full precision (no quantization)")
        logger.info("")

    # Convert model to specified dtype and move to device
    model = model.to(device=device, dtype=dtype)

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
            model, train_loader, criterion, optimizer, scheduler, device, logger, writer, epoch, global_step, args
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
