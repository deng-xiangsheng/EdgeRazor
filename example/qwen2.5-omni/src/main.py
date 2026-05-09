# ruff: noqa: F541, F401

import os
import logging
from typing import Any
from datetime import datetime
import numpy._core.multiarray

from config import EdgeRazorTrainConfig
from dataset import MultimodalDataset
from modeling import (
    create_student_model,
    create_teacher_model,
)
import torch
from torch.serialization import add_safe_globals
from trainer import EdgeRazorTrainer
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DefaultDataCollator,
    TrainingArguments,
    Qwen2_5OmniProcessor,
)
from functools import partial

from edgerazor import EdgeRazor

logger = logging.getLogger(__name__)
IGNORE_INDEX = -100

# Define collate function (requires pad_token_id)
def multimodal_collate_fn(
    batch: list[dict],
    pad_token_id: int = 151643,  # <|endoftext|>
    debug: bool = False,
) -> dict[str, Any]:
    """
    Qwen2.5-Omni multimodal data collate function

    Args:
        batch: Data batch
        pad_token_id: padding token ID
        debug: Whether to print debug information

    Returns:
        Batch dictionary containing:
        - input_ids, attention_mask, labels: Text tensors
        - pixel_values_videos, video_grid_thw: Video tensors (optional)
        - video_patch_counts: Patch count per sample (for debugging)
        - video_counts: Video count per sample (for debugging)
    """
    # batch_size = len(batch)
    
    if debug:
        _log_batch_debug_info(batch)
    
    # ========== 1. Text Padding ==========
    result = _pad_text_inputs(batch, pad_token_id)
    
    # ========== 2. Video Features ==========
    if _has_modality(batch, 'pixel_values_videos'):
        video_result = _concat_video_features(batch, debug)
        result.update(video_result)
    
    # ========== 3. Image Features ==========
    if _has_modality(batch, 'pixel_values'):
        result.update(_concat_image_features(batch))
    
    # ========== 4. Audio Features ==========
    if _has_modality(batch, 'input_features'):
        result.update(_pad_audio_features(batch))
    
    return result


def _has_modality(batch: list[dict], key: str) -> bool:
    """Check if a modality exists in the batch"""
    has = [item.get(key) is not None for item in batch]
    if any(has) and not all(has):
        raise ValueError(
            f"Mixed batch: some samples have '{key}', some don't. "
            f"Ensure your data is consistent or use separate dataloaders."
        )
    return all(has)


def _pad_text_inputs(batch: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    """
    Pad text inputs

    Returns:
        dict with input_ids, attention_mask, labels
    """
    batch_size = len(batch)
    
    input_ids_list = [item['input_ids'] for item in batch]
    attention_mask_list = [item['attention_mask'] for item in batch]
    labels_list = [item['labels'] for item in batch]
    
    max_len = max(ids.size(0) for ids in input_ids_list)
    dtype = input_ids_list[0].dtype
    
    # Initialize
    padded_input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=dtype)
    padded_attention_mask = torch.zeros((batch_size, max_len), dtype=dtype)
    padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
    
    # Fill
    for i, (ids, mask, lbl) in enumerate(zip(input_ids_list, attention_mask_list, labels_list)):
        length = ids.size(0)
        padded_input_ids[i, :length] = ids
        padded_attention_mask[i, :length] = mask
        padded_labels[i, :length] = lbl
    
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels,
    }


def _concat_video_features(batch: list[dict], debug: bool = False) -> dict[str, torch.Tensor]:
    """
    Concatenate video features

    Validation: pixel_values_videos.shape[0] == Σ(T·H·W) from video_grid_thw

    Returns:
        dict with:
        - pixel_values_videos: [total_patches, 1176]
        - video_grid_thw: [total_videos, 3]
        - video_second_per_grid: [total_videos] (optional)
        - video_patch_counts: [batch_size] - patch count per sample
        - video_counts: [batch_size] - video count per sample
    """
    videos = [item['pixel_values_videos'] for item in batch]
    grids = [item['video_grid_thw'] for item in batch]
    
    # Record per-sample info (for debugging and alignment)
    video_patch_counts = torch.tensor([v.shape[0] for v in videos], dtype=torch.long)
    video_counts = torch.tensor([g.shape[0] for g in grids], dtype=torch.long)
    
    # Validate consistency for each sample
    for i, (video, grid) in enumerate(zip(videos, grids)):
        num_patches = video.shape[0]
        expected_patches = sum(t * h * w for t, h, w in grid.tolist())
        
        if num_patches != expected_patches:
            raise ValueError(
                f"Sample {i}: patches mismatch! "
                f"pixel_values_videos.shape[0]={num_patches}, "
                f"Σ(T·H·W)={expected_patches}"
            )
    
    result = {
        'pixel_values_videos': torch.cat(videos, dim=0),
        'video_grid_thw': torch.cat(grids, dim=0),
        'video_patch_counts': video_patch_counts,
        'video_counts': video_counts,
    }
    
    # Optional fields
    if batch[0].get('video_second_per_grid') is not None:
        seconds = [item['video_second_per_grid'] for item in batch]
        result['video_second_per_grid'] = torch.cat(seconds, dim=0)
    
    if debug:
        total_patches = result['pixel_values_videos'].shape[0]
        total_videos = result['video_grid_thw'].shape[0]
        logger.info(
            f"Video features: total_patches={total_patches}, "
            f"total_videos={total_videos}, "
            f"patch_counts={video_patch_counts.tolist()}, "
            f"video_counts={video_counts.tolist()}"
        )
    
    return result


def _concat_image_features(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Concatenate image features"""
    return {
        'pixel_values': torch.cat(
            [item['pixel_values'] for item in batch], dim=0
        ),
        'image_grid_thw': torch.cat(
            [item['image_grid_thw'] for item in batch], dim=0
        ),
    }


def _pad_audio_features(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad audio features"""
    batch_size = len(batch)
    audio_features = [item['input_features'] for item in batch]
    audio_masks = [item.get('feature_attention_mask') for item in batch]
    
    max_len = max(f.size(-1) for f in audio_features)
    feat_dim = audio_features[0].size(0)
    dtype = audio_features[0].dtype
    
    padded_audio = torch.zeros((batch_size, feat_dim, max_len), dtype=dtype)
    padded_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, (feat, mask) in enumerate(zip(audio_features, audio_masks)):
        length = feat.size(-1)
        padded_audio[i, :, :length] = feat
        padded_mask[i, :length] = mask if mask is not None else 1
    
    return {
        'input_features': padded_audio,
        'feature_attention_mask': padded_mask,
    }


def _log_batch_debug_info(batch: list[dict]) -> None:
    """Print batch debug info"""
    logger.info("=" * 70)
    logger.info(f"Collating batch of size {len(batch)}")
    
    total_patches = 0
    total_videos = 0
    
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        
        patches = 0
        num_videos = 0
        grid_info = ""
        
        if item.get('pixel_values_videos') is not None:
            patches = item['pixel_values_videos'].shape[0]
            total_patches += patches
            
            grids = item['video_grid_thw'].tolist()
            num_videos = len(grids)
            total_videos += num_videos
            
            expected = sum(t * h * w for t, h, w in grids)
            status = "✓" if patches == expected else "✗"
            grid_info = f", grids={grids}, expected={expected} {status}"
        
        logger.info(
            f"  [{i}] seq_len={seq_len}, patches={patches}, "
            f"videos={num_videos}{grid_info}"
        )
    
    logger.info(f"  TOTAL: patches={total_patches}, videos={total_videos}")
    logger.info("=" * 70)

# Training configuration
config = EdgeRazorTrainConfig()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


if __name__ == "__main__":
    add_safe_globals([numpy._core.multiarray._reconstruct])
    
    # Print the final configuration being used
    print("="*80)
    print("Final Training Configuration:")
    print(f"  Teacher Path: {config.teacher_path}")
    print(f"  Student Path: {config.student_path}")
    print(f"  Config Path: {config.config_path}")
    print(f"  Output Dir: {config.output_dir}")
    print(f"  Epochs: {config.epoch}")
    print(f"  Learning Rate: {config.lr}")
    print(f"  Per Device Batch Size: {config.per_device_bs}")
    print(f"  Gradient Accumulation Steps: {config.grad_acc_steps}")
    print(f"  Max Sequence Length: {config.max_seq_len}")
    print("="*80)
    
    # Create detailed TensorBoard experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    teacher_name = os.path.basename(config.teacher_path)
    student_name = os.path.basename(config.student_path)
    # config_name = config.config_path.split("_")[-1].replace(".yaml", "")
    config_name = config.config_path.split("/")[-1].replace(".yaml", "")
    # tensorboard_name = f"{timestamp}_ep{config.epoch}_lr{config.lr}_{config_name}"
    tensorboard_name = f"{timestamp}"
    if config.tag_name:
        tensorboard_name = f"{config.tag_name}" + "_" + tensorboard_name
    
    # Distributed training configuration
    device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
    print(f"Device map: {device_map}")
    print(f"TensorBoard experiment name: {tensorboard_name}")
    
    # Create teacher model
    teacher_model = create_teacher_model(
        teacher_path=config.teacher_path,
        device_map=device_map,
        attn_implementation=config.attn_implementation,
        debug=False,
    )
    
    # Create student model
    edgerazor = EdgeRazor(config=config.config_path)
    student_model = create_student_model(
        student_path=config.student_path,
        device_map=device_map,
        attn_implementation=config.attn_implementation,
        edgerazor=edgerazor,
        debug=False,
    )
    
    print(f"Teacher model device: {next(teacher_model.parameters()).device}")
    print(f"Student model device: {next(student_model.parameters()).device}")
    tokenizer = AutoTokenizer.from_pretrained(config.student_path, use_cache=False, use_fast=True)
    processor = Qwen2_5OmniProcessor.from_pretrained(config.student_path)
    
    args = TrainingArguments(
        # Basic configuration
        output_dir=config.output_dir,           # Model output directory
        run_name=tensorboard_name,              # Experiment run name
        num_train_epochs=config.epoch,          # Number of training epochs
        max_steps=config.steps,                 # Maximum training steps, -1 means using epoch-based step count; use 100 to test training code correctness
        do_train=True,                          # Enable training mode
        seed=3407,                              # Random seed
        data_seed=3407,                         # Data random seed
        
        # --------------------------------------------------------------------------
        per_device_train_batch_size=config.per_device_bs,   # Batch size per device
        gradient_accumulation_steps=config.grad_acc_steps,  # Gradient accumulation steps
        # --------------------------------------------------------------------------
        
        # Optimizer configuration
        learning_rate=config.lr,                # Learning rate
        lr_scheduler_type=config.lr_scheduler,  # Learning rate scheduler type
        lr_scheduler_kwargs={
            'min_lr': config.min_lr,            # Set minimum learning rate (ignored by constant_with_warmup)
        },
        warmup_ratio=config.warmup_ratio,       # Warmup ratio: x% of steps used for warmup
        max_grad_norm=config.max_grad_norm,     # Gradient clipping: prevents gradient explosion
        optim=config.optim,                     # Optimizer: 8-bit AdamW (saves GPU memory)
        adam_beta1=config.adam_beta1,           # Adam optimizer beta1 parameter (exponential decay rate for first moment estimate)
        adam_beta2=config.adam_beta2,           # Adam optimizer beta2 parameter (exponential decay rate for second moment estimate)
        adam_epsilon=config.adam_epsilon,       # Adam optimizer epsilon parameter (numerical stability)
        weight_decay=config.weight_decay,       # Weight decay coefficient (L2 regularization)
        
        # Precision and memory optimization
        bf16=True,                                 # Enable BF16 mixed precision training
        fp16=False,                                # Disable FP16 (mutually exclusive with BF16)
        gradient_checkpointing=config.grad_chkpt,  # Enable gradient checkpointing (saves GPU memory) if True => use_cache=False
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Non-reentrant mode (more stable)
        
        # Save strategy
        save_strategy=config.save_strategy,     # Save strategy: save by steps
        save_steps=config.save_steps,           # Save interval: save every [?] steps
        save_total_limit=20,                    # Save limit: keep at most 20 checkpoints => requires disk space XXGB*20=[?]GB
        save_safetensors=True,                  # Save model in safetensors format (safer)
        
        # Logging and monitoring - enhanced TensorBoard configuration
        logging_steps=1,                        # Logging interval: log every 1 step
        logging_dir=f"{config.output_dir}/tensorboard/{tensorboard_name}",  # TensorBoard log directory
        report_to='tensorboard',                # Log reporter: use TensorBoard
        logging_first_step=True,                # Log the first step
        log_level='info',                       # Log level
        do_eval=False,                          # Disable evaluation
        eval_strategy='no',                     # Evaluation strategy: evaluate by steps
        
        # Data loading optimization
        dataloader_num_workers=16,              # Data loader worker processes: 16
        dataloader_pin_memory=True,             # Enable memory pinning (accelerates GPU transfer)

        # Distributed training configuration - adapted for dedicated GPU setup
        deepspeed=config.ds_path,               # DeepSpeed configuration: ZeRO-3 optimization
        ddp_find_unused_parameters=False,       # Don't search for unused parameters (teacher model doesn't participate in training)
        dataloader_drop_last=True,              # Drop the last incomplete batch
    )
    
    dataset = MultimodalDataset(
        dataset_path=config.dataset_path,
        processor=processor,
        max_seq_len=config.max_seq_len,
        add_system_prompt=True,
        use_audio_in_video=config.use_audio_in_video,
        validate=True,  # Validate patches consistency
    )

    # collate
    collate_config = dataset.get_collate_config()
    print(f"Collate config: {collate_config}")
    data_collator = partial(
        multimodal_collate_fn,
        pad_token_id=collate_config['pad_token_id'],
        debug=False,  # Disable during training, enable for debugging
    )

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 1
    batch_size = num_gpus * args.per_device_train_batch_size * args.gradient_accumulation_steps
    total_steps = len(dataset) // batch_size * args.num_train_epochs
    dataset_len = len(dataset)
    # total_tokens, max_len, max_cot, max_answer, seq_lengths = dataset.get_token_stats()
    print(f"PyTorch batch_size: {batch_size}, HF-Trainer batch_size: {args.per_device_train_batch_size}*{args.gradient_accumulation_steps}*{args.world_size}={args.per_device_train_batch_size * args.gradient_accumulation_steps * args.world_size}")
    print(f"Total steps: {total_steps}, Dataset samples: {dataset_len}")
    
    # Create EdgeRazorTrainer to handle quantization training loop
    trainer = EdgeRazorTrainer(
                                model=student_model, # if pass model.thinker, trainer need to be modified + partial_combine.py need to be executed for combining thinker weights and other weights
                                teacher_model=teacher_model,
                                args=args,
                                train_dataset=dataset,
                                tokenizer=processor.tokenizer,
                                data_collator=data_collator,  # Multimodal collate function (customized for Qwen2.5-Omni)
                                router_aux_loss_coef=config.router_aux_loss_coef,
                                router_z_loss_coef=config.router_z_loss_coef,
                                edgerazor=edgerazor,
                                use_audio_in_video=config.use_audio_in_video,
                                )
    
    print(f"Starting training with the following configuration:")
    print(f"  - Teacher model: {teacher_name}")
    print(f"  - Student model: {student_name}")
    print(f"  - Dataset size: {dataset_len}")
    print(f"  - Configured max_seq_len: {config.max_seq_len}")
    print(f"  - Batch size: {args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Total steps: {total_steps}")
    print(f"  - TensorBoard logs: {args.logging_dir}")
    
    # If first training: resume_from_checkpoint=False
    # If resume training: trainer.train(resume_from_checkpoint=True)
    trainer.train(resume_from_checkpoint=False)
    
    # Make sure to convert model to bfloat16 before saving to save disk space, and it won't affect the evaluation results since we will load the model in bfloat16 for evaluation.
    trainer.model.to(torch.bfloat16)
    trainer.save_model(config.final_model)
    trainer.save_state()
