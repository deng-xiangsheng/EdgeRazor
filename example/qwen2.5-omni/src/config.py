#!/usr/bin/env python3
import os

# Environment paths
PATH_PREFIX = "/path/to/your/environment"  # Change this to your actual path prefix
CODE_ROOT = f"{PATH_PREFIX}/code/EdgeRazor-Omni"
DATA_ROOT = f"{PATH_PREFIX}/data"

# Create necessary directories if they don't exist
os.makedirs(DATA_ROOT, exist_ok=True)


# Model: Qwen2.5-Omni-7B
class EdgeRazorTrainConfig:
    config_path   = f"{CODE_ROOT}/src/train.yaml"
    ds_path       = f"{CODE_ROOT}/src/ds_z3_config_qwen2_5omni.json"
    teacher_path  = "Qwen/Qwen2.5-Omni-7B"
    student_path  = "Qwen/Qwen2.5-Omni-7B"
    dataset_path  = [
        f"{DATA_ROOT}/video_distilled_tgif_sub10k.jsonl",
    ]
    output_dir    = f"{CODE_ROOT}/train"
    final_model   = f"{output_dir}/final_model"
    
    # Training
    max_seq_len   = 1024   # 10k=OOM, 5k=OOM(grad_ckp=False), 2.5K=OOM, 1K=OK
    epoch         = 2      # 2 epochs
    steps         = -1     # -1 means using epoch to control training length
    optim         = "adamw_8bit"
    lr            = 5e-6
    lr_scheduler  = "cosine_with_min_lr"
    min_lr        = 0
    warmup_ratio  = 0.01
    weight_decay  = 0.10
    adam_beta1    = 0.90
    adam_beta2    = 0.95
    adam_epsilon  = 1e-8
    max_grad_norm = 1.00

    # Omni
    use_audio_in_video = False # No audio from video input during training
    
    # Training environment
    per_device_bs  = 1        # total_batch_size: 1*8*8=64
    grad_acc_steps = 8        # 
    save_strategy  = "epoch"  # 
    save_steps     = 100      # 
    eval_steps     = 100      # 
    
    # Attn
    attn_implementation  = "flash_attention_2"  # ["eager", "flash_attention_2", "sdpa"]
    grad_chkpt           = False                # gradient_checkpointing=True has error
    
    # MoE Loss factors
    router_aux_loss_coef = 0.01
    router_z_loss_coef   = 0.001
    
    # Custom configuration
    tag_name             = "exp_for_sed"