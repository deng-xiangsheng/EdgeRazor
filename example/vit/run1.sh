#!/bin/bash

# ============================================================================
# Training Configuration Variables
# ============================================================================
MODEL_CONFIG="src/ViT-S-16.json"
BATCH_SIZE=640
EPOCHS=10
LR=1e-4
WEIGHT_DECAY=0.2
WARMUP_STEPS=100
OUTPUT_DIR="./runs1"
EARLY_STOPPING_PATIENCE=0

mkdir -p $OUTPUT_DIR

# ============================================================================
# Training Commands
# ============================================================================

# Baseline model training: W16-A16
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE

# Quantized model training: W4-A16
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w4_a16.yaml

# Quantized model training: W4-A8
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w4_a8.yaml

# Quantized model training: W4-A4
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w4_a4.yaml

# Quantized model training: W1.58-A16
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w1.58_a16.yaml

# Quantized model training: W1.58-A8
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w1.58_a8.yaml

# Quantized model training: W1.58-A4
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w1.58_a4.yaml

# Quantized model training: W1.58+4-A16
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w1.58mp4_a16.yaml

# Quantized model training: W1.58+4-A8
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w1.58mp4_a8.yaml

# Quantized model training: W1.58+4-A4
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --quant_config ./q_vit_w1.58mp4_a4.yaml

# Quantized model distillation(kldr+fd): W1.58+4-A16
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --edgerazor_config ./q_vit_w1.58mp4_a16_kldr_fd.yaml \
    --teacher_pretrained_path ./$OUTPUT_DIR/checkpoints/fp_vit_w16_a16/best_model.pth

# Quantized model distillation(kldr+fd): W1.58+4-A8
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --edgerazor_config ./q_vit_w1.58mp4_a8_kldr_fd.yaml \
    --teacher_pretrained_path ./$OUTPUT_DIR/checkpoints/fp_vit_w16_a16/best_model.pth

# Quantized model distillation(kldr+fd): W1.58+4-A4
python -m src.train \
    --model_config $MODEL_CONFIG \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --edgerazor_config ./q_vit_w1.58mp4_a4_kldr_fd.yaml \
    --teacher_pretrained_path ./$OUTPUT_DIR/checkpoints/fp_vit_w16_a16/best_model.pth
