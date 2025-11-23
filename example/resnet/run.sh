#!/bin/bash

# ============================================================================
# Training Configuration Variables
# ============================================================================
MODEL_CONFIG="src/ResNet-18.json"
BATCH_SIZE=512
EPOCHS=10
LR=1e-4
WEIGHT_DECAY=0.2
WARMUP_STEPS=100
OUTPUT_DIR="./runs"
EARLY_STOPPING_PATIENCE=0

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
    --quant_config ./q_resnet_w1.58_a16.yaml

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
    --quant_config ./q_resnet_w1.58_a8.yaml

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
    --quant_config ./q_resnet_w1.58_a4.yaml

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
    --quant_config ./q_resnet_w4_a16.yaml

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
    --quant_config ./q_resnet_w4_a8.yaml

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
    --quant_config ./q_resnet_w4_a4.yaml
