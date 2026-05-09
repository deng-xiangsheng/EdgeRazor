#!/bin/bash

# ============================================================================
# Environment Configuration (shared across all experiments)
# ============================================================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_RETRY_TIMES=10
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_LEVEL=NVL
export DEEPSPEED_TIMEOUT=5400
export NCCL_TIMEOUT=5400
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false

################################################################################
# EdgeRazor-Omni Training & Evaluation Pipeline
################################################################################
# Usage: modify the User Configuration section below, then run:
#   bash run.sh
################################################################################

# ============================================================================
# User Configuration (ONLY modify this section for different experiments)
# ============================================================================

# ---- Environment ----
PATH_PREFIX="/path/to/your/environment"     # Root path for code/models/data

# ---- Model ----
MODEL_NAME="Qwen2.5-Omni-7B"               # Model identifier

# ---- Quantization ----
QUANT_CONFIG="W4A8KV8"                     # W4A8KV8 | W4A16KV16
TRAIN_YAML="train_w4"                      # Config file: ./train_w4.yaml

# ---- Experiment ----
RUN_NAME="qwen2.5-omni-7b-edgerazor-4bit"  # Experiment tag (written to config.py)
TRAIN_VERSION="train"                      # Output subdirectory name

# ============================================================================
# Derived Paths (AUTO-GENERATED — DO NOT MODIFY)
# ============================================================================

CODE_ROOT="${PATH_PREFIX}/code/EdgeRazor-Omni"
MODEL_ROOT="${PATH_PREFIX}/model"

TEMPLATE_PATH="${CODE_ROOT}/template/qwen2_5_omni_7B_edgerazor"

TRAIN_ROOT="${CODE_ROOT}/${TRAIN_VERSION}"
FINAL_MODEL="${TRAIN_ROOT}/final_model"
EVAL_MODEL="${TRAIN_ROOT}/qwen2_5_omni_7B_edgerazor"     # trust_remote_code=False → A16KV16
EVAL_MODEL_A8KV8="${EVAL_MODEL}"                           # trust_remote_code=True  → A8KV8 (same weights, different code path)

TRAIN_LOG_DIR="${TRAIN_ROOT}/logs"
TENSORBOARD_DIR="${TRAIN_ROOT}/tensorboard"
RESULT_DIR="${TRAIN_ROOT}/results"

QUANT_CONFIG_PATH="${CODE_ROOT}/src/${TRAIN_YAML}.yaml"
CONVERT_SCRIPT="${CODE_ROOT}/src/convert/convert_qweight.py"
TEST_SCRIPT="${CODE_ROOT}/src/convert/test_qweight_equivalence.py"

SPK_DICT_PATH="${MODEL_ROOT}/Qwen2.5-Omni-7B-EdgeRazor-4bit_v0122/spk_dict.pt"

export PATH="${HOME}/.conda/envs/edgerazor/bin:${PATH}"

# ============================================================================
# Pre-flight
# ============================================================================

# Copy local train config to source tree, set experiment tag
cp "./${TRAIN_YAML}.yaml" "${QUANT_CONFIG_PATH}"
sed -i.bak "s/= \"exp_for_sed\"/= \"${RUN_NAME}\"/" "${CODE_ROOT}/src/config.py"

# ============================================================================
# Pipeline Execution
# ============================================================================

echo "================================================================================"
echo "EdgeRazor-Omni Training & Evaluation Pipeline"
echo "================================================================================"
echo "Model:           ${MODEL_NAME}"
echo "Quantization:    ${QUANT_CONFIG}"
echo "Experiment:      ${RUN_NAME}"
echo "Train config:    ${TRAIN_YAML}.yaml"
echo "Output:          ${TRAIN_ROOT}"
echo "================================================================================"
echo ""

# ---- Step 1: Prepare environment ----
echo "[Step 1/5] Preparing training environment..."
mkdir -p "${TRAIN_LOG_DIR}" "${TENSORBOARD_DIR}" "${RESULT_DIR}"
cp -rf "${TEMPLATE_PATH}" "${EVAL_MODEL}"
echo "  ✓ Output directories created"
echo "  ✓ Template copied to: ${EVAL_MODEL}"
echo ""

# ---- Step 2: Train ----
echo "[Step 2/5] Starting distributed training with DeepSpeed..."
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
TRAINING_LOG="${TRAIN_LOG_DIR}/train_${timestamp}.log"
time {
    deepspeed --num_gpus=8 "${CODE_ROOT}/src/main.py"
} 2>&1 | tee "${TRAINING_LOG}"
echo "  ✓ Training completed → ${TRAINING_LOG}"
echo ""

# ---- Step 3: Convert weights ----
echo "[Step 3/5] Converting weights to quantized format..."
cp "${SPK_DICT_PATH}" "${FINAL_MODEL}/"
python "${CONVERT_SCRIPT}" \
    --quant_config "${QUANT_CONFIG_PATH}" \
    --unquantized_model "${FINAL_MODEL}" \
    --quantized_model "${EVAL_MODEL}" \
    --dtype bfloat16
echo "  ✓ Weight quantization completed"
echo ""

# ---- Step 4: Verify quantized weights ----
echo "[Step 4/5] Verifying quantized weight equivalence..."
python "${TEST_SCRIPT}" \
    --quant_config "${QUANT_CONFIG_PATH}" \
    --original_model "${FINAL_MODEL}" \
    --quantized_model "${EVAL_MODEL}" \
    --dtype bfloat16 \
    --rtol 1e-2 \
    --atol 1e-2
echo "  ✓ Equivalence test completed"
echo ""

# ---- Step 5: Evaluate ----
echo "[Step 5/5] You need manually run evaluation on the 4-bit model."

# ============================================================================
# Pipeline Summary
# ============================================================================
echo "================================================================================"
echo "Pipeline Completed Successfully!"
echo "================================================================================"
echo "Model:           ${MODEL_NAME}"
echo "Quantization:    ${QUANT_CONFIG}"
echo "All outputs:     ${TRAIN_ROOT}"
echo "  Final model:              ${FINAL_MODEL}"
echo "  Quantized model (A16KV16): ${EVAL_MODEL}"
echo "  Quantized model (A8KV8):   ${EVAL_MODEL_A8KV8}"
echo "  Training logs:            ${TRAIN_LOG_DIR}"
echo "  TensorBoard:              ${TENSORBOARD_DIR}"
echo "  Results:                  ${RESULT_DIR}"
echo "================================================================================"
echo ""
