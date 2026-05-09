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
# EdgeRazor-QLLM Training & Evaluation Pipeline — MobileLLM-350M
################################################################################
# Usage: modify the User Configuration section below, then run:
#   bash run.sh
################################################################################

# ============================================================================
# User Configuration (ONLY modify this section for different experiments)
# ============================================================================

# ---- Environment ----
PATH_PREFIX="/path/to/your/environment"   # Root path for code/models/data

# ---- Model ----
MODEL_NAME="facebook/MobileLLM-ParetoQ-350M-BF16"
MODEL_SHORT="MobileLLM-350M"              # Short name for template/filesystem paths

# ---- Quantization ----
QUANT_CONFIG="W4A8KV8"                    # W1.58A8KV8 | W1.88A8KV8 | W2.79A8KV8 | W4A8KV8
TRAIN_YAML="train"                        # Config file: ./train.yaml  (change to train_1, train_2, ...)

# ---- Experiment ----
RUN_NAME="mobilellm-350m_w4a8kv8"         # Experiment tag (written to config.py)
TRAIN_VERSION="train"                     # Output subdirectory name

# ---- Optional ----
BACKUP_NFS=false                          # Set to true to backup results to NFS
NFS_BASE="/path/to/nfs/backup"            # NFS backup root directory

# ---- AML keep-alive (container environments) ----
AML_KEEP_ALIVE=false                      # Set to true to call aml.sh after pipeline

# ============================================================================
# Derived Paths (AUTO-GENERATED — DO NOT MODIFY)
# ============================================================================

TEMPLATE_NAME="${MODEL_SHORT}-${QUANT_CONFIG}-Template"

CODE_ROOT="${PATH_PREFIX}/code/EdgeRazor-QLLM"
TEMPLATE_PATH="${CODE_ROOT}/template/${MODEL_SHORT}/${TEMPLATE_NAME}"

TRAIN_ROOT="${CODE_ROOT}/${TRAIN_VERSION}"
FINAL_MODEL="${TRAIN_ROOT}/final_model"
EVAL_MODEL="${TRAIN_ROOT}/${MODEL_SHORT}-EdgeRazor"

TRAIN_LOG_DIR="${TRAIN_ROOT}/logs"
TENSORBOARD_DIR="${TRAIN_ROOT}/tensorboard"
RESULT_DIR="${TRAIN_ROOT}/results"

QUANT_CONFIG_PATH="${CODE_ROOT}/src/${TRAIN_YAML}.yaml"
CONVERT_SCRIPT="${CODE_ROOT}/src/convert/convert_qweight.py"
TEST_SCRIPT="${CODE_ROOT}/src/convert/test_qweight_equivalence.py"

NFS_DIR="${NFS_BASE}/${RUN_NAME}"

export PATH="${HOME}/.conda/envs/edgerazor/bin:${PATH}"

# ============================================================================
# Pre-flight
# ============================================================================

# Copy local train config to source tree, set experiment tag
cp "./${TRAIN_YAML}.yaml" "${QUANT_CONFIG_PATH}"
sed -i.bak "s/= \"exp_for_sed\"/= \"${RUN_NAME}\"/" "${CODE_ROOT}/src/config.py"

# Determine config class based on MODEL_NAME and inject into main.py
case "${MODEL_NAME}" in
    "Qwen/Qwen3-0.6B")
        CONFIG_CLASS="EdgeRazorTrainConfigForQwen3_0_6B"
        ;;
    "Qwen/Qwen3-1.7B")
        CONFIG_CLASS="EdgeRazorTrainConfigForQwen3_1_7B"
        ;;
    "facebook/MobileLLM-ParetoQ-350M-BF16")
        CONFIG_CLASS="EdgeRazorTrainConfigForMobileLLM_350M"
        ;;
    *)
        echo "ERROR: Unknown MODEL_NAME: ${MODEL_NAME}"
        exit 1
        ;;
esac
sed -i.bak "s/config = EdgeRazorTrainConfigFor[^(]*()/config = ${CONFIG_CLASS}()/" "${CODE_ROOT}/src/main.py"


# ============================================================================
# Pipeline Execution
# ============================================================================

echo "================================================================================"
echo "EdgeRazor-QLLM Training & Evaluation Pipeline — MobileLLM-350M"
echo "================================================================================"
echo "Model:           ${MODEL_NAME}"
echo "Quantization:    ${QUANT_CONFIG}"
echo "Experiment:      ${RUN_NAME}"
echo "Train config:    ${TRAIN_YAML}.yaml"
echo "Output:          ${TRAIN_ROOT}"
echo "================================================================================"
echo ""

# ---- Step 1: Prepare environment ----
echo "[Step 1/6] Preparing environment..."
mkdir -p "${TRAIN_LOG_DIR}" "${TENSORBOARD_DIR}" "${RESULT_DIR}"
cp -rf "${TEMPLATE_PATH}" "${EVAL_MODEL}"
echo "  ✓ Output directories created"
echo "  ✓ Template copied to: ${EVAL_MODEL}"
echo ""

# ---- Step 2: Train ----
echo "[Step 2/6] Starting distributed training with DeepSpeed..."
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
TRAINING_LOG="${TRAIN_LOG_DIR}/train_${timestamp}.log"
time {
    deepspeed --num_gpus=8 "${CODE_ROOT}/src/main.py"
} 2>&1 | tee "${TRAINING_LOG}"
echo "  ✓ Training completed → ${TRAINING_LOG}"
echo ""

# ---- Step 3: Convert weights ----
echo "[Step 3/6] Converting weights to quantized format..."
python "${CONVERT_SCRIPT}" \
    --quant_config "${QUANT_CONFIG_PATH}" \
    --unquantized_model "${FINAL_MODEL}" \
    --quantized_model "${EVAL_MODEL}" \
    --dtype bfloat16
echo "  ✓ Weight quantization completed"
echo ""

# ---- Step 4: Verify quantization ----
echo "[Step 4/6] Verifying quantized weight equivalence..."
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
echo "[Step 5/6] Running benchmark evaluation..."
TASK_NAME="EdgeRazor_Eval_QLLM_Base"
TASK_NAME_CHAT="EdgeRazor_Eval_QLLM_Base_Chat"

# (a) A16KV16 — standard HF modeling, no trust_remote_code
QUANT_EVAL="a16kv16"
RESULT_DIR_SUB="${RESULT_DIR}/${QUANT_EVAL}"
EVAL_LOG="${RESULT_DIR_SUB}/eval_${QUANT_EVAL}_${timestamp}.log"
mkdir -p "${RESULT_DIR_SUB}"

time {
    accelerate launch -m lm_eval --model hf \
        --model_args pretrained="${EVAL_MODEL}" \
        --tasks "${TASK_NAME}" --log_samples --batch_size 128 \
        --output_path "${RESULT_DIR_SUB}/${TASK_NAME}" \
        --confirm_run_unsafe_code
} 2>&1 | tee "${EVAL_LOG}"

time {
    accelerate launch -m lm_eval --model hf \
        --model_args pretrained="${EVAL_MODEL}" --apply_chat_template \
        --tasks "${TASK_NAME_CHAT}" --log_samples --batch_size auto \
        --output_path "${RESULT_DIR_SUB}/${TASK_NAME_CHAT}" \
        --confirm_run_unsafe_code
} 2>&1 | tee -a "${EVAL_LOG}"

# (b) A8KV8 — custom quantization kernels, trust_remote_code=True
QUANT_EVAL="a8kv8"
RESULT_DIR_SUB="${RESULT_DIR}/${QUANT_EVAL}"
EVAL_LOG="${RESULT_DIR_SUB}/eval_${QUANT_EVAL}_${timestamp}.log"
mkdir -p "${RESULT_DIR_SUB}"

time {
    accelerate launch -m lm_eval --model hf \
        --model_args pretrained="${EVAL_MODEL}",trust_remote_code=True \
        --tasks "${TASK_NAME}" --log_samples --batch_size 128 \
        --output_path "${RESULT_DIR_SUB}/${TASK_NAME}" \
        --confirm_run_unsafe_code --trust_remote_code
} 2>&1 | tee "${EVAL_LOG}"

time {
    accelerate launch -m lm_eval --model hf \
        --model_args pretrained="${EVAL_MODEL}",trust_remote_code=True --apply_chat_template \
        --tasks "${TASK_NAME_CHAT}" --log_samples --batch_size auto \
        --output_path "${RESULT_DIR_SUB}/${TASK_NAME_CHAT}" \
        --confirm_run_unsafe_code --trust_remote_code
} 2>&1 | tee -a "${EVAL_LOG}"

echo "  ✓ Evaluation completed → ${RESULT_DIR}"
echo ""

# ---- Step 6: AML keep-alive + NFS backup ----
if [ "${AML_KEEP_ALIVE}" = true ]; then
    echo "[Step 6/6] AML keep-alive..."
    bash aml.sh
fi

if [ "${BACKUP_NFS}" = true ]; then
    echo "[Step 6/6] Backing up results to NFS..."
    mkdir -p "${NFS_DIR}"
    cp -rf "${TRAIN_ROOT}" "${NFS_DIR}/"
    echo "  ✓ Backup completed → ${NFS_DIR}"
else
    echo "[Step 6/6] NFS backup skipped (set BACKUP_NFS=true to enable)"
fi
echo ""

# ============================================================================
# Pipeline Summary
# ============================================================================
echo "================================================================================"
echo "Pipeline Completed Successfully!"
echo "================================================================================"
echo "Model:           ${MODEL_NAME}"
echo "Quantization:    ${QUANT_CONFIG}"
echo "All outputs:     ${TRAIN_ROOT}"
echo "  Final model:   ${FINAL_MODEL}"
echo "  Eval model:    ${EVAL_MODEL}"
echo "  Training logs: ${TRAIN_LOG_DIR}"
echo "  TensorBoard:   ${TENSORBOARD_DIR}"
echo "  Results:       ${RESULT_DIR}"
if [ "${BACKUP_NFS}" = true ]; then
echo "  NFS Backup:    ${NFS_DIR}"
fi
echo "================================================================================"
