#!/bin/bash
# ============================================================================
# Qwen3-1.7B Template Generator
# ============================================================================
# Generates all W*A8KV8 template directories from:
#   1. Template/                — custom .py files + base config.json (with placeholder)
#   2. Qwen/Qwen3-1.7B (HF)    — tokenizer, generation_config, README, LICENSE, index
#
# Output directories:
#   Qwen3-1.7B-W1.58A8KV8-Template/
#   Qwen3-1.7B-W1.88A8KV8-Template/
#   Qwen3-1.7B-W2.19A8KV8-Template/
#   Qwen3-1.7B-W2.79A8KV8-Template/
#   Qwen3-1.7B-W4A8KV8-Template/
#
# Usage:
#   bash generate_qwen3-1.7b_template.sh
# ============================================================================

set -e

# ============================================================================
# User Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="Qwen3-1.7B"                    # Used for output directory naming
HF_MODEL_ID="Qwen/Qwen3-1.7B"              # HuggingFace model ID

# Source: custom EdgeRazor files
TEMPLATE_BASE="${SCRIPT_DIR}/Template"

# ---- Quantization variants ----
# Format: "DIR_SUFFIX|quant_mode_value"
QUANT_VARIANTS=(
    "W1.58A8KV8|w1_58a8kv8_embint4_qwen3"
    "W1.88A8KV8|w1_88a8kv8_embint4_qwen3"
    "W2.19A8KV8|w2_19a8kv8_qwen3"
    "W2.79A8KV8|w2_79a8kv8_embint4_qwen3"
    "W4A8KV8|w4a8kv8_qwen3"
)

# ---- Files from Template/ (copied as-is) ----
PY_FILES=(
    "configuration_qwen3.py"
    "modeling_qwen3.py"
    "modular_qwen3.py"
)

# ---- Files from HuggingFace (downloaded once, reused for all variants) ----
HF_FILES=(
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
    "merges.txt"
    "generation_config.json"
    "LICENSE"
    "README.md"
)

# ============================================================================
# Pre-flight checks
# ============================================================================

if [ ! -d "${TEMPLATE_BASE}" ]; then
    echo "ERROR: Template/ directory not found: ${TEMPLATE_BASE}"
    exit 1
fi

if [ ! -f "${TEMPLATE_BASE}/config.json" ]; then
    echo "ERROR: config.json not found in Template/"
    exit 1
fi

for f in "${PY_FILES[@]}"; do
    if [ ! -f "${TEMPLATE_BASE}/${f}" ]; then
        echo "ERROR: ${f} not found in Template/"
        exit 1
    fi
done

# ============================================================================
# Step 1: Download base files from HuggingFace (once, shared by all variants)
# ============================================================================

HF_CACHE_DIR="${SCRIPT_DIR}/.hf_cache"

echo "================================================================================"
echo "Qwen3-1.7B Template Generator"
echo "================================================================================"
echo "Model:           ${HF_MODEL_ID}"
echo "Variants:        ${#QUANT_VARIANTS[@]} (W1.58, W1.88, W2.19, W2.79, W4)"
echo "Template base:   ${TEMPLATE_BASE}"
echo "HF cache:        ${HF_CACHE_DIR}"
echo "================================================================================"
echo ""

if [ ! -f "${HF_CACHE_DIR}/tokenizer.json" ]; then
    echo "[1/3] Downloading base files from HuggingFace: ${HF_MODEL_ID} ..."
    mkdir -p "${HF_CACHE_DIR}"

    python -c "
import sys
from huggingface_hub import hf_hub_download

files = [$(printf '"%s",' "${HF_FILES[@]}" | sed 's/,$//')]
repo = '${HF_MODEL_ID}'
cache = '${HF_CACHE_DIR}'

for f in files:
    print(f'  Downloading {f} ... ', end='', flush=True)
    try:
        path = hf_hub_download(repo_id=repo, filename=f, local_dir=cache, local_dir_use_symlinks=False)
        print('OK')
    except Exception as e:
        print(f'FAILED: {e}')
        sys.exit(1)
print(f'  Done - downloaded {len(files)} files')
"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Failed to download from HuggingFace."
        echo "Make sure you have internet access and huggingface_hub is installed:"
        echo "  pip install huggingface_hub"
        exit 1
    fi
else
    echo "[1/3] Using cached HuggingFace files: ${HF_CACHE_DIR}"
fi
echo ""

# ============================================================================
# Step 2: Generate template directories for each quantization variant
# ============================================================================

echo "[2/3] Generating template directories..."
echo ""

for entry in "${QUANT_VARIANTS[@]}"; do
    SUFFIX="${entry%%|*}"
    QUANT_MODE="${entry##*|}"
    TARGET_DIR="${SCRIPT_DIR}/${MODEL_NAME}-${SUFFIX}-Template"

    echo "  --- ${MODEL_NAME}-${SUFFIX}-Template ---"

    mkdir -p "${TARGET_DIR}"

    # 2a. Copy custom .py files from Template/
    for f in "${PY_FILES[@]}"; do
        cp "${TEMPLATE_BASE}/${f}" "${TARGET_DIR}/${f}"
    done

    # 2b. Copy HuggingFace base files
    for f in "${HF_FILES[@]}"; do
        cp "${HF_CACHE_DIR}/${f}" "${TARGET_DIR}/${f}"
    done

    # 2c. Generate config.json — replace quant_mode placeholder
    sed "s/\"quant_mode\": \"change_here_for_different_quant_config\"/\"quant_mode\": \"${QUANT_MODE}\"/" \
        "${TEMPLATE_BASE}/config.json" > "${TARGET_DIR}/config.json"

    echo "      quant_mode = ${QUANT_MODE}"
    echo ""
done

# ============================================================================
# Step 3: Summary
# ============================================================================

echo "[3/3] Done — generated ${#QUANT_VARIANTS[@]} template directories:"
echo ""
for entry in "${QUANT_VARIANTS[@]}"; do
    SUFFIX="${entry%%|*}"
    TARGET="${SCRIPT_DIR}/${MODEL_NAME}-${SUFFIX}-Template"
    echo "  ${TARGET}/"
    echo "    ├── config.json              (quant_mode: ${entry##*|})"
    echo "    ├── configuration_qwen3.py"
    echo "    ├── modeling_qwen3.py"
    echo "    ├── modular_qwen3.py"
    echo "    ├── tokenizer.json"
    echo "    ├── tokenizer_config.json"
    echo "    ├── vocab.json"
    echo "    ├── merges.txt"
    echo "    ├── generation_config.json"
    echo "    ├── LICENSE"
    echo "    └── README.md"
    echo ""
done

echo "================================================================================"
echo "Tip: delete .hf_cache/ when no longer needed to save space."
echo "================================================================================"
