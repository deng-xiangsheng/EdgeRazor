#!/bin/bash
# ============================================================================
# Qwen2.5-Omni-7B Template Generator
# ============================================================================
# Generates the qwen2_5_omni_7B_edgerazor template directory from:
#   1. Template/                — custom .py files + base config.json + __init__.py
#   2. Qwen/Qwen2.5-Omni-7B (HF) — tokenizer, generation_config, README, LICENSE, etc.
#
# Output directory:
#   qwen2_5_omni_7B_edgerazor/
#
# Usage:
#   bash generate_qwen2.5-omni-7b_template.sh
# ============================================================================

set -e

# ============================================================================
# User Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_NAME="qwen2_5_omni_7B_edgerazor"       # Output directory name
HF_MODEL_ID="Qwen/Qwen2.5-Omni-7B"            # HuggingFace model ID

# Source: custom EdgeRazor files
TEMPLATE_BASE="${SCRIPT_DIR}/Template"

# ---- Files from Template/ (copied as-is) ----
TEMPLATE_FILES=(
    "__init__.py"
    "config.json"
    "configuration_qwen2_5_omni.py"
    "modeling_qwen2_5_omni.py"
    "modular_qwen2_5_omni.py"
    "processing_qwen2_5_omni.py"
    "README.md"
)

# ---- Files from HuggingFace (downloaded once, reused for all variants) ----
HF_FILES=(
    "added_tokens.json"
    "chat_template.json"
    "generation_config.json"
    "LICENSE"
    "merges.txt"
    "model.safetensors.index.json"
    "preprocessor_config.json"
    "special_tokens_map.json"
    "spk_dict.pt"
    "tokenizer.json"
    "tokenizer_config.json"
    "vocab.json"
)

# ============================================================================
# Pre-flight checks
# ============================================================================

if [ ! -d "${TEMPLATE_BASE}" ]; then
    echo "ERROR: Template/ directory not found: ${TEMPLATE_BASE}"
    exit 1
fi

for f in "${TEMPLATE_FILES[@]}"; do
    if [ ! -f "${TEMPLATE_BASE}/${f}" ]; then
        echo "ERROR: ${f} not found in Template/"
        exit 1
    fi
done

# ============================================================================
# Step 1: Download base files from HuggingFace (once, shared by all variants)
# ============================================================================

HF_CACHE_DIR="${SCRIPT_DIR}/.hf_cache_qwen2_5_omni"

echo "================================================================================"
echo "Qwen2.5-Omni-7B Template Generator"
echo "================================================================================"
echo "Model:           ${HF_MODEL_ID}"
echo "Output:          ${MODEL_NAME}"
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
        path = hf_hub_download(repo_id=repo, filename=f, local_dir=cache)
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
# Step 2: Generate template directory
# ============================================================================

echo "[2/3] Generating template directory..."
echo ""

TARGET_DIR="${SCRIPT_DIR}/${MODEL_NAME}"

echo "  --- ${MODEL_NAME} ---"

mkdir -p "${TARGET_DIR}"

# 2a. Copy custom files from Template/
for f in "${TEMPLATE_FILES[@]}"; do
    cp "${TEMPLATE_BASE}/${f}" "${TARGET_DIR}/${f}"
done

# 2b. Copy HuggingFace base files
for f in "${HF_FILES[@]}"; do
    cp "${HF_CACHE_DIR}/${f}" "${TARGET_DIR}/${f}"
done

echo ""

# ============================================================================
# Step 3: Summary
# ============================================================================

echo "[3/3] Done — generated template directory:"
echo ""
echo "  ${TARGET_DIR}/"
echo "    ├── __init__.py"
echo "    ├── config.json"
echo "    ├── configuration_qwen2_5_omni.py"
echo "    ├── modeling_qwen2_5_omni.py"
echo "    ├── modular_qwen2_5_omni.py"
echo "    ├── processing_qwen2_5_omni.py"
echo "    ├── added_tokens.json"
echo "    ├── chat_template.json"
echo "    ├── generation_config.json"
echo "    ├── LICENSE"
echo "    ├── merges.txt"
echo "    ├── model.safetensors.index.json"
echo "    ├── preprocessor_config.json"
echo "    ├── README.md"
echo "    ├── special_tokens_map.json"
echo "    ├── spk_dict.pt"
echo "    ├── tokenizer.json"
echo "    ├── tokenizer_config.json"
echo "    └── vocab.json"
echo ""

echo "================================================================================"
echo "Tip: delete .hf_cache_qwen2_5_omni/ when no longer needed to save space."
echo "================================================================================"
