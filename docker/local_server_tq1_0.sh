#!/usr/bin/env bash
set -euo pipefail

# Serving Qwen3-1.7B-EdgeRazor-GGUF model using llama.cpp's server in a Docker container
# Quantization: W1.58-A8-KV8

# Model
LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-"/path/to/Qwen3-1.7B-EdgeRazor-GGUF"}
LOCAL_MODEL_NAME=${LOCAL_MODEL_NAME:-"Qwen3-1.7B-EdgeRazor-TQ1_0.gguf"}

# Quant types for KV cache
KV_CACHE_TYPE=${KV_CACHE_TYPE:-q8_0}

# macOS environment
P_CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 10)
E_CORES=$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 4)
TOTAL_CORES=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 14)

THREADS=$((P_CORES - 2))
THREADS_BATCH=$((P_CORES - 1))

if [ "${THREADS}" -lt 1 ]; then THREADS=1; fi
if [ "${THREADS_BATCH}" -lt 1 ]; then THREADS_BATCH=1; fi

# Single-user server defaults
THREADS_HTTP=${THREADS_HTTP:-1}
N_PARALLEL=${N_PARALLEL:-1}

# Inference configurations
TEMPERATURE=${TEMPERATURE:-0.6}
MIN_P=${MIN_P:-0.00}
REPEAT_PENALTY=${REPEAT_PENALTY:-1.0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-1.5}
TOP_K=${TOP_K:-20}
TOP_P=${TOP_P:-0.95}
GENERATION_LENGTH=${GENERATION_LENGTH:-512}

# Server configurations
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8888}

echo "[local_server] model=${LOCAL_MODEL_NAME} port=${PORT} p_cores=${P_CORES} e_cores=${E_CORES} total=${TOTAL_CORES} threads=${THREADS} threads_batch=${THREADS_BATCH}"

exec docker run --rm \
    --ulimit memlock=-1:-1 \
    --cap-add IPC_LOCK \
    -v "${LOCAL_MODEL_DIR}":/models \
    -p "${PORT}:${PORT}" \
    ghcr.io/ggml-org/llama.cpp:server -m "/models/${LOCAL_MODEL_NAME}" \
    --port "${PORT}" --host "${HOST}" \
    --flash-attn "on" \
    --mlock \
    --threads "${THREADS}" \
    --threads-batch "${THREADS_BATCH}" \
    --threads-http "${THREADS_HTTP}" \
    --parallel "${N_PARALLEL}" \
    --cache-type-k "${KV_CACHE_TYPE}" \
    --cache-type-v "${KV_CACHE_TYPE}" \
    --n-gpu-layers 0 \
    --temp "${TEMPERATURE}" \
    --min-p "${MIN_P}" \
    --repeat-penalty "${REPEAT_PENALTY}" \
    --presence-penalty "${PRESENCE_PENALTY}" \
    --top-k "${TOP_K}" \
    --top-p "${TOP_P}" \
    -n "${GENERATION_LENGTH}"