# Run and exactly reproduce qwen2.5-omni results!
# mme as an example
export HF_HOME="~/.cache/huggingface"
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
# pip3 install qwen_vl_utils
# use `interleave_visuals=True` to control the visual token position, currently only for mmmu_val and mmmu_pro (and potentially for other interleaved image-text tasks), please do not use it unless you are sure about the operation details.

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false

# --------------------------------------------------------------------------

BATCH_SIZE=1
TASKS=(mlvu_test videomme)

# format: model_name|pretrained_path|output_dir|use_cli_trust_remote_code
MODELS=(
    "BF16|/path/to/Qwen2.5-Omni-7B|/path/to/qwen2_5_omni_vllm_bf16|true"
    "EdgeRazor 4-bit|/path/to/Qwen2.5-Omni-7B-EdgeRazor-4bit|/path/to/qwen2_5_omni_vllm_edgerazor_4bit|true"
    "GPTQ 4-bit|/path/to/Qwen2.5-Omni-7B-GPTQ-Int4|/path/to/qwen2_5_omni_vllm_gptq_4bit|false"
    "AWQ 4-bit|/path/to/Qwen2.5-Omni-7B-AWQ|/path/to/qwen2_5_omni_vllm_awq_4bit|false"
)

echo "========== Qwen2.5-Omni Batch Eval Start =========="
echo "Models: BF16, EdgeRazor 4-bit, GPTQ 4-bit, AWQ 4-bit"
echo "Tasks: ${TASKS[*]}"
echo "Batch size: ${BATCH_SIZE}"

for model_cfg in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_NAME PRETRAINED OUTPUT_DIR USE_CLI_TRUST <<< "$model_cfg"
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "--------------------------------------------------"
    echo "[Model] ${MODEL_NAME}"
    echo "[Path] ${PRETRAINED}"
    echo "[Output] ${OUTPUT_DIR}"

    for task in "${TASKS[@]}"; do
        echo "[Run] model=${MODEL_NAME}, task=${task}"
        if [ "$USE_CLI_TRUST" = "true" ]; then
            accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval --model vllm --trust_remote_code \
                --model_args model=$PRETRAINED,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_num_seqs=64,trust_remote_code=True \
                --tasks "$task" \
                --log_samples --output_path "$OUTPUT_DIR/$task" \
                --batch_size "$BATCH_SIZE"
        else
            accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval --model vllm \
                --model_args model=$PRETRAINED,tensor_parallel_size=1,gpu_memory_utilization=0.9,max_num_seqs=64,trust_remote_code=True \
                --tasks "$task" \
                --log_samples --output_path "$OUTPUT_DIR/$task" \
                --batch_size "$BATCH_SIZE"
        fi
        echo "[Done] model=${MODEL_NAME}, task=${task}"
    done

    echo "[Done] model=${MODEL_NAME} all tasks finished"
done

echo "========== Qwen2.5-Omni Batch Eval Finished =========="