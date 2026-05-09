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

RESULT_DIR=results
TASK_NAME=EdgeRazor_Eval_QLLM
TASK_NAME_CHAT=EdgeRazor_Eval_QLLM_Chat
QUANT_EVAL=qwen3-1.7b-fp
RESULT_DIR_SUB=$RESULT_DIR/$QUANT_EVAL

EVAL_MODEL=Qwen/Qwen3-1.7B

mkdir -p $RESULT_DIR
mkdir -p $RESULT_DIR_SUB

EVAL_LOG=$RESULT_DIR_SUB/eval_${QUANT_EVAL}_${timestamp}.log
mkdir -p $RESULT_DIR_SUB
time {
accelerate launch -m lm_eval --model hf \
  --model_args pretrained=$EVAL_MODEL \
  --tasks $TASK_NAME --log_samples --batch_size 32 \
  --output_path $RESULT_DIR_SUB/$TASK_NAME \
  --confirm_run_unsafe_code;
} 2>&1 | tee $EVAL_LOG

time {
accelerate launch -m lm_eval --model hf \
  --model_args pretrained=$EVAL_MODEL --apply_chat_template \
  --tasks $TASK_NAME_CHAT --log_samples --batch_size 32 \
  --output_path $RESULT_DIR_SUB/$TASK_NAME_CHAT \
  --confirm_run_unsafe_code;
} 2>&1 | tee $EVAL_LOG