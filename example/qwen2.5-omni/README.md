# Qwen2.5-Omni-7B-EdgeRazor-4bit

- Install [EdgeRazor](../../EdgeRazor/README.md) and required packages `pip install -r requirements.txt`
- Download and distill data from HuggingFaceM4/TGIF: [here](../data-for-qllm/src/omni/distill_qwen2.5-omni.sh)
- Run training: [here](./script/qwen2_5_omni-7B/run.sh)
- Evaluate: [here](../../src/eval/tasks/lmms-eval/models/qwen2_5_omni.sh)