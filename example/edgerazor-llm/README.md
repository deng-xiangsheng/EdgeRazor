## Intall EdgeRazor

```bash
cd /path/to/EdgeRazor
pip install -e .
```

## Preparation

```bash
pip install -r requirements.txt
```

flash_attn can be installed via: pip install flash-attn --no-build-isolation

## File Structure

- `/path/to/your/environment/EdgeRazor-QLLM`: training code
- `/path/to/your/environment/EdgeRazor-QLLM/model`: output model weights
- `/path/to/your/environment/EdgeRazor-QLLM/data`: training data

## Prepare

1. You need to install EdgeRazor toolkit first
2. Prepare environment `bash ./prepare.sh`
   1. Run `./template/*/generate_xxx.sh` to prepare model templates
   2. Go to data directory to prepare datasets: [here](../../data/README.md)
3. Prepare customized evaluations for corresponding frameworks (lm_eval for LLMs, lmms-eval for multimodal LLMs)
   1. `/path/to/EdgeRazor/src/eval/tasks/lm_eval` directory:
      - replace `ifeval` to lm_eval task directory
      - add `edgerazor` to lm_eval task directory
      - Target task directory: `/path/to/lm-evaluation-harness/lm_eval/tasks/`
   2. `/path/to/EdgeRazor/src/eval/tasks/lmms-eval/models/qwen2_5_omni.sh`: install to `/path/to/lmms-eval/examples/models/qwen2_5_omni.sh`

## Run

| Model          | Note                                 | Script                                      |
| -------------- | ------------------------------------ | ------------------------------------------- |
| Qwen3-0.6B     | Qwen3-0.6B-EdgeRazor-4bit            | `./script/qwen3-0.6b/4-bit/run.sh`          |
|                | Qwen3-0.6B-EdgeRazor-2.79bit         | `./script/qwen3-0.6b/2.79-bit/run.sh`       |
|                | Qwen3-0.6B-EdgeRazor-1.88bit         | `./script/qwen3-0.6b/1.88-bit/run.sh`       |
|                | Qwen3-0.6B-EdgeRazor-1.58bit         | `./script/qwen3-0.6b/1.58-bit/run.sh`       |
| Qwen3-1.7B     | Qwen3-1.7B-EdgeRazor-4bit            | `./script/qwen3-1.7b/4-bit/run.sh`          |
|                | Qwen3-1.7B-EdgeRazor-2.79bit         | `./script/qwen3-1.7b/2.79-bit/run.sh`       |
|                | Qwen3-1.7B-EdgeRazor-1.88bit         | `./script/qwen3-1.7b/1.88-bit/run.sh`       |
|                | Qwen3-1.7B-EdgeRazor-1.58bit         | `./script/qwen3-1.7b/1.58-bit/run.sh`       |
| MobileLLM-350M | MobileLLM-350M-EdgeRazor-4bit        | `./script/mobilellm-350m/4-bit/run.sh`      |
|                | MobileLLM-350M-EdgeRazor-2.79bit     | `./script/mobilellm-350m/2.79-bit/run.sh`   |
|                | MobileLLM-350M-EdgeRazor-1.88bit     | `./script/mobilellm-350m/1.88-bit/run.sh`   |
|                | MobileLLM-350M-EdgeRazor-1.58bit     | `./script/mobilellm-350m/1.58-bit/run.sh`   |
| Ablation       | Qwen3-0.6B-EdgeRazor-2.79bit, SG+A+E | `./script/ablations/2.79-bit/SG+A+E/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-2.79bit, ST+A+E | `./script/ablations/2.79-bit/ST+A+E/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-2.19bit, SG+A+E | `./script/ablations/2.19-bit/SG+A+E/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-2.19bit, SG+A+C | `./script/ablations/2.19-bit/SG+A+C/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-2.19bit, SG+F+E | `./script/ablations/2.19-bit/SG+F+E/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-2.19bit, SG+F+F | `./script/ablations/2.19-bit/SG+F+F/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-1.88bit, SG+A+E | `./script/ablations/1.88-bit/SG+A+E/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-1.88bit, SG+A+C | `./script/ablations/1.88-bit/SG+A+C/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-1.88bit, SG+F+E | `./script/ablations/1.88-bit/SG+F+E/run.sh` |
|                | Qwen3-0.6B-EdgeRazor-1.88bit, SG+F+F | `./script/ablations/1.88-bit/SG+F+F/run.sh` |