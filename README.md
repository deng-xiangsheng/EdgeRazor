![EdgeRazor Title](./asset/Title.png)

**EdgeRazor** is a unified, lightweight framework for edge AI that seamlessly integrates model compression techniques into existing full-precision training pipelines with minimal code modification. It is designed to produce models that are smaller, faster, and readily deployable across diverse hardware targets, ranging from mobile devices and embedded systems to resource-constrained edge endpoints and latency-sensitive cloud clusters, while preserving promising task performance.

EdgeRazor currently focuses on low-bit LLM compression via configurable quantization-aware distillation. On the **quantization** side, it supports quantizing weights (including embedding and lm_head layers), activations, and the KV cache. Supported bit-widths include uniform 4-bit and 1.58-bit, as well as matrix-wise mixed-precision configurations such as 2.79-bit (50% 4-bit + 50% 1.58-bit) and 1.88-bit (12.5% 4-bit + 87.5% 1.58-bit). On the **distillation** side, it offers logits, feature, and attention distillation, all of which can be flexibly combined within a unified configuration interface.

EdgeRazor achieves state-of-the-art compression–accuracy trade-offs across a range of LLMs. Taking Qwen3-0.6B as an example, **Qwen3-0.6B-EdgeRazor** attains benchmark scores of **47.80 / 44.10 / 41.76 / 39.81** at 4-bit / 2.79-bit / 1.88-bit / 1.58-bit performance, corresponding to compression ratios of **3.94× / 5.05× / 6.40× / 7.03×**, respectively. In comparison, the best prior methods achieve only 45.74 / 37.38 / 30.49 at 4-bit / 3-bit / 2-bit with compression ratios of 2.21× / 2.47× / 2.78×, demonstrating that EdgeRazor delivers substantially better performance with significantly greater compression.

<p align="center">
  <img src="./asset/Architeacture.png" alt="EdgeRazor Architecture">
  <br>
  <em>Figure: EdgeRazor framework with lightweight model training pipeline.</em>
</p>

## News

- [2026/02] 🔥 EdgeRazor V1 is released! Check our paper [here](...).
- [2025/10] 📝 TernaryCLIP is released! Check our paper [here](https://arxiv.org/abs/2510.21879).

## Contents

- [News](#news)
- [Contents](#contents)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Main Techniques](#main-techniques)
- [Applications](#applications)
- [Model Zoo](#model-zoo)
  - [LLM](#llm)
  - [MLLM](#mllm)
- [Citation](#citation)
- [Contributor List](#contributor-list)

## Getting Started

### Installation

```
git clone https://github.com/zhangsq-nju/EdgeRazor.git && cd EdgeRazor
conda create -n edgerazor python=3.10 -y
conda activate edgerazor
pip install -e .[cu128]
```

### Usage

1. Use unified configuration by [yaml](./example/configs/qad/qat_w4_a8_kd_fd.yaml), [json](./example/configs/qad/qat_w4_a8_kd_fd.json) or [dict](./example/configs/qad/qat_w4_a8_kd_fd.py).

2. Seamlessly integrate EdgeRazor into your FULL-PRECISION model training and enjoy your lightweight journey!

```python
# Init EdgeRazor for lightweight model
edgerazor = EdgeRazor(config="/path/to/config.yaml")
student = edgerazor.quantize(student)
# Training loop
student_outputs = student(inputs)
teacher_outputs = teacher(inputs)
# Calculate loss
loss, loss_dict = edgerazor.compute_loss(student_outputs, teacher_outputs, labels)
```

## Main Techniques

- Quantization-aware distillation, QAD
  - Configurable quantization-aware training for weights, activations, and KV cache
  - Customizable knowledge distillation pipelines between 16-bit and N-bit LLMs
- Pruning (Work in Progress)

## Applications

- Lightweight ViT-S/16 on MNIST, check [here](./example/vit/README.md).
- Lightweight ResNet-18 on MNIST, check [here](./example/resnet/README.md).
- Lightweight Qwen3-0.6B/1.7B, check [here](./example/qwen3/README.md).
- Lightweight MobileLLM-ParetoQ-1.5B-BF16, check [here](./example/mobilellm/README.md).
- Lightweight Qwen2.5-Omni-7B, check [here](./example/qwen2_5-omni/README.md).

## Model Zoo

### LLM

- Avg. Performance: average of performance scores in multiple tasks with [lm-eval v0.4.9.1](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.1).
  - Instruct model tasks: arc_easy, arc_challenge, hellaswag, boolq, social_iqa, openbookqa, piqa, winogrande, truthfulqa_mc2, hendrycks_ethics, mmlu, gsm8k, humaneval_instruct, ifeval.
  - Base model tasks: arc_easy, arc_challenge, hellaswag, boolq, social_iqa, openbookqa, piqa, winogrande, truthfulqa_mc2, hendrycks_ethics, mmlu, gsm8k, humaneval.
  - Except for 5-shot gsm8k, all other tasks are zero-shot.

- Hub Link: EdgeRazor indicates the original quantized checkpoints. We also transfer the checkpoints into GGUF ([llama.cpp](https://github.com/ggml-org/llama.cpp)) and GPTQ ([GPTQModel](https://github.com/ModelCloud/GPTQModel)) formats if compatible.

| Model          | Quantization | Group Size | Avg. Performance | Hub Link                                                            |
| -------------- | ------------ | ---------- | ---------------- | ------------------------------------------------------------------- |
| Qwen3-0.6B     | W16-A16-KV16 | -          | 47.35            | [Base](https://huggingface.co/Qwen/Qwen3-0.6B)                      |
| Qwen3-0.6B     | W4-A8-KV8    | 256        | 47.80            | EdgeRazor\|GGUF\|GPTQ                                               |
| Qwen3-0.6B     | W2.79-A8-KV8 | 256        | 44.10            | EdgeRazor                                                           |
| Qwen3-0.6B     | W1.88-A8-KV8 | 256        | 41.76            | EdgeRazor                                                           |
| Qwen3-0.6B     | W1.58-A8-KV8 | 256        | 39.81            | EdgeRazor\|GGUF\|GPTQ                                               |
| Qwen3-1.7B     | W16-A16-KV16 | -          | 58.65            | [Base](https://huggingface.co/Qwen/Qwen3-1.7B)                      |
| Qwen3-1.7B     | W4-A8-KV8    | 256        | 58.57            | EdgeRazor\|GGUF\|GPTQ                                               |
| Qwen3-1.7B     | W2.79-A8-KV8 | 256        | 53.00            | EdgeRazor                                                           |
| Qwen3-1.7B     | W1.88-A8-KV8 | 256        | 47.14            | EdgeRazor                                                           |
| Qwen3-1.7B     | W1.58-A8-KV8 | 256        | 43.91            | EdgeRazor\|GGUF\|GPTQ                                               |
| MobileLLM-350M | W16-A16-KV16 | -          | 41.18            | [Base](https://huggingface.co/facebook/MobileLLM-ParetoQ-350M-BF16) |
| MobileLLM-350M | W4-A8-KV8    | 256        | 41.86            | EdgeRazor\|GGUF\|GPTQ                                               |
| MobileLLM-350M | W2.79-A8-KV8 | 64         | 40.62            | EdgeRazor                                                           |
| MobileLLM-350M | W1.88-A8-KV8 | 64         | 39.02            | EdgeRazor                                                           |
| MobileLLM-350M | W1.58-A8-KV8 | 64         | 38.12            | EdgeRazor\|GGUF\|GPTQ                                               |

### MLLM

- Video-MME and MLVU are video understanding tasks with [lmms-eval v0.5](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/v0.5).

| Model           | Quantization | Group Size | Video-MME | MLVU  | Hub Link              |
| --------------- | ------------ | ---------- | --------- | ----- | --------------------- |
| Qwen2.5-Omni-7B | W16-A16-KV16 | -          | 62.81     | 48.01 | Base                  |
| Qwen2.5-Omni-7B | W4-A16-KV16  | 32         | 62.22     | 48.82 | EdgeRazor\|GGUF\|GPTQ |

## Citation

If you find our papar and code useful in your research, please consider giving a star ⭐️ and kindly cite our paper ✏️:

```
@article{zhangsh-ternaryclip,
  title={{TernaryCLIP}: Efficiently Compressing Vision-Language Models with Ternary Weights and Distilled Knowledge}, 
  author={Shu-Hao Zhang and Wei-Cheng Tang and Chen Wu and Peng Hu and Nan Li and Liang-Jie Zhang and Qi Zhang and Shao-Qun Zhang},
  year={2025},
  journal={arXiv preprint arXiv:2510.21879}
}
```

## Contributor List

- Shu-Hao Zhang: Core developer and maintainer of EdgeRazor-V1.