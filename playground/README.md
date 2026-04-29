---
title: EdgeRazor Playground
emoji: 🚀
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.5.1
python_version: 3.12.2
app_file: app.py
pinned: true
license: apache-2.0
---

## EdgeRazor Playground

A CPU-friendly chatbot powered by **[Qwen3-EdgeRazor-nbit](https://huggingface.co/collections/zhangsq-nju/edgerazor-nbit)**, running locally via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). Displays real-time efficiency metrics (output tokens, time, decoding throughput) per turn.

## Dependencies

- [llama-cpp-python](https://abetlen.github.io/llama-cpp-python/whl/cpu/llama-cpp-python)
- Qwen3-EdgeRazor gguf files:
  - [Qwen3-0.6B-EdgeRazor-GGUF](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-GGUF)
  - [Qwen3-1.7B-EdgeRazor-GGUF](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF)