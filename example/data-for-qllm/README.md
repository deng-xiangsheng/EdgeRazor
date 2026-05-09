## Usage

Scripts are in `./src` directory:

- `cache_old_datasets.sh`: some datasets employ lower datasets version
- `data_prepare.sh`: download and reformat datasets for text LLMs
- `distill_qwen3.sh`: distill datasets from Qwen/Qwen3-0.6B
- `distill_qwenomni.sh`: distill datasets from Qwen/Qwen2.5-Omni-7B, used by 4-bit Qwen2.5-Omni-7B-EdgeRazor targeting video understanding task
- `omni/distill_qwen2.5-omni.sh`: download gif from url, convert to mp4, distill video understanding reponse, and collect the teacher-distilled dataset

## Datasets for EdgeRazor-LLMs

This repository contains datasets used in EdgeRazor framework to obtain diverse low-bit models, including base, instruction-tuned, and multimodal LLMs. The file type is all converted to jsonl and maintains the format of prompts and responses.

| Filenames                  | Datasets                                 | Subsets         | Split | Data Sizes |
| -------------------------- | ---------------------------------------- | --------------- | ----- | ---------- |
| ii_7M_instruct.jsonl       | BAAI/Infinity-Instruct                   | 7M_domains      | train | 7.45M      |
| ii_gen_1.4M_instruct.jsonl | BAAI/Infinity-Instruct                   | Gen             | train | 1.46M      |
| tulu_0.6M_instruct.jsonl   | allenai/tulu-v3.1-mix-preview-4096-OLMoE | –               | train | 0.61M      |
| am_1.4M_instruct.jsonl     | a-m-team/AM-DeepSeek-R1-Distilled-1.4M   | am_0.5M+am_0.9M | train | 1.40M      |
| task_0.2M_instruct.jsonl   | Mixed Downstream Dataset                 | –               | train | 0.24M      |
| ii_1.5M_base.jsonl         | BAAI/Infinity-Instruct                   | 7M_core         | train | 1.48M      |
| tgif_10k_distilled.jsonl   | HuggingFaceM4/TGIF                       | –               | train | 10K        |

* `_instruct` datasets is for instruct LLMs such as Qwen3-0.6B/1.7B. 
* `_base` datasets is for base LLMs such as MobileLLM-350M.
* `_distilled` datasets is distilled from the 16-bit teacher model, while other datasets comprise human-annotated and externally distilled data.
* Mixed downstream dataset contains: allenai/ai2_arc, Rowan/hellaswag, super_glue.boolq, baber/piqa, winogrande, social_i_qa, openbookqa, EleutherAI/hendrycks_ethics
