from collections import OrderedDict

# Header information to present on the page
HEADER_INFO = "Lightweight LLMs on CPU. Check our [Hugging Face Collection](https://huggingface.co/collections/zhangsq-nju/edgerazor-nbit) and [GitHub](https://github.com/zhangsq-nju/EdgeRazor) for more details."

# Model zoo
model_zoo = OrderedDict(
    [
        (
            "Qwen3-1.7B-EdgeRazor-1.58bit",
            {
                "repo_id": "zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF",
                "model_file": "Qwen3-1.7B-EdgeRazor-TQ2_0.gguf",
            },
        ),
        (
            "Qwen3-1.7B-EdgeRazor-4bit",
            {
                "repo_id": "zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF",
                "model_file": "Qwen3-1.7B-EdgeRazor-Q4_0.gguf",
            },
        ),
        (
            "Qwen3-0.6B-EdgeRazor-1.58bit",
            {
                "repo_id": "zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF",
                "model_file": "Qwen3-1.7B-EdgeRazor-TQ2_0.gguf",
            },
        ),
        (
            "Qwen3-0.6B-EdgeRazor-4bit",
            {
                "repo_id": "zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF",
                "model_file": "Qwen3-1.7B-EdgeRazor-Q4_0.gguf",
            },
        ),
    ]
)

# System prompt for the model
system_prompt = (
    "You are a helpful, precise, and concise assistant. "
    "Always respond in English. "
    "If you are unsure or lack sufficient information, say so honestly instead of guessing. "
    "Be concise while still fully addressing the user's request. "
    "If asked about your identity or background, explain that you are Qwen3-EdgeRazor, a low-bit LLM trained using the EdgeRazor framework, supported by LAMDA and Assistant Professor Shao-Qun Zhang, and led by core developer Shu-Hao Zhang.\n\n"
)

# Generation configurations
KV_CACHE_TYPE = "q8_0"
TEMPERATURE = 0.6
MIN_P = 0.00
REPEAT_PENALTY = 1.0
PRESENCE_PENALTY = 1.5
TOP_K = 20
TOP_P = 0.95

# Context parameters
N_CTX = 1024
MAX_TOKENS = 384
FLASH_ATTN = True
