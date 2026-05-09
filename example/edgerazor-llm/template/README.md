## LLM Transformers Templates

In this directory, we add EdgeRazor quantization functions into Qwen3 implementation codes in transformers, including weight and activation quantization.

- `config.json`: add `quant_mode` and `is_w_quantized`
- `configurations_qwen3.py`: add operation of `config.json`
- `modeling_qwen3.py`: add quantization implementation using EdgeRazor

## quant_mode

`_embint4`: decoder layer with mixed-precision quantization and embedding&lm_head with int4 quantization

- `w1_58a8kv8_embint4_{model}`: int4_prop=0
- `w1_88a8kv8_embint4_{model}`: int4_prop=12.5
- `w2_79a8kv8_embint4_{model}`: int4_prop=50
- `w4a8kv8_{model}`: int4_prop=100

model choice:

- `qwen3`
- `mobilellm`