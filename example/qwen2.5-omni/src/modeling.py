# ruff: noqa: F401

import torch
from transformers import AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

from edgerazor import EdgeRazor

ATTENTION_CLASSES = ["eager", "flash_attention_2", "sdpa"]


def create_teacher_model_qllm(
    teacher_path=None,
    device_map="auto",
    attn_implementation="eager",  # make `output_attentions=True` work
    debug=True,
):
    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_implementation
    )

    # Make sure MoE model return router logits
    if isinstance(model, (OlmoeForCausalLM, )):
        model.config.output_router_logits = True
    model.eval()
    
    if debug:
        print("Teacher model config:")
        print(model.config)
        print("Teacher model architecture:")
        print(model)

    return model


def create_student_model_qllm(
    student_path=None,
    device_map="auto",
    attn_implementation="eager",  # make `output_attentions=True` work
    edgerazor: EdgeRazor=None,
    debug=True,
):
    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_implementation
    )
    
    # Make sure MoE model return router logits
    if isinstance(model, (OlmoeForCausalLM, )):
        model.config.output_router_logits = True
    model.enable_input_require_grads() # Set requires_grad = False -> True for the first layer embedding weights

    # EdgeRazor for student model
    if edgerazor is not None:
        qmodel = edgerazor.quantize(model)
    else:
        raise ValueError("edgerazor must be provided for student model quantization")
    
    if debug:
        print("Student qmodel config:")
        print(qmodel.config)
        print("Student qmodel architecture:")
        print(qmodel)
    
    return qmodel


def create_teacher_model(
    teacher_path=None,
    device_map="auto",
    attn_implementation="eager",  # make `output_attentions=True` work
    debug=True,
):
    # Create model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        teacher_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_implementation
    )
    
    # Ensure teacher model is in eval mode
    for param in model.parameters():
        param.requires_grad = False
    
    if debug:
        print("Teacher model config:")
        print(model.config)
        # print("Teacher model architecture:")
        # print(model)
    
    model.eval()
    
    return model

def create_student_model(
    student_path=None,
    device_map="auto",
    attn_implementation="eager",  # make `output_attentions=True` work
    edgerazor: EdgeRazor=None,
    debug=True,
):
    # Create model
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        student_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation=attn_implementation
    )

    # EdgeRazor for student model
    if edgerazor is not None:
        qmodel = edgerazor.quantize(model)
    else:
        raise ValueError("edgerazor must be provided for student model quantization")
    
    # Omni 4-bit Quant: Ensure only quantized weights require gradients
    for param in qmodel.parameters():
        param.requires_grad = False
    try:
        from edgerazor.qat import QLinear, QEmbedding

        for name, module in model.named_modules():
            if isinstance(module, (QLinear, QEmbedding)):
                if debug:
                    print(f"Quantized Module: {name}, Type: {type(module)}")
                module.weight.requires_grad = True
    except ImportError:
        print("edgerazor.qat module not found. Skipping setting requires_grad for quantized layers.")
    except Exception as e:
        print(f"An error occurred while setting requires_grad for quantized layers: {e}")
    
    if debug:
        print("Student qmodel config:")
        print(qmodel.config)
        # print("Student qmodel architecture:")
        # print(qmodel)
    
    qmodel.train()
    
    return qmodel