# ruff: noqa: F401

import torch
from transformers import AutoModelForCausalLM
from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from edgerazor import EdgeRazor

ATTENTION_CLASSES = ["eager", "flash_attention_2", "sdpa"]


def create_teacher_model(
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


def create_student_model(
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
    model.enable_input_require_grads()

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
