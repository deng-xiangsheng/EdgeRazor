import torch.nn as nn
from transformers import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.olmoe.modeling_olmoe import (
    OlmoeAttention,
    OlmoeFlashAttention2,
    OlmoeSdpaAttention,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeAttention

from .block import (
    copy_llamaattention_to_qkvcache_llamaattention,
    copy_multiheadattention_to_qmultiheadattention,
    copy_olmoeattention_qkvcache_olmoeattention,
    copy_qwen2_5omniattention_to_qkvcache_qwen2_5omniattention,
    copy_qwen3attention_to_qkvcache_qwen3attention,
    copy_qwen3moeattention_to_qkvcache_qwen3moeattention,
)
from .module import (
    copy_conv1d_to_qconv1d,
    copy_conv2d_to_qconv2d,
    copy_conv3d_to_qconv3d,
    copy_embedding_to_qembedding,
    copy_linear_to_qlinear,
)
from .util import QuantConfig, QuantSelector


class ModuleSpecificQuantConfig:
    """
    Wrapper that provides module-specific QuantConfig with overrides applied.
    This allows each module to have its own effective configuration while
    maintaining the same interface as QuantConfig.
    """
    def __init__(self, base_config: QuantConfig, module_name: str, module_type: type[nn.Module]):
        self.base_config = base_config
        self.module_name = module_name
        self.module_type = module_type
        
        # Get the effective function config for this specific module
        self.function = base_config.get_function_config(module_name, module_type)
        
        # Pass through other attributes from base config
        self.method = base_config.method
        self.select = base_config.select
        self.training = base_config.training
        self.overrides = base_config.overrides


def apply_quantization(
    model: nn.Module,
    quant_config: QuantConfig,
    selector: QuantSelector,
    qlinear_cls: nn.Module = None,               # Quantized Linear class to replace with
    qembedding_cls: nn.Module = None,            # Quantized Embedding class to replace with (optional)
    qconv1d_cls: nn.Module = None,               # Quantized Conv1d class to replace with (optional)
    qconv2d_cls: nn.Module = None,               # Quantized Conv2d class to replace with (optional)
    qconv3d_cls: nn.Module = None,               # Quantized Conv3d class to replace with (optional)
    qmultiheadattention_cls: nn.Module = None,   # Quantized MultiheadAttention class to replace with (optional)
    qkvcacheolmoeattention_cls: nn.Module = None,  # Quantized QKVCacheOlmoeAttention class to replace with (optional)
    qkvcacheolmoeflashattention2_cls: nn.Module = None,  # Quantized QKVCacheOlmoeFlashAttention2 class to replace with (optional)
    qkvcacheolmoesdpaattention_cls: nn.Module = None,  # Quantized QKVCacheOlmoeSdpaAttention class to replace with (optional)
    qkvcacheqwen2_5omniattention_cls: nn.Module = None,  # Quantized QKVCacheQwen2_5OmniAttention class to replace with (optional)
    qkvcacheqwen3attention_cls: nn.Module = None,  # Quantized QKVCacheQwen3Attention class to replace with (optional)
    qkvcacheqwen3moeattention_cls: nn.Module = None,  # Quantized QKVCacheQwen3MoeAttention class to replace with (optional)
    qkvcachellamaattention_cls: nn.Module = None,  # Quantized QKVCacheLlamaAttention class to replace with (optional)
) -> nn.Module:
    """Apply quantization to the model"""

    def replace_structure(parent_module, child_name, new_module):
        """Replace submodule/block"""
        setattr(parent_module, child_name, new_module)
    
    # Execute replacements: block > module
    # 2. Collect modules to replace
    block_replacements = []

    for name, module in model.named_modules():
        # Quantize module/block based on selector and supported types
        if not selector.should_quantize(name):
            continue

        # Get parent module and child module name
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
        else:
            parent_module = model
            child_name = name

        # Create module-specific config with overrides applied
        module_specific_quant_config = ModuleSpecificQuantConfig(quant_config, name, type(module))

        # Create quantized module based on type
        if isinstance(module, nn.MultiheadAttention) and qmultiheadattention_cls is not None:
            new_module = copy_multiheadattention_to_qmultiheadattention(
                module, qmultiheadattention_cls, module_specific_quant_config
            )
            block_replacements.append((parent_module, child_name, new_module))
        # Check subclasses first before checking parent class OlmoeAttention
        elif isinstance(module, OlmoeFlashAttention2) and qkvcacheolmoeflashattention2_cls is not None:
            new_module = copy_olmoeattention_qkvcache_olmoeattention(
                module, qkvcacheolmoeflashattention2_cls, module_specific_quant_config
            )
            block_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, OlmoeSdpaAttention) and qkvcacheolmoesdpaattention_cls is not None:
            new_module = copy_olmoeattention_qkvcache_olmoeattention(
                module, qkvcacheolmoesdpaattention_cls, module_specific_quant_config
            )
            block_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, OlmoeAttention) and qkvcacheolmoeattention_cls is not None:
            new_module = copy_olmoeattention_qkvcache_olmoeattention(
                module, qkvcacheolmoeattention_cls, module_specific_quant_config
            )
            block_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, Qwen2_5OmniAttention) and qkvcacheqwen2_5omniattention_cls is not None:
            new_module = copy_qwen2_5omniattention_to_qkvcache_qwen2_5omniattention(
                module, qkvcacheqwen2_5omniattention_cls, module_specific_quant_config
            )
            block_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, Qwen3MoeAttention) and qkvcacheqwen3moeattention_cls is not None:
            new_module = copy_qwen3moeattention_to_qkvcache_qwen3moeattention(
                module, qkvcacheqwen3moeattention_cls, module_specific_quant_config
            )
            block_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, Qwen3Attention) and qkvcacheqwen3attention_cls is not None:
            new_module = copy_qwen3attention_to_qkvcache_qwen3attention(
                module, qkvcacheqwen3attention_cls, module_specific_quant_config
            )
            block_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, LlamaAttention) and qkvcachellamaattention_cls is not None:
            new_module = copy_llamaattention_to_qkvcache_llamaattention(
                module, qkvcachellamaattention_cls, module_specific_quant_config
            )
            block_replacements.append((parent_module, child_name, new_module))

    # Execute block replacements
    for parent, child_name, new_module in block_replacements:
        replace_structure(parent, child_name, new_module)

    # 2. Collect modules to replace
    module_replacements = []

    for name, module in model.named_modules():
        # Quantize module/block based on selector and supported types
        if not selector.should_quantize(name):
            continue

        # Get parent module and child module name
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
        else:
            parent_module = model
            child_name = name

        # Create module-specific config with overrides applied
        module_specific_quant_config = ModuleSpecificQuantConfig(quant_config, name, type(module))

        # Create quantized module based on type
        if isinstance(module, nn.Linear) and qlinear_cls is not None:
            new_module = copy_linear_to_qlinear(
                module, qlinear_cls, module_specific_quant_config
            )
            module_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, nn.Embedding) and qembedding_cls is not None:
            new_module = copy_embedding_to_qembedding(
                module, qembedding_cls, module_specific_quant_config
            )
            module_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, nn.Conv1d) and qconv1d_cls is not None:
            new_module = copy_conv1d_to_qconv1d(
                module, qconv1d_cls, module_specific_quant_config
            )
            module_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, nn.Conv2d) and qconv2d_cls is not None:
            new_module = copy_conv2d_to_qconv2d(
                module, qconv2d_cls, module_specific_quant_config
            )
            module_replacements.append((parent_module, child_name, new_module))
        elif isinstance(module, nn.Conv3d) and qconv3d_cls is not None:
            new_module = copy_conv3d_to_qconv3d(
                module, qconv3d_cls, module_specific_quant_config
            )
            module_replacements.append((parent_module, child_name, new_module))

    # Execute module replacements
    for parent, child_name, new_module in module_replacements:
        replace_structure(parent, child_name, new_module)

    return model

def replace_applied_quantized_weights(
    model: nn.Module,
    selector: QuantSelector,
    qlinear_cls: nn.Module = None,               # Quantized Linear class to replace with q_weights
    qembedding_cls: nn.Module = None,            # Quantized Embedding class to replace with q_weights (optional)
    qconv1d_cls: nn.Module = None,               # Quantized Conv1d class to replace with q_weights (optional)
    qconv2d_cls: nn.Module = None,               # Quantized Conv2d class to replace with q_weights (optional)
    qconv3d_cls: nn.Module = None,               # Quantized Conv3d class to replace with q_weights (optional)
    qmultiheadattention_cls: nn.Module = None,   # Quantized MultiheadAttention class to replace with q_weights (optional)
    replace_weights=True,
) -> nn.Module:
    """
    Replace weights in quantized modules with their quantized versions.
    
    This function iterates through all modules in the model and replaces the weights
    of quantized modules (QLinear, QEmbedding, QConv*, QMultiheadAttention) with their
    quantized counterparts by calling their _weight_quant method with replace_self=True.
    
    Note: ❗️ If `tie_word_embeddings=True`, make sure QEmbedding and QLinear tied weights are only quantized once to avoid double quantization, and configure QEmebedding not to replace weights.
    
    Args:
        model: PyTorch model with quantized modules
        selector: QuantSelector to determine which modules to process
        qlinear_cls: Quantized Linear class
        qembedding_cls: Quantized Embedding class
        qconv1d_cls: Quantized Conv1d class
        qconv2d_cls: Quantized Conv2d class
        qconv3d_cls: Quantized Conv3d class
        qmultiheadattention_cls: Quantized MultiheadAttention class
        replace_weights: Whether to actually replace weights (default: True)
        
    Returns:
        Model with replaced quantized weights
    """
    if not replace_weights:
        return model
    
    for name, module in model.named_modules():
        # Replace weights for module/block based on selector and supported types
        if not selector.should_quantize(name):
            continue

        # Replace weights for QLinear
        if qlinear_cls is not None and isinstance(module, qlinear_cls):
            if hasattr(module, '_weight_quant'):
                module._weight_quant(replace_self=True)
        
        # Replace weights for QEmbedding
        elif qembedding_cls is not None and isinstance(module, qembedding_cls):
            if hasattr(module, '_weight_quant'):
                # Fix `tie_word_embeddings=True` issue
                model_class_name = model.__class__.__name__
                if isinstance(model, (PreTrainedModel, )) and ('CausalLM' in model_class_name or 'GPT' in model_class_name):
                    if model.config.tie_word_embeddings:
                        # LOG
                        print(f"⚠️ Skipped replacing weights for QEmbedding '{name}' due to tied embeddings.")
                        continue
                module._weight_quant(replace_self=True)
        
        # Replace weights for QConv1d
        elif qconv1d_cls is not None and isinstance(module, qconv1d_cls):
            if hasattr(module, '_weight_quant'):
                module._weight_quant(replace_self=True)
                
        # Replace weights for QConv2d
        elif qconv2d_cls is not None and isinstance(module, qconv2d_cls):
            if hasattr(module, '_weight_quant'):
                module._weight_quant(replace_self=True)
                
        # Replace weights for QConv3d
        elif qconv3d_cls is not None and isinstance(module, qconv3d_cls):
            if hasattr(module, '_weight_quant'):
                module._weight_quant(replace_self=True)
                
        # Replace weights for QMultiheadAttention
        elif qmultiheadattention_cls is not None and isinstance(module, qmultiheadattention_cls):
            if hasattr(module, '_weight_quant'):
                module._weight_quant(replace_self=True)
    
    return model
