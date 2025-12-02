import torch.nn as nn
from torch import Tensor

from ..util.quant_config import QuantConfig


class QLinear(nn.Linear):
    def __init__(
        self,
        # Standard nn.Linear parameters
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        # Additional QAT hyperparameters
        quant_config: QuantConfig = None,   # Quantization configuration
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        if quant_config is None:
            raise ValueError("quant_config must be provided for QLinear.")

        # Small value to prevent division by zero
        self.epsilon = quant_config.function.epsilon
        # Whether the weights are already quantized: {-2^(n-1), 0, 2^(n-1)} * w_scale
        self.is_w_quantized = quant_config.function.is_w_quantized

        # Quantization configuration
        ## Weight
        self.w_quant_function = quant_config.function.weight_function
        self.w_scale_factor = quant_config.function.w_scale_factor
        self.w_block_size = quant_config.function.w_block_size
        self.w_mixed_precision_prop = quant_config.function.w_mixed_precision_prop
        self.w_kwargs = {'epsilon': self.epsilon}
        if self.w_scale_factor > 0:
            self.w_kwargs['w_scale_factor'] = self.w_scale_factor
        if self.w_block_size > 0:
            self.w_kwargs['block_size'] = self.w_block_size
        if self.w_mixed_precision_prop > 0:
            self.w_kwargs['mixed_precision_prop'] = self.w_mixed_precision_prop
        
        ## Activation
        self.a_quant_function = quant_config.function.activation_function
        self.a_block_size = quant_config.function.a_block_size
        self.a_mixed_precision_prop = quant_config.function.a_mixed_precision_prop
        self.a_kwargs = {'epsilon': self.epsilon}
        if self.a_block_size > 0:
            self.a_kwargs['block_size'] = self.a_block_size
        if self.a_mixed_precision_prop > 0:
            self.a_kwargs['mixed_precision_prop'] = self.a_mixed_precision_prop

    def _weight_quant(self, replace_self: bool = False) -> Tensor:
        # Ternarize weight into {-1, 0, 1} * w_scale
        W = self.weight.data.clone()
        w_quant = self.w_quant_function(w=W, **self.w_kwargs)

        if replace_self:
            if not self.is_w_quantized:
                # IF need to replace self.weight with ternary weights
                self.weight.data = w_quant.clone()
                self.is_w_quantized = True
            else:
                raise RuntimeError("Weights are already ternarized. Cannot replace self again.")
        return w_quant

    def _activation_quant(self, x: Tensor) -> Tensor:
        # Quantize activation
        x_quant = self.a_quant_function(x=x, **self.a_kwargs)
        return x_quant

    def forward(self, x: Tensor) -> Tensor:
        W = self.weight
        B = self.bias

        if self.training:
            # Straight-Through Estimator for training
            w_quant = self._weight_quant(replace_self=False)
            w_quant = W + (w_quant - W).detach()
        else: # is_inference_mode
            if self.is_w_quantized:
                w_quant = W
            else:
                w_quant = self._weight_quant(replace_self=False)

        # Use standard linear during training to ensure correct gradient propagation
        if self.a_quant_function is not None:
            x_quant = self._activation_quant(x)
            x_quant = x + (x_quant - x).detach()
            output = nn.functional.linear(x_quant, w_quant, B)
        else:
            output = nn.functional.linear(x, w_quant, B)

        return output


def copy_linear_to_qlinear(
    linear: nn.Linear,
    qlinear_cls: nn.Module,
    quant_config: QuantConfig = None
):
    """Copy Linear to quantized Linear (adjust according to your QLinear implementation)"""
    # Adjust according to your QLinear implementation
    qlinear = qlinear_cls(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
        quant_config=quant_config
    )
    # Copy weights and bias
    qlinear.weight.data = linear.weight.data.clone()
    if linear.bias is not None:
        qlinear.bias.data = linear.bias.data.clone()
    # Copy state
    qlinear.training = linear.training
    return qlinear


# LinearKVCacheQuant is for quantized attention computation attn(q_q, k_q, v_q)
# Replace linear like q_proj, k_proj, v_proj with LinearKVCacheQuant
# Function: KV Cache quantization before RoPE
# NOTE [AI Infra]: Conflict with llama.cpp gguf quantization implementation. 2025/10/20
class TnLinearKVCacheQuant(QLinear):
    def __init__(
        self,
        # Standard nn.Linear parameters
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        # Additional QAT hyperparameters
        quant_config: QuantConfig = None,   # Quantization configuration
    ) -> None:
        super().__init__(
            in_features, out_features, bias, device, dtype,
            quant_config
        )

        # KV Cache quantization configuration
        self.kv_quant_function = quant_config.function.kv_cache_function
        self.kv_block_size = quant_config.function.kv_block_size
        self.kv_mixed_precision_prop = quant_config.function.kv_mixed_precision_prop
        self.kv_kwargs = {'epsilon': self.epsilon}
        if self.kv_block_size > 0:
            self.kv_kwargs['block_size'] = self.kv_block_size
        if self.kv_mixed_precision_prop > 0:
            self.kv_kwargs['mixed_precision_prop'] = self.kv_mixed_precision_prop

    def forward(self, x: Tensor) -> Tensor:
        W = self.weight
        B = self.bias

        if self.training:
            # Straight-Through Estimator for training
            w_quant = self._weight_quant(replace_self=False)
            w_quant = W + (w_quant - W).detach()
        else: # is_inference_mode
            if self.is_w_quantized:
                w_quant = W
            else:
                w_quant = self._weight_quant(replace_self=False)

        # Use standard linear during training to ensure correct gradient propagation
        if self.a_quant_function is not None:
            x_quant = self._activation_quant(x)
            x_quant = x + (x_quant - x).detach()
            output = nn.functional.linear(x_quant, w_quant, B)
        else:
            output = nn.functional.linear(x, w_quant, B)

        # KV Cache quantization before RoPE
        if self.kv_quant_function is not None:
            kv_quant = self.kv_quant_function(x=output, **self.kv_kwargs)
            output = output + (kv_quant - output).detach()

        return output
