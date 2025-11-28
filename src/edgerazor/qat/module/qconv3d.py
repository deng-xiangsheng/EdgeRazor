"""
Conv3d weight shape: [out_channels, in_channels, kernel_depth, kernel_height, kernel_width]
- out_channels: Number of output channels (filters)
- in_channels: Number of input channels
- kernel_depth: Depth of the convolution kernel
- kernel_height: Height of the convolution kernel
- kernel_width: Width of the convolution kernel

Compatible with quantization functions (per-tensor, per-channel, per-block):
1. reshape to [out_channels, in_channels * kernel_depth * kernel_height * kernel_width]
2. quantize, then reshape back
"""
from typing import Union

import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_3_t

from ..util.quant_config import QuantConfig


class QConv3d(nn.Conv3d):
    def __init__(
        self,
        # Standard nn.Conv3d parameters
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        # Additional QAT hyperparameters
        quant_config: QuantConfig = None,   # Quantization configuration
    ) -> None:
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode, device, dtype
        )
        if quant_config is None:
            raise ValueError("quant_config must be provided for QConv3d.")

        # Small value to prevent division by zero
        self.epsilon = quant_config.function.epsilon
        # Whether the weights are already quantized: {-2^(n-1), 0, 2^(n-1)} * w_scale
        self.is_w_quantized = quant_config.function.is_w_quantized

        # Quantization configuration
        ## Weight
        self.w_quant_function = quant_config.function.weight_function
        self.w_scale_factor = quant_config.function.w_scale_factor
        self.w_block_size = quant_config.function.w_block_size
        ## Activation
        self.a_quant_function = quant_config.function.activation_function
        self.a_block_size = quant_config.function.a_block_size

    def _weight_quant(self, replace_self: bool = False) -> Tensor:
        # Quantize weight into {-2^(n-1), 0, 2^(n-1)} * w_scale
        # Conv3d weight shape: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        # Need to reshape to [out_channels, in_channels * kernel_d * kernel_h * kernel_w] for quantization
        W = self.weight.data.clone()
        original_shape = W.shape
        
        # Reshape: flatten(1) ensures kernel_d * kernel_h * kernel_w are adjacent
        # Shape: [out_channels, in_channels, kernel_d, kernel_h, kernel_w] -> [out_channels, in_channels * kernel_d * kernel_h * kernel_w]
        W_reshaped = W.flatten(1)
        
        # Apply quantization function on reshaped weight
        if self.w_scale_factor > 0 and self.w_block_size > 0:
            w_quant = self.w_quant_function(w=W_reshaped, epsilon=self.epsilon, w_scale_factor=self.w_scale_factor, block_size=self.w_block_size)
        elif self.w_scale_factor > 0:
            w_quant = self.w_quant_function(w=W_reshaped, epsilon=self.epsilon, w_scale_factor=self.w_scale_factor)
        elif self.w_block_size > 0:
            w_quant = self.w_quant_function(w=W_reshaped, epsilon=self.epsilon, block_size=self.w_block_size)
        else:
            w_quant = self.w_quant_function(w=W_reshaped, epsilon=self.epsilon)
        
        # Reshape back to original shape
        w_quant = w_quant.view(original_shape)

        if replace_self:
            if not self.is_w_quantized:
                # IF need to replace self.weight with quantized weights
                self.weight.data = w_quant.clone()
                self.is_w_quantized = True
            else:
                raise RuntimeError("Weights are already quantized. Cannot replace self again.")
        return w_quant

    def _activation_quant(self, x: Tensor) -> Tensor:
        # Quantize activation
        if self.a_block_size > 0:
            x_quant = self.a_quant_function(x=x, epsilon=self.epsilon, block_size=self.a_block_size)
        else:
            x_quant = self.a_quant_function(x=x, epsilon=self.epsilon)
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

        # Use standard convolution during training to ensure correct gradient propagation
        if self.a_quant_function is not None:
            x_quant = self._activation_quant(x)
            x_quant = x + (x_quant - x).detach()
            output = self._conv_forward(x_quant, w_quant, B)
        else:
            output = self._conv_forward(x, w_quant, B)

        return output


def copy_conv3d_to_qconv3d(
    conv: nn.Conv3d,
    qconv_cls: nn.Module,
    quant_config: QuantConfig = None
):
    """Copy Conv3d to quantized Conv3d (adjust according to your QConv3d implementation)"""
    # Adjust according to your QConv3d implementation
    qconv = qconv_cls(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
        quant_config=quant_config
    )
    # Copy weights and bias
    qconv.weight.data = conv.weight.data.clone()
    if conv.bias is not None:
        qconv.bias.data = conv.bias.data.clone()
    # Copy state
    qconv.training = conv.training
    return qconv
