r"""
# Quantization Functions

## Function Naming Convention

### Weight Quantization
For example, `weight_quant_uniform_symmetric_clip_per_block_int1_58`:

- `weight_quant`: specifies that this function is for weight quantization.
- `uniform`: specifies that this function is uniform quantization.
- `symmetric_clip`: specifies the quantization method used (symmetric clipping).
- `per_block`: specifies that the quantization is performed per-block.
- `int1_58`: specifies the bit-width for corresponding values.

### State Quantization (Activation & KV Cache)
For example, `state_quant_uniform_symmetric_absmax_per_token_int8`:

- `state_quant`: specifies that this function is for state quantization (activation/kv_cache).
- `uniform`: specifies that this function is uniform quantization.
- `symmetric_absmax`: specifies the quantization method used (symmetric absmax).
- `per_token`: specifies that the quantization is performed per-token (last dimension).
- `int8`: specifies the bit-width for quantized values.

"""
import torch
from torch import Tensor

from .quant_function_config import (
    mixed_precision_prop,
    w2a8_block_size,
    w4a8_block_size,
)

# =============================================================
# INT1_58 (Ternary) Weight Quantization - Clip Method
# =============================================================


def weight_quant_uniform_symmetric_clip_per_tensor_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-tensor using clip method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed as mean(|w|) * w_scale_factor.
    
    Args:
        w: Weight tensor to quantize
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Calculate the scale factor: mean(|w|) * w_scale_factor
        w_scale = w.abs().mean().mul_(w_scale_factor).clamp_(min=epsilon)

        # Quantize to INT1_58: {-1, 0, 1}
        w_quantized = w.div(w_scale).round_().clamp_(-1, 1)

        return w_quantized * w_scale


def weight_quant_uniform_symmetric_clip_per_channel_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-channel using clip method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per output channel.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Compute scale factor for each output channel
        # Shape: (out_dim, 1)
        w_scale = w.abs().mean(dim=-1, keepdim=True).mul_(w_scale_factor).clamp_(min=epsilon)

        # Quantize to INT1_58: {-1, 0, 1}
        w_quantized = w.div(w_scale).round_().clamp_(-1, 1)

    return w_quantized * w_scale


def weight_quant_uniform_symmetric_clip_per_block_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-block using clip method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per block within each output channel.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        original_shape = w.shape
        new_shape = list(original_shape[:-1]) + [-1, block_size]
        w = w.view(new_shape)
        
        # Compute scale factor for each block
        # Shape: (out_dim, block_num, 1)
        w_scale = w.abs().mean(dim=-1, keepdim=True).mul_(w_scale_factor).clamp_(min=epsilon)
        
        # Quantize to INT1_58: {-1, 0, 1}
        w_quant = w.div(w_scale).round_().clamp_(-1, 1)

        w_quant = w_quant * w_scale
        w_quant = w_quant.view(original_shape)

    return w_quant


def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_dynamic(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using dynamic mixed precision.
    
    Dynamic Strategy (Variance-based Block Selection):
    - Divides weight into blocks and calculates variance for each block
    - Dynamically selects top mixed_precision_prop (e.g., 1%) blocks with highest variance
    - High-variance blocks (important weights) are quantized to INT4 using absmax method
    - Remaining blocks are quantized to INT1_58 using clip method
    
    Example:
    - With mixed_precision_prop=0.01, the top 1% most variable blocks use INT4: [-7, 7]
    - The remaining 99% less variable blocks use INT1_58: {-1, 0, 1}
    
    This approach adapts to weight distribution, allocating higher precision to
    more important blocks based on their variance.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT4)
    
    Returns:
        Quantized weight tensor with dynamic mixed precision quantization
    """
    bits_int4 = 4
    max_val_int4 = 2**(bits_int4 - 1) - 1  # 7 for INT4
    
    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        # Make sure the last dimension is divisible by block_size
        original_shape = w.shape
        intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
        w_blocked = w.view(intermediate_shape)
        
        # Flatten all dimensions except the last one (block_size) to make indexing easier
        # Shape: (total_num_blocks, block_size)
        num_blocks = w_blocked.numel() // block_size
        w_flat = w_blocked.view(num_blocks, block_size)
        
        # Calculate importance score for each block (using variance as metric)
        # Shape: (num_blocks,)
        block_variance = w_flat.var(dim=-1)
        
        # Determine how many blocks should use INT4 (higher precision)
        num_int4_blocks = min(max(1, int(num_blocks * mixed_precision_prop)), num_blocks)
        
        # Select top-k blocks with highest variance for INT4 quantization
        # Shape: (num_int4_blocks,)
        _, top_indices = torch.topk(block_variance, k=num_int4_blocks, dim=-1)
        
        # Create mask for INT4 blocks
        # Shape: (num_blocks,)
        int4_mask = torch.zeros_like(block_variance, dtype=torch.bool)
        int4_mask.scatter_(dim=-1, index=top_indices, value=True)
        
        # Expand mask to match block dimension
        # Shape: (num_blocks, 1)
        int4_mask_expanded = int4_mask.unsqueeze(-1)
        
        # ===== INT4 Quantization (absmax method) for selected blocks =====
        # Compute scale factor for INT4 blocks using maximum absolute value
        w_scale_int4 = w_flat.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val_int4
        # Quantize to INT4: [-7, 7]
        w_quant_int4 = (w_flat / w_scale_int4).round().clamp(-max_val_int4, max_val_int4) * w_scale_int4
        
        # ===== INT1_58 Quantization (clip method) for remaining blocks =====
        # Compute scale factor for INT1_58 blocks using mean absolute value
        w_scale_int1_58 = w_flat.abs().mean(dim=-1, keepdim=True).mul(w_scale_factor).clamp_(min=epsilon)
        # Quantize to INT1_58: {-1, 0, 1}
        w_quant_int1_58 = (w_flat / w_scale_int1_58).round().clamp(-1, 1) * w_scale_int1_58
        
        # ===== Combine INT4 and INT1_58 blocks =====
        # Use INT4 for high-variance blocks, INT1_58 for others
        w_quant_flat = torch.where(int4_mask_expanded, w_quant_int4, w_quant_int1_58)
        
        # Reshape back to intermediate shape then to original shape
        w_quant = w_quant_flat.view(original_shape)
    
    return w_quant


def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using static mixed precision.
    
    Static Strategy (Position-based Block Selection):
    - Divides weight into blocks in sequential order
    - Statically assigns the first mixed_precision_prop (e.g., 1%) blocks to INT4
    - First blocks (by position) are quantized to INT4 using absmax method
    - Remaining blocks are quantized to INT1_58 using clip method
    
    Example:
    - With mixed_precision_prop=0.01, the first 1% blocks use INT4: [-7, 7]
    - The remaining 99% blocks use INT1_58: {-1, 0, 1}
    
    This approach uses a fixed pattern without analyzing weight importance,
    assuming that earlier blocks in the tensor are more critical.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT4)
    
    Returns:
        Quantized weight tensor with static mixed precision quantization
    """
    bits_int4 = 4
    max_val_int4 = 2**(bits_int4 - 1) - 1  # 7 for INT4
    
    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        # Make sure the last dimension is divisible by block_size
        original_shape = w.shape
        intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
        w_blocked = w.view(intermediate_shape)
        
        # Flatten all dimensions except the last one (block_size) to make indexing easier
        # Shape: (total_num_blocks, block_size)
        num_blocks = w_blocked.numel() // block_size
        w_flat = w_blocked.view(num_blocks, block_size)
        
        # Determine how many blocks should use INT4 (higher precision)
        # Directly use the first num_int4_blocks as important blocks
        num_int4_blocks = min(max(1, int(num_blocks * mixed_precision_prop)), num_blocks)
        
        # ===== INT4 Quantization (absmax method) for first num_int4_blocks =====
        w_int4_blocks = w_flat[:num_int4_blocks, :]
        w_scale_int4 = w_int4_blocks.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val_int4
        w_quant_int4 = (w_int4_blocks / w_scale_int4).round().clamp(-max_val_int4, max_val_int4) * w_scale_int4
        
        # ===== INT1_58 Quantization (clip method) for remaining blocks =====
        w_int1_58_blocks = w_flat[num_int4_blocks:, :]
        w_scale_int1_58 = w_int1_58_blocks.abs().mean(dim=-1, keepdim=True).mul(w_scale_factor).clamp_(min=epsilon)
        w_quant_int1_58 = (w_int1_58_blocks / w_scale_int1_58).round().clamp(-1, 1) * w_scale_int1_58
        
        # ===== Combine INT4 and INT1_58 blocks =====
        w_quant_flat = torch.cat([w_quant_int4, w_quant_int1_58], dim=0)
        
        # Reshape back to intermediate shape then to original shape
        w_quant = w_quant_flat.view(original_shape)
    
    return w_quant


def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_sparse(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using sparse static mixed precision.
    
    Sparse Static Strategy (Interspersed Block Selection):
    - Divides weight into sequential blocks.
    - Within every 100-block chunk, the first `100 * mixed_precision_prop` blocks
      are statically assigned to INT4 precision.
    - This creates a sparse, regular pattern of higher-precision blocks.
    
    Example:
    - With `mixed_precision_prop=0.01`, the 1st block of every 100 is INT4,
      while blocks 2-100 are INT1_58.
    - Block indices 0, 100, 200, ... get INT4.
    - Block indices 1-99, 101-199, ... get INT1_58.
    
    This differs from the standard static method by distributing INT4 blocks
    evenly throughout the tensor rather than concentrating them at the start.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT4)
    
    Returns:
        Quantized weight tensor with sparse static mixed precision quantization.
    """
    bits_int4 = 4
    max_val_int4 = 2**(bits_int4 - 1) - 1  # 7 for INT4
    
    with torch.no_grad():
        # Reshape to a flat 2D tensor of (num_blocks, block_size)
        original_shape = w.shape
        intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
        w_blocked = w.view(intermediate_shape)
        
        num_blocks = w_blocked.numel() // block_size
        w_flat = w_blocked.view(num_blocks, block_size)

        # --- Create the sparse mask for INT4 blocks ---
        group_size = 100
        num_int4_per_group = min(max(1, int(group_size * mixed_precision_prop)), group_size)

        # Create indices from 0 to num_blocks - 1
        indices = torch.arange(num_blocks, device=w.device)

        # Mask is True for the first `num_int4_per_group` blocks in every `group_size` chunk
        int4_mask = (indices % group_size) < num_int4_per_group
        
        # Expand mask to match block dimension for torch.where
        # Shape: (num_blocks, 1)
        int4_mask_expanded = int4_mask.unsqueeze(-1)
        
        # ===== INT4 Quantization (absmax method) for all blocks =====
        w_scale_int4 = w_flat.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val_int4
        w_quant_int4 = (w_flat / w_scale_int4).round().clamp(-max_val_int4, max_val_int4) * w_scale_int4
        
        # ===== INT1_58 Quantization (clip method) for all blocks =====
        w_scale_int1_58 = w_flat.abs().mean(dim=-1, keepdim=True).mul(w_scale_factor).clamp_(min=epsilon)
        w_quant_int1_58 = (w_flat / w_scale_int1_58).round().clamp(-1, 1) * w_scale_int1_58
        
        # ===== Combine using the sparse mask =====
        # Use INT4 for blocks where mask is True, INT1_58 otherwise
        w_quant_flat = torch.where(int4_mask_expanded, w_quant_int4, w_quant_int1_58)
        
        # Reshape back to original shape
        w_quant = w_quant_flat.view(original_shape)
    
    return w_quant


# =============================================================
# INT1_58 (Ternary) Weight Quantization - Absmax Method
# =============================================================


def weight_quant_uniform_symmetric_absmax_per_tensor_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-tensor using absmax method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed as max(|w|).
    
    Args:
        w: Weight tensor to quantize
        epsilon: Small value to prevent division by zero
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Calculate the scale factor using maximum absolute value
        w_scale = w.abs().max().clamp_(min=epsilon)
        
        # Quantize to INT1_58: {-1, 0, 1}
        w_quantized = w.div(w_scale).round_().clamp_(-1, 1)
        
        return w_quantized * w_scale


def weight_quant_uniform_symmetric_absmax_per_channel_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-channel using absmax method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per output channel using max(|w|).
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Compute scale factor for each output channel using maximum absolute value
        # Shape: (out_dim, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon)
        
        # Quantize to INT1_58: {-1, 0, 1}
        w_quantized = w.div(w_scale).round_().clamp_(-1, 1)
    
    return w_quantized * w_scale


def weight_quant_uniform_symmetric_absmax_per_block_int1_58(
    w: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-block using absmax method.
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per block using max(|w|).
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        original_shape = w.shape
        new_shape = list(original_shape[:-1]) + [-1, block_size]
        w = w.view(new_shape)
        
        # Compute scale factor for each block using maximum absolute value
        # Shape: (out_dim, block_num, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon)
        
        # Quantize to INT1_58: {-1, 0, 1}
        w_quant = w.div(w_scale).round_().clamp_(-1, 1)
        
        w_quant = w_quant * w_scale
        w_quant = w_quant.view(original_shape)
    
    return w_quant


# =============================================================
# INT4 Weight Quantization - Absmax Method
# =============================================================


def weight_quant_uniform_symmetric_absmax_per_tensor_int4(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT4 per-tensor using absmax method.
    
    Quantizes weight to INT4: [-7, 7] * w_scale.
    Scale factor is computed as max(|w|) / 7.
    
    Args:
        w: Weight tensor to quantize
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized weight tensor with values in [-7, 7] * w_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Calculate the scale factor using maximum absolute value
        w_scale = w.abs().max().clamp_(min=epsilon) / max_val

        # Quantize to INT4: [-7, 7]
        w_quantized = w.div(w_scale).round_().clamp_(-max_val, max_val)

        return w_quantized * w_scale


def weight_quant_uniform_symmetric_absmax_per_channel_int4(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT4 per-channel using absmax method.
    
    Quantizes weight to INT4: [-7, 7] * w_scale.
    Scale factor is computed per output channel.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized weight tensor with values in [-7, 7] * w_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Compute scale factor for each output channel
        # Shape: (out_dim, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val

        # Quantize to INT4: [-7, 7]
        w_quantized = w.div(w_scale).round_().clamp_(-max_val, max_val)

    return w_quantized * w_scale


def weight_quant_uniform_symmetric_absmax_per_block_int4(
    w: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT4 per-block using absmax method.
    
    Quantizes weight to INT4: [-7, 7] * w_scale.
    Scale factor is computed per block within each output channel.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized weight tensor with values in [-7, 7] * w_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        original_shape = w.shape
        new_shape = list(original_shape[:-1]) + [-1, block_size]
        w = w.view(new_shape)
        
        # Compute scale factor for each block
        # Shape: (out_dim, block_num, 1)
        w_scale = w.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT4: [-7, 7]
        w_quant = w.div(w_scale).round_().clamp_(-max_val, max_val)

        w_quant = w_quant * w_scale
        w_quant = w_quant.view(original_shape)

    return w_quant


def weight_quant_uniform_asymmetric_max_per_tensor_int4(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT4 per-tensor using asymmetric max method.
    
    Quantizes weight to INT4: [-8, 7] * w_scale.
    Scale factor is computed using signed max value (not absmax).
    It is compatible with the GGML Q4_0 quantization scheme.
    
    Args:
        w: Weight tensor to quantize
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized weight tensor with values in [-8, 7] * w_scale
    """
    bits = 4
    # For INT4: range is [-2^(bits-1), 2^(bits-1) - 1] = [-8, 7]
    min_val = -(2 ** (bits - 1))     # -8 for INT4
    # max_val = 2 ** (bits - 1) - 1  # 7 for INT4
    zero_point = -min_val            # 8 for INT4
    
    with torch.no_grad():
        # Find the maximum value (with sign) for the entire tensor
        w_max = w.max()
        
        # Compute scale factor: d = max / min_val
        # For INT4: d = max / -8
        w_scale = w_max / min_val
        
        # Handle zero scale (add epsilon to prevent division by zero)
        w_scale = torch.where(
            w_scale.abs() < epsilon,
            torch.tensor(epsilon, device=w_scale.device, dtype=w_scale.dtype),
            w_scale
        )
        
        # Quantization formula: q = clamp(round(w / scale + zero_point), 0, 2^bits - 1)
        # This maps [min_val, max_val] range to [0, 2^bits - 1] storage range
        w_div_scale = w / w_scale
        w_quant_storage = (w_div_scale + zero_point).round().clamp(0, 2**bits - 1)
        
        # Dequantization: w = (q - zero_point) * scale
        # This converts back from [0, 15] storage to [min_val, max_val] * scale
        w_quant = (w_quant_storage - zero_point) * w_scale

    return w_quant


def weight_quant_uniform_asymmetric_max_per_channel_int4(
    w: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize weight to INT4 per-channel using asymmetric max method.
    
    Quantizes weight to INT4: [-8, 7] * w_scale.
    Scale factor is computed per output channel using signed max value (not absmax).
    It is compatible with the GGML Q4_0 quantization scheme.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized weight tensor with values in [-8, 7] * w_scale
    """
    bits = 4
    # For INT4: range is [-2^(bits-1), 2^(bits-1) - 1] = [-8, 7]
    min_val = -(2 ** (bits - 1))     # -8 for INT4
    # max_val = 2 ** (bits - 1) - 1  # 7 for INT4
    zero_point = -min_val            # 8 for INT4
    
    with torch.no_grad():
        # Find the maximum value (with sign) for each output channel
        # Shape: (out_dim, 1)
        w_max = w.max(dim=-1, keepdim=True).values
        
        # Compute scale factor: d = max / min_val
        # For INT4: d = max / -8
        w_scale = w_max / min_val
        
        # Handle zero scale (add epsilon to prevent division by zero)
        w_scale = torch.where(
            w_scale.abs() < epsilon,
            torch.tensor(epsilon, device=w_scale.device, dtype=w_scale.dtype),
            w_scale
        )
        
        # Quantization formula: q = clamp(round(w / scale + zero_point), 0, 2^bits - 1)
        # This maps [min_val, max_val] range to [0, 2^bits - 1] storage range
        w_div_scale = w / w_scale
        w_quant_storage = (w_div_scale + zero_point).round().clamp(0, 2**bits - 1)
        
        # Dequantization: w = (q - zero_point) * scale
        # This converts back from [0, 15] storage to [min_val, max_val] * scale
        w_quant = (w_quant_storage - zero_point) * w_scale

    return w_quant


def weight_quant_uniform_asymmetric_max_per_block_int4(
    w: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w4a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT4 per-block using asymmetric max method.
    
    Quantizes weight to INT4: [-8, 7] * w_scale.
    Scale factor is computed per block using signed max value (not absmax).
    It is compatible with the GGML Q4_0 quantization scheme.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block (default 32 for Q4_0)

    Returns:
        Quantized weight tensor with values in [-8, 7] * w_scale
    """
    bits = 4
    # For INT4: range is [-2^(bits-1), 2^(bits-1) - 1] = [-8, 7]
    min_val = -(2 ** (bits - 1))     # -8 for INT4
    # max_val = 2 ** (bits - 1) - 1  # 7 for INT4
    zero_point = -min_val            # 8 for INT4
    
    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        original_shape = w.shape
        new_shape = list(original_shape[:-1]) + [-1, block_size]
        w_reshaped = w.view(new_shape)
        
        # Find the maximum value (with sign) for each block
        # Shape: (out_dim, block_num, 1)
        w_max = w_reshaped.max(dim=-1, keepdim=True).values
        
        # Compute scale factor: d = max / min_val
        # For INT4: d = max / -8
        w_scale = w_max / min_val
        
        # Handle zero scale (add epsilon to prevent division by zero)
        w_scale = torch.where(
            w_scale.abs() < epsilon,
            torch.tensor(epsilon, device=w_scale.device, dtype=w_scale.dtype),
            w_scale
        )
        
        # Quantization formula: q = clamp(round(w / scale + zero_point), 0, 2^bits - 1)
        # This maps [min_val, max_val] range to [0, 2^bits - 1] storage range
        w_div_scale = w_reshaped / w_scale
        w_quant_storage = (w_div_scale + zero_point).round().clamp(0, 2**bits - 1)
        
        # Dequantization: w = (q - zero_point) * scale
        # This converts back from [0, 15] storage to [min_val, max_val] * scale
        w_quant = (w_quant_storage - zero_point) * w_scale
        
        # Reshape back to original shape
        w_quant = w_quant.view(original_shape)

    return w_quant


# =============================================================
# Stepped Weight Quantization - Symmetric Method
# =============================================================


def weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static_nested(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize weight to INT1_58 and INT4 per-block using static mixed precision
    and stepped quantization function.
    
    SteppedQuantFunc(W) = MP_INT1.58+INT4( INT4(W) )
    SteppedQuantFunc(X) = MP_INT4+INT8( INT8(W) )
    
    Static Strategy (Position-based Block Selection):
    - Divides weight into blocks in sequential order
    - Statically assigns the first mixed_precision_prop (e.g., 1%) blocks to INT4
    - First blocks (by position) are quantized to INT4 using absmax method
    - Remaining blocks are quantized to INT1_58 using clip method
    
    Example:
    - With mixed_precision_prop=0.01, the first 1% blocks use INT4: [-7, 7]
    - The remaining 99% blocks use INT1_58: {-1, 0, 1}
    
    This approach uses a fixed pattern without analyzing weight importance,
    assuming that earlier blocks in the tensor are more critical.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT4)
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    w_hidden_int4 = weight_quant_uniform_symmetric_absmax_per_block_int4(
        w=w,
        epsilon=epsilon,
        block_size=block_size,
    )
    
    w_quant = weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static(
        w=w_hidden_int4,
        epsilon=epsilon,
        w_scale_factor=w_scale_factor,
        block_size=block_size,
        mixed_precision_prop=mixed_precision_prop,
    )
    
    return w_quant


def weight_quant_uniform_symmetric_clip_per_block_int1_58_nested(
    w: Tensor,
    epsilon: float = 1e-5,
    w_scale_factor: float = 2.0,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize weight to INT1_58 per-block using clip method and stepped quantization.
    
    SteppedQuantFunc(W) = INT1.58( INT4(W) )
    SteppedQuantFunc(X) = INT4( INT8(W) )
    
    Ternarizes weight into INT1_58: {-1, 0, 1} * w_scale.
    Scale factor is computed per block within each output channel.
    
    Args:
        w: Weight tensor to quantize, shape (out_dim, in_dim)
        epsilon: Small value to prevent division by zero
        w_scale_factor: Multiplier for the mean absolute value (default: 2.0)
        block_size: Size of each quantization block
    
    Returns:
        Quantized weight tensor with values in {-1, 0, 1} * w_scale
    """
    w_hidden_int4 = weight_quant_uniform_symmetric_absmax_per_block_int4(
        w=w,
        epsilon=epsilon,
        block_size=block_size,
    )
    
    w_quant = weight_quant_uniform_symmetric_clip_per_block_int1_58(
        w=w_hidden_int4,
        epsilon=epsilon,
        w_scale_factor=w_scale_factor,
        block_size=block_size,
    )
    
    return w_quant


# =============================================================
# INT2 State Quantization - Absmax Method
# =============================================================


def state_quant_uniform_symmetric_absmax_per_token_int2(
    x: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT2 per-token using absmax method.
    
    Quantizes along the last dimension (per-token/per-feature).
    Assumes input was normalized before quantization (e.g., LayerNorm, RMSNorm).
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized tensor with values in [-1, 1] * x_scale
    """
    bits = 2
    max_val = 2**(bits - 1) - 1  # 1 for INT2

    with torch.no_grad():
        # Compute scale factor for each token (last dimension)
        # Shape: ([batch,] seq_len, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT2: [-1, 1]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        
        return x_quant * x_scale


def state_quant_uniform_symmetric_absmax_per_block_int2(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT2 per-block using absmax method.
    
    Applies group quantization by dividing features into blocks.
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized tensor with values in [-1, 1] * x_scale
    """
    bits = 2
    max_val = 2**(bits - 1) - 1  # 1 for INT2

    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        original_shape = x.shape
        new_shape = list(original_shape[:-1]) + [-1, block_size]
        x = x.view(new_shape)

        # Compute scale factor for each block
        # Shape: ([batch,] seq_len, block_num, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT2: [-1, 1]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        x_quant = x_quant * x_scale
        x_quant = x_quant.view(original_shape)

        return x_quant


# =============================================================
# INT4 State Quantization - Absmax Method
# =============================================================


def state_quant_uniform_symmetric_absmax_per_token_int4(
    x: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT4 per-token using absmax method.
    
    Quantizes along the last dimension (per-token/per-feature).
    Assumes input was normalized before quantization (e.g., LayerNorm, RMSNorm).
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized tensor with values in [-7, 7] * x_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Compute scale factor for each token (last dimension)
        # Shape: ([batch,] seq_len, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT4: [-7, 7]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        
        return x_quant * x_scale


def state_quant_uniform_symmetric_absmax_per_block_int4(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT4 per-block using absmax method.
    
    Applies group quantization by dividing features into blocks.
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized tensor with values in [-7, 7] * x_scale
    """
    bits = 4
    max_val = 2**(bits - 1) - 1  # 7 for INT4

    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        original_shape = x.shape
        new_shape = list(original_shape[:-1]) + [-1, block_size]
        x = x.view(new_shape)

        # Compute scale factor for each block
        # Shape: ([batch,] seq_len, block_num, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT4: [-7, 7]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        x_quant = x_quant * x_scale
        x_quant = x_quant.view(original_shape)

        return x_quant


def state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w4a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT4 and INT8 per-block using dynamic mixed precision.
    
    Dynamic Strategy (Variance-based Block Selection):
    - Divides activation into blocks and calculates variance for each block
    - Dynamically selects top mixed_precision_prop (e.g., 1%) blocks with highest variance
    - High-variance blocks (important activations) are quantized to INT8 using absmax method
    - Remaining blocks are quantized to INT4 using absmax method
    
    Example:
    - With mixed_precision_prop=0.01, the top 1% most variable blocks use INT8: [-127, 127]
    - The remaining 99% less variable blocks use INT4: [-7, 7]
    
    This approach adapts to activation distribution, allocating higher precision to
    more important blocks based on their variance.
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to quantize to higher precision (INT8)
    
    Returns:
        Quantized tensor with dynamic mixed precision quantization
    """
    bits_int4 = 4
    max_val_int4 = 2**(bits_int4 - 1) - 1  # 7 for INT4
    bits_int8 = 8
    max_val_int8 = 2**(bits_int8 - 1) - 1  # 127 for INT8
    
    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        original_shape = x.shape
        intermediate_shape = list(original_shape[:-1]) + [-1, block_size]
        x_blocked = x.view(intermediate_shape)
        
        # Flatten all dimensions except the last one (block_size) to make indexing easier
        # Shape: (total_num_blocks, block_size)
        num_blocks = x_blocked.numel() // block_size
        x_flat = x_blocked.view(num_blocks, block_size)
        
        # Calculate importance score for each block (using variance as metric)
        # Shape: (num_blocks,)
        block_variance = x_flat.var(dim=-1)
        
        # Determine how many blocks should use INT8 (higher precision)
        num_int8_blocks = min(max(1, int(num_blocks * mixed_precision_prop)), num_blocks)
        
        # Select top-k blocks with highest variance for INT8 quantization
        # Shape: (num_int8_blocks,)
        _, top_indices = torch.topk(block_variance, k=num_int8_blocks, dim=-1)
        
        # Create mask for INT8 blocks
        # Shape: (num_blocks,)
        int8_mask = torch.zeros_like(block_variance, dtype=torch.bool)
        int8_mask.scatter_(dim=-1, index=top_indices, value=True)
        
        # Expand mask to match block dimension
        # Shape: (num_blocks, 1)
        int8_mask_expanded = int8_mask.unsqueeze(-1)
        
        # ===== INT8 Quantization (absmax method) for selected blocks =====
        # Compute scale factor for INT8 blocks using maximum absolute value
        x_scale_int8 = x_flat.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val_int8
        # Quantize to INT8: [-127, 127]
        x_quant_int8 = (x_flat / x_scale_int8).round().clamp(-max_val_int8, max_val_int8) * x_scale_int8
        
        # ===== INT4 Quantization (absmax method) for remaining blocks =====
        # Compute scale factor for INT4 blocks using maximum absolute value
        x_scale_int4 = x_flat.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val_int4
        # Quantize to INT4: [-7, 7]
        x_quant_int4 = (x_flat / x_scale_int4).round().clamp(-max_val_int4, max_val_int4) * x_scale_int4
        
        # ===== Combine INT8 and INT4 blocks =====
        # Use INT8 for high-variance blocks, INT4 for others
        x_quant_flat = torch.where(int8_mask_expanded, x_quant_int8, x_quant_int4)
        
        # Reshape back to original shape
        x_quant = x_quant_flat.view(original_shape)
    
    return x_quant


# =============================================================
# INT8 State Quantization - Absmax Method
# =============================================================


def state_quant_uniform_symmetric_absmax_per_token_int8(
    x: Tensor,
    epsilon: float = 1e-5,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT8 per-token using absmax method.
    
    Quantizes along the last dimension (per-token/per-feature).
    Assumes input was normalized before quantization (e.g., LayerNorm, RMSNorm).
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero

    Returns:
        Quantized tensor with values in [-127, 127] * x_scale
    """
    bits = 8
    max_val = 2**(bits - 1) - 1  # 127 for INT8

    with torch.no_grad():
        # Compute scale factor for each token (last dimension)
        # Shape: ([batch,] seq_len, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT8: [-127, 127]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        
        return x_quant * x_scale


def state_quant_uniform_symmetric_absmax_per_block_int8(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w2a8_block_size,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT8 per-block using absmax method.
    
    Applies group quantization by dividing features into blocks.
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block

    Returns:
        Quantized tensor with values in [-127, 127] * x_scale
    """
    bits = 8
    max_val = 2**(bits - 1) - 1  # 127 for INT8

    with torch.no_grad():
        # Reshape to [..., -1, block_size]
        original_shape = x.shape
        new_shape = list(original_shape[:-1]) + [-1, block_size]
        x = x.view(new_shape)

        # Compute scale factor for each block
        # Shape: ([batch,] seq_len, block_num, 1)
        x_scale = x.abs().max(dim=-1, keepdim=True).values.clamp_(min=epsilon) / max_val
        
        # Quantize to INT8: [-127, 127]
        x_quant = x.div(x_scale).round().clamp_(-max_val, max_val)
        x_quant = x_quant * x_scale
        x_quant = x_quant.view(original_shape)

        return x_quant


def state_quant_uniform_symmetric_absmax_per_block_int4_nested(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w4a8_block_size,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT4 per-block using absmax method and stepped quantization.
    
    SteppedQuantFunc(X) = INT4( INT8(X) )
    
    Applies two-stage quantization:
    1. First quantize to INT8 per-block
    2. Then quantize the INT8 result to INT4 per-block
    
    This stepped approach can help preserve more information compared to direct INT4 quantization.
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block
    
    Returns:
        Quantized tensor with values in [-7, 7] * x_scale
    """
    # Step 1: Quantize to INT8
    x_hidden_int8 = state_quant_uniform_symmetric_absmax_per_block_int8(
        x=x,
        epsilon=epsilon,
        block_size=block_size,
    )
    
    # Step 2: Quantize INT8 result to INT4
    x_quant = state_quant_uniform_symmetric_absmax_per_block_int4(
        x=x_hidden_int8,
        epsilon=epsilon,
        block_size=block_size,
    )
    
    return x_quant


def state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic_nested(
    x: Tensor,
    epsilon: float = 1e-5,
    block_size: int = w4a8_block_size,
    mixed_precision_prop: float = mixed_precision_prop,
) -> Tensor:
    """
    Quantize state (activation/kv_cache) to INT4 and INT8 per-block using dynamic mixed precision
    and stepped quantization function.
    
    SteppedQuantFunc(X) = MP_INT4+INT8( INT8(X) )
    
    Dynamic Strategy (Variance-based Block Selection):
    - First quantizes input to INT8 per-block
    - Then divides the INT8 result into blocks and calculates variance for each block
    - Dynamically selects top mixed_precision_prop (e.g., 1%) blocks with highest variance
    - High-variance blocks are kept at INT8 precision
    - Remaining blocks are further quantized to INT4
    
    Example:
    - With mixed_precision_prop=0.01, the top 1% most variable blocks stay at INT8: [-127, 127]
    - The remaining 99% less variable blocks are quantized to INT4: [-7, 7]
    
    This approach combines stepped quantization with dynamic mixed precision for better
    quality preservation.
    
    Args:
        x: Input tensor to quantize, shape ([batch,] seq_len, hidden_dim)
        epsilon: Small value to prevent division by zero
        block_size: Size of each quantization block
        mixed_precision_prop: Proportion of blocks to keep at higher precision (INT8)
    
    Returns:
        Quantized tensor with dynamic mixed precision stepped quantization
    """
    # Step 1: Quantize to INT8
    x_hidden_int8 = state_quant_uniform_symmetric_absmax_per_block_int8(
        x=x,
        epsilon=epsilon,
        block_size=block_size,
    )
    
    # Step 2: Apply dynamic mixed precision quantization
    x_quant = state_quant_uniform_symmetric_absmax_per_block_mp_int4_int8_dynamic(
        x=x_hidden_int8,
        epsilon=epsilon,
        block_size=block_size,
        mixed_precision_prop=mixed_precision_prop,
    )
    
    return x_quant
