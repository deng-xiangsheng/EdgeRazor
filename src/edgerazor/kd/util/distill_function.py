# ruff: noqa: N812

import torch
import torch.nn.functional as F

from .distill_config import LossConfig


# ========================================= Logits Distillation: Kullback-Leibler Divergence =========================================
def compute_kld(
    input_logits: torch.Tensor,
    target_logits: torch.Tensor,
    labels: torch.Tensor | None,
    kd_config_loss: LossConfig,
) -> torch.Tensor:
    """
    Core function for computing Kullback-Leibler divergence.
    KL(input || target) = sum(input_probs * log(input_probs / target_probs))

    Supported Shapes:
    1. Logits:
       - input/target: [batch_size, seq_len, vocab_size]
       - labels: [batch_size, seq_len]
    2. Attentions:
       - input/target: [batch_size, num_attention_heads, seq_len, seq_len]
       - labels: [batch_size, seq_len] -> [batch_size, num_attention_heads, seq_len] copy last dim num_attention_heads times for masking

    Args:
        input_logits: Logits from the input distribution or Attention scores.
        target_logits: Logits from the target distribution or Attention scores.
        labels: Original labels for computing padding mask. If None, no masking is applied.
        kd_config_loss: LossConfig object.
    """
    # Extract parameters from configuration
    padding_id = kd_config_loss.padding_id
    is_router_logits = kd_config_loss.is_router_logits
    reduction = kd_config_loss.reduction
    temp = kd_config_loss.temperature
    EPS = 1e-8

    # Determine if input is attention/past_key_values map based on dimensions
    # Logits: [B, S, V] (3 dims), Attention: [B, H, S, S] (4 dims)
    is_attention = input_logits.dim() == 4

    if not is_attention and input_logits.dim() == 3:
        # Apply temperature scaling (clamped to minimum 0.1 for numerical stability)
        temp = max(temp, 0.1)
        input_logits = input_logits / temp
        target_logits = target_logits / temp

        # Ensure numerical stability by clamping logits
        log_probs = F.softmax(input_logits, dim=-1).clamp(min=EPS).log()
        target_probs = F.softmax(target_logits, dim=-1).clamp(min=EPS).log()
    elif is_attention and input_logits.dim() == 4:
        # Attention scores: assume already in probability space (after softmax)
        # Ensure numerical stability by clamping logits
        log_probs = input_logits.clamp(min=EPS).log()
        target_probs = target_logits.clamp(min=EPS).log()

    # Compute raw KL divergence elements
    kl_raw = F.kl_div(
        input=log_probs, target=target_probs, reduction="none", log_target=True
    ).sum(dim=-1)  # [batch_size, seq_len] or [batch_size, num_attention_heads, seq_len]

    if not is_attention and input_logits.dim() == 3:
        # Scale KL divergence by temperature squared for logits
        kl_raw = kl_raw * (temp**2)

    # Apply padding mask if target is provided (similar to compute_fd)
    if labels is not None:
        # Create padding mask (computed early for efficiency)
        if not is_router_logits:
            if not is_attention:
                pad_mask = labels.eq(padding_id)
            else:
                # Copy the last dimension: [batch_size, seq_len] -> [batch_size, num_attention_heads, seq_len]
                # pad_mask = labels.eq(padding_id).unsqueeze(1).expand(-1, input_logits.size(1), -1)
                # Do not expand here, use broadcasting in masked_fill
                pad_mask = labels.eq(padding_id).unsqueeze(1)
        else:
            pad_mask = labels.view(1, -1).eq(padding_id)
        valid_elements = (~pad_mask).sum(
            dim=-1
        )  # Number of valid tokens per sample [batch_size] or [batch_size, num_attention_heads]

        # Apply reduction mode with masking
        if reduction == "sum":
            kl_elements = kl_raw.masked_fill(pad_mask, 0.0).sum()

        elif reduction == "mean":
            # Average over all non-padding positions
            kl_sum = kl_raw.masked_fill(pad_mask, 0.0).sum()
            non_pad_total = valid_elements.sum().clamp(min=1)  # Avoid division by zero
            kl_elements = kl_sum / non_pad_total

        elif reduction == "batch_mean":
            # Average per sample, then average across batch
            kl_per_sample = kl_raw.masked_fill(pad_mask, 0.0).sum(
                dim=-1
            )  # [batch_size] or [batch_size, num_attention_heads]
            kl_per_sample = kl_per_sample / valid_elements.clamp(
                min=1
            )  # Mean per sample
            kl_elements = kl_per_sample.mean()

        elif reduction == "none":
            # Keep original shape, but zero out padding positions
            kl_elements = kl_raw.masked_fill(pad_mask, 0.0)

        else:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
    else:
        # No target provided (e.g., ViT models without padding) - no masking needed
        if reduction == "sum":
            kl_elements = kl_raw.sum()
        elif reduction == "mean" or reduction == "batch_mean":
            kl_elements = kl_raw.mean()
        elif reduction == "none":
            kl_elements = kl_raw
        else:
            raise ValueError(f"Unsupported reduction mode: {reduction}")

    return kl_elements


def compute_kld_forward(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None,
    kd_config_loss: LossConfig,
) -> torch.Tensor:
    """
    Forward KL divergence: compute teacher distribution relative to student distribution.
    KL(teacher || student) = sum(teacher_probs * log(teacher_probs / student_probs))
    This is the standard knowledge distillation loss (mode-seeking behavior).

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        labels: Original labels for computing padding mask. If None (e.g., ViT models), no masking is applied.
        kd_config_loss: LossConfig object
    """
    return compute_kld(
        input_logits=student_logits,  # Denominator (student distribution)
        target_logits=teacher_logits,  # Numerator (teacher distribution)
        labels=labels,
        kd_config_loss=kd_config_loss,
    )


def compute_kld_reverse(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None,
    kd_config_loss: LossConfig,
) -> torch.Tensor:
    """
    Reverse KL divergence: compute student distribution relative to teacher distribution.
    KL(student || teacher) = sum(student_probs * log(student_probs / teacher_probs))
    This formulation exhibits mode-covering behavior.

    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        labels: Original labels for computing padding mask. If None (e.g., ViT models), no masking is applied.
        kd_config_loss: LossConfig object
    """
    return compute_kld(
        input_logits=teacher_logits,  # Denominator (teacher distribution)
        target_logits=student_logits,  # Numerator (student distribution)
        labels=labels,
        kd_config_loss=kd_config_loss,
    )


def compute_teacher_confidence(
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None,
    kd_config_loss: LossConfig,
) -> torch.Tensor:
    """
    Compute teacher confidence coefficient with optional entropy-based calculation.

    Method 1 (use_entropy=False): Original CAKLD approach
    γ = (1/B) ∑_{b=1}^B [(1/L_b) ∑_{i∈valid_b} P_T(y_{b,i} | x_b, y_{b,<i})]

    Method 2 (use_entropy=True): Entropy-based approach
    γ = 1 - (1/B) ∑_{b=1}^B [(1/L_b) ∑_{i∈valid_b} H(topk(P_T^{b,i}))] / log(|V_k|)
    γ = 1 - (1/B) ∑_{b=1}^B [(1/L_b) ∑_{i∈valid_b} min(H(P_T^{b,i}),log(|V_k|))] / log(|V_k|)

    Notation:
    - B: batch size
    - L_b: number of valid tokens in sample b
    - valid_b: set of non-padding positions in sample b
    - P_T^{b,i}: teacher's probability distribution at position i in sample b
    - H(P_T^{b,i}) = -∑_{c=1}^{|V|} P_T^{b,i}(c) × log(P_T^{b,i}(c))
    - |V|: vocabulary size
    - |V_k|: top-k vocabulary size

    Args:
        teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]. If None (e.g., ViT models), uses entropy-based method or default value.
        kd_config_loss: LossConfig object containing padding_id, is_router_logits, temperature, use_entropy, etc.

    Returns:
        gamma: Confidence coefficient in [0.0, 1.0]
    """
    # Extract parameters from configuration
    padding_id = kd_config_loss.padding_id
    is_router_logits = kd_config_loss.is_router_logits
    use_entropy = kd_config_loss.use_entropy
    k = (
        kd_config_loss.confidence_k
    )  # Controls H(uniform-k): smaller k favors Forward KLD (mode covering)

    teacher_logits = teacher_logits
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # Handle case when labels is None (e.g., ViT models without padding)
    if labels is None:
        # No masking - all elements are valid
        if use_entropy:
            # Entropy-based confidence
            entropy = -torch.sum(
                teacher_probs * torch.log(teacher_probs + 1e-8), dim=-1
            )
            max_entropy = torch.log(
                torch.tensor(k, dtype=torch.float, device=teacher_probs.device)
            )
            normalized_entropy = entropy.mean() / max_entropy
            gamma = 1.0 - normalized_entropy
        else:
            # Cannot compute label probability without labels
            # Default to entropy-based or return a default value
            gamma = torch.tensor(0.5, dtype=torch.float, device=teacher_probs.device)

        return torch.clamp(gamma, min=0.0, max=1.0)

    # Original logic when labels is provided
    if not is_router_logits:
        pad_mask = labels.eq(padding_id)
    else:
        pad_mask = labels.view(1, -1).eq(padding_id)
    # Shape considerations for MoE architecture:
    # - teacher_logits: [batch_size, seq_len, vocab_size]
    # - teacher_router_logits: [num_hidden_layers, batch_size*seq_len, num_experts]
    # - labels: [batch_size, seq_len]
    #
    # For router logits, entropy has an extra dimension (num_hidden_layers) due to MoE architecture.
    # The function handles both cases:
    # - Regular logits: pad_mask matches entropy shape directly
    # - Router logits: pad_mask is reshaped to [1, batch_size*seq_len] to broadcast across layers
    #
    # This broadcasting works correctly because masked_fill applies the same [1, seq_len] mask
    # to all hidden layers, which is semantically correct for padding positions.

    if use_entropy:
        # Entropy-based confidence:
        # - Lower entropy → higher confidence → prefer Reverse KLD (mode seeking)
        # - Higher entropy → lower confidence → prefer Forward KLD (mode covering)
        # Compute entropy of teacher distribution
        entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + 1e-8), dim=-1)
        
        # If attentions/past_key_values map, shape: [batch_size, num_heads, seq_len]
        if entropy.dim() != pad_mask.dim():
            while pad_mask.dim() < entropy.dim():
                pad_mask = pad_mask.unsqueeze(1)
        
        entropy = entropy.masked_fill(
            pad_mask, 0.0
        )  # shape: [num_hidden_layers, batch_size*seq_len]

        valid_lengths = (~pad_mask).sum(dim=-1).float().clamp(min=1)
        sample_avg_entropy = entropy.sum(dim=-1) / valid_lengths

        # Maximum entropy based on top-k uniform distribution
        max_entropy = torch.log(
            torch.tensor(k, dtype=torch.float, device=teacher_probs.device)
        )
        normalized_entropy = sample_avg_entropy / max_entropy
        gamma = 1.0 - normalized_entropy.mean()

        # # LOG
        # print(f"avg_entropy={sample_avg_entropy}, max_entropy={max_entropy}")
    else:
        # Original method: based on label token probability
        # Replace padding_id with 0 to avoid invalid index in gather
        # (padding positions will be masked out later)
        safe_labels = labels.clone()
        safe_labels[pad_mask] = 0
        
        target_expanded = safe_labels.unsqueeze(-1)
        target_probs = torch.gather(
            teacher_probs, dim=-1, index=target_expanded
        ).squeeze(-1)
        target_probs = target_probs.masked_fill(pad_mask, 0.0)

        valid_lengths = (~pad_mask).sum(dim=-1).float().clamp(min=1)
        sample_avg_probs = target_probs.sum(dim=-1) / valid_lengths
        gamma = sample_avg_probs.mean()

    return torch.clamp(gamma, min=0.0, max=1.0)


def compute_kld_confidence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None,
    kd_config_loss: LossConfig,
) -> torch.Tensor:
    """
    Confidence-Aware KL Divergence (CAKLD) from paper:
    "BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation"

    CAKLD = γ*Reverse_KL(student || teacher) + (1-γ)*Forward_KL(teacher || student)
    where γ = E_{(x,y)~D}[1/|{y}| ∑_{i=1}^{|y|} P_T(y_i | x, y_{<i})]

    Args:
        student_logits: Student model logits [batch_size, seq_len, vocab_size]
        teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
        labels: Target labels [batch_size, seq_len]. If None (e.g., ViT models), no masking is applied.
        kd_config_loss: LossConfig object containing all KLD-related parameters
    """
    # 1. Compute γ (teacher's average token probability / confidence coefficient)
    # When teacher confidence is high (γ ≈ 1):
    #   Favor KL(student || teacher) - force student to imitate teacher (mode covering)
    # When teacher confidence is low (γ ≈ 0):
    #   Favor KL(teacher || student) - allow student to make its own decisions (mode seeking)
    gamma = compute_teacher_confidence(
        teacher_logits=teacher_logits, labels=labels, kd_config_loss=kd_config_loss
    )

    # 2. Compute Reverse KL: KL(student || teacher)
    reverse_kl = compute_kld_reverse(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        kd_config_loss=kd_config_loss,
    )

    # 3. Compute Forward KL: KL(teacher || student)
    forward_kl = compute_kld_forward(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        labels=labels,
        kd_config_loss=kd_config_loss,
    )

    # 4. Compute weighted CAKLD
    cakld = gamma * reverse_kl + (1 - gamma) * forward_kl

    # # LOG
    # print(f"gamma={gamma}, kld_r={reverse_kl}, kld_f={forward_kl}")
    # print(f"cakld={cakld}")

    return cakld


# ========================================= State Distillation: MSE =========================================
def compute_mse(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    labels: torch.Tensor | None,
    kd_config_loss: LossConfig,
) -> torch.Tensor:
    """
    Feature distillation using Mean Squared Error (MSE) loss.
    MSE Loss = ||student_features - teacher_features||^2

    Supported Shapes:
    1. hidden_states:
       - input/target: [batch_size, seq_len, hidden_size]
       - labels: [batch_size, seq_len]
    2. past_key_values, key [0] or value [1]:
       - input/target: [batch_size, num_key_value_heads, seq_len, head_dim]
       - labels: [batch_size, seq_len]

    Args:
        student_features: Student model features
        teacher_features: Teacher model features
        labels: Original labels for computing padding mask. If None, no masking is applied.
        kd_config_loss: LossConfig object.

    Returns:
        MSE loss value
    """
    # Extract parameters from configuration
    padding_id = kd_config_loss.padding_id
    reduction = kd_config_loss.reduction
    is_past_key_values = student_features.dim() == 4

    if student_features.shape != teacher_features.shape:
        raise ValueError(
            f"Student and teacher features must have same shape! "
            f"Got student: {student_features.shape}, teacher: {teacher_features.shape}"
        )

    # Compute element-wise MSE
    # [batch_size, seq_len, hidden_size] or [batch_size, num_key_value_heads, seq_len, head_dim]
    mse = F.mse_loss(student_features, teacher_features, reduction="none")

    # Apply padding mask if labels is provided
    if labels is not None:
        # 1. Create Mask
        if not is_past_key_values and student_features.dim() == 3:
            pad_mask = labels.eq(padding_id).unsqueeze(-1)  # [B, S, 1]
            feature_dim_size = student_features.shape[-1]
        elif is_past_key_values:
            # [B, H, S, D] -> Mask needs to be [B, 1, S, 1] to broadcast
            pad_mask = labels.eq(padding_id).unsqueeze(1).unsqueeze(-1)
            feature_dim_size = student_features.shape[1] * student_features.shape[-1]
        else:
            # Fallback or error handling if needed, assuming 3D here for safety based on logic
            pad_mask = labels.eq(padding_id).unsqueeze(-1)
            feature_dim_size = student_features.shape[-1]

        # 2. Apply Mask
        mse = mse.masked_fill(pad_mask, 0.0)

        # 3. Compute number of valid tokens (scalar count per batch)
        valid_token_count = (
            (~labels.eq(padding_id)).sum(dim=-1).float().clamp(min=1)
        )  # [batch_size]

        if reduction == "sum":
            return mse.sum()

        elif reduction == "mean":
            # Average over all non-padding positions
            # Denominator = Total valid tokens * features per token
            total_valid_elements = valid_token_count.sum() * feature_dim_size
            return mse.sum() / total_valid_elements

        elif reduction == "batch_mean":
            # Average per sample, then average across batch
            # Flatten all dims except batch to sum easily: [B, ...] -> [B, -1]
            mse_per_sample = mse.view(mse.size(0), -1).sum(dim=-1)  # [batch_size]

            # Normalize by (valid tokens * feature size)
            mse_per_sample = mse_per_sample / (valid_token_count * feature_dim_size)
            return mse_per_sample.mean()

        elif reduction == "none":
            return mse
        else:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
    else:
        # No labels provided or 2D features
        if reduction == "sum":
            return mse.sum()
        elif reduction == "mean":
            return mse.mean()
        elif reduction == "batch_mean":
            # Flatten [B, ...] -> [B, N] -> mean(1) -> mean()
            return mse.view(mse.size(0), -1).mean(dim=-1).mean()
        elif reduction == "none":
            return mse
        else:
            raise ValueError(f"Unsupported reduction mode: {reduction}")
