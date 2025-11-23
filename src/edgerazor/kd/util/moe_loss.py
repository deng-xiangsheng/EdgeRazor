import torch


# ========================================= MoE =========================================
# router_z_loss_func(stacked_router_logits) = router_z_losses_func(output.router_logits)
def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://huggingface.co/papers/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]
            Same result of shape [batch_size, sequence_length, num_experts] or [1, batch_size*sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def router_z_losses_func(router_logits_tuple: tuple) -> float:
    r"""
    Compute the average router z-loss implemented in PyTorch.
    
    MoE model output of router_logits: tuple of tensor(shape=[batch_size, sequence_length, num_experts])
    len(tuple)=num_decoder_layers
    
    Args:
        router_logits_tuple (`tuple`):
            Tuple of router logits tensors, one for each decoder layer.
            Each tensor has shape [batch_size*sequence_length, num_experts]
    
    Returns:
        Scalar average router z-loss across all layers.
    """
    if not router_logits_tuple:
        return 0.0
    
    total_z_loss = 0.0
    num_layers = len(router_logits_tuple)
    
    for router_logits in router_logits_tuple:
        layer_z_loss = router_z_loss_func(router_logits)
        total_z_loss += layer_z_loss
    
    return total_z_loss / num_layers
