"""Knowledge Distillation (KD) module for EdgeRazor

This module provides knowledge distillation functionality for model compression.

Supported distillation strategies:
- Logits-based: KLD forward/reverse/confidence
- Feature-based: Hidden states distillation (future)
- Attention-based: Attention maps distillation (future)

Examples:
    >>> from edgerazor import KD
    >>> kd = KD("configs/kd_logits.yaml")
    >>> total_loss, loss_dict = kd.compute_loss(
    ...     student_outputs, teacher_outputs, labels
    ... )
"""
# ruff: noqa: F401

from .kd import KD
from .util import (
    DistillConfig,
    LossConfig,
    compute_kld_confidence,
    compute_kld_forward,
    compute_kld_reverse,
    compute_state_distill,
)

__all__ = [
    # Main KD class
    "KD",
    # Configuration classes
    "DistillConfig",
    "LossConfig",
    # Loss functions
    "compute_kld_forward",
    "compute_kld_reverse",
    "compute_kld_confidence",
    "compute_state_distill",
]

