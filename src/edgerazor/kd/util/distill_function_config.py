"""
Distillation loss function configuration and mappings

This module provides mappings between loss function names (strings) and their
actual implementations for knowledge distillation.
"""

from .distill_function import (
    compute_kld_confidence,
    compute_kld_forward,
    compute_kld_reverse,
    compute_state_distill,
)

# Collect all distillation loss functions automatically
_distill_functions = [
    compute_kld_forward,
    compute_kld_reverse,
    compute_kld_confidence,
    compute_state_distill,
]

# Build the map automatically: function_name -> function
distill_function_map = {func.__name__: func for func in _distill_functions}


def get_distill_function(function_name: str):
    """
    Get distillation loss function by name
    
    Args:
        function_name: Name of the distillation loss function
        
    Returns:
        Distillation loss function
        
    Raises:
        ValueError: If function name is not found
    """
    if function_name not in distill_function_map:
        available_functions = ', '.join(distill_function_map.keys())
        raise ValueError(
            f"Unknown distillation loss function: '{function_name}'. "
            f"Available functions: {available_functions}"
        )
    
    return distill_function_map[function_name]
