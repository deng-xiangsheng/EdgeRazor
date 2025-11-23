"""
Distillation loss function configuration and mappings

This module provides mappings between loss function names (strings) and their
actual implementations for knowledge distillation.
"""

from .distill_function import (
    compute_fd,
    compute_kld_confidence,
    compute_kld_forward,
    compute_kld_reverse,
)

# Mapping from string names to distillation loss functions
# Naming convention: compute_xxx -> abbreviation
# - compute_kld_forward -> kldf
# - compute_kld_reverse -> kldr
# - compute_kld_confidence -> kldc
# - compute_fd -> fd
distill_function_map = {
    # Abbreviated format (recommended)
    'kldf': compute_kld_forward,
    'kldr': compute_kld_reverse,
    'kldc': compute_kld_confidence,
    'fd': compute_fd,
    
    # Backward compatibility
    'kld_forward': compute_kld_forward,
    'kld_reverse': compute_kld_reverse,
    'kld_confidence': compute_kld_confidence,
    
    # Full function names (backward compatibility)
    'compute_kld_forward': compute_kld_forward,
    'compute_kld_reverse': compute_kld_reverse,
    'compute_kld_confidence': compute_kld_confidence,
    'compute_fd': compute_fd,
}


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
