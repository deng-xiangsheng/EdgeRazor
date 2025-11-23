"""
Distillation configuration classes for knowledge distillation

Supports multi-loss configuration format:
loss_1:
  loss_type: logits
  loss_function: kldc
  alpha: 0.7
  ...
loss_2:
  loss_type: features
  loss_function: fd
  ...
"""

import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

import yaml


@dataclass
class LossConfig:
    """Configuration for a single loss function"""
    
    # Loss type identifier (used for configuration validation only)
    loss_type: str = "logits"  # logits, features, attention, hidden
    
    # Loss function name (abbreviated form)
    loss_function: str = "compute_kld_reverse"  # compute_kld_reverse, compute_kld_confidence, compute_fd, etc.
    
    # Loss weight: distill_loss = alpha_1 * loss_1 + ... + alpha_n * loss_n
    alpha: float = 0.5
    
    # KLD-related parameters
    temperature: float = 2.0
    use_entropy: bool = True
    padding_id: int = -100
    confidence_k: int = 5
    is_router_logits: bool = False
    
    # General parameters
    reduction: str = "batch_mean"  # sum, mean, batch_mean, none
    normalize: bool = False  # Whether to normalize features (for fd function)
    
    # Layer selection for hidden_states distillation
    # Can be: int (single layer), list of ints (multiple layers), str (predefined names), list of strs (predefined names), or None (all layers)
    # For hidden_states, `0` usually refers to the embeddings layer, `1-last_l_id` are transformer layers
    # Predefined string choices: "low", "mid", "high"
    # Examples: 0, -1, [0, 3, 6], [1, -1], "low", "mid", "high", ["low", "mid", "high"]
    layer_index: int | str | list | None = None
    
    def __post_init__(self):
        """Validate configuration parameters"""
        valid_loss_types = ["logits", "hidden_states", "attention"] # "router_logits" TODO
        if self.loss_type not in valid_loss_types:
            raise ValueError(f"loss_type must be one of {valid_loss_types}, got '{self.loss_type}'")
        
        valid_reductions = ["sum", "mean", "batch_mean", "none"]
        if self.reduction not in valid_reductions:
            raise ValueError(f"reduction must be one of {valid_reductions}, got '{self.reduction}'")
        
        # Validate layer_index if it's a string or list of strings
        valid_layer_names = ["low", "mid", "high"]
        if isinstance(self.layer_index, str):
            if self.layer_index not in valid_layer_names:
                raise ValueError(
                    f"layer_index string must be one of {valid_layer_names}, got '{self.layer_index}'"
                )
        elif isinstance(self.layer_index, list):
            for idx in self.layer_index:
                if isinstance(idx, str) and idx not in valid_layer_names:
                    raise ValueError(
                        f"layer_index string must be one of {valid_layer_names}, got '{idx}'"
                    )


@dataclass
class DistillConfig:
    """Knowledge distillation configuration class - supports multi-loss configuration"""
    
    method: str = "KD"
    
    # Task loss weight coefficient
    loss_task_alpha: float = 1.0
    
    # Multiple loss configurations: {loss_1: LossConfig, loss_2: LossConfig, ...}
    losses: Dict[str, LossConfig] = field(default_factory=dict)
    
    # MoE-related parameters (used by EdgeRazorTrainer, specific to MoE architectures)
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.method != "KD":
            raise ValueError(f"method must be 'KD', got '{self.method}'")
        
        if not self.losses:
            raise ValueError("At least one loss configuration is required")
        
        # Ensure all values in losses are LossConfig instances
        for key, loss in list(self.losses.items()):
            if isinstance(loss, dict):
                self.losses[key] = LossConfig(**loss)
            elif not isinstance(loss, LossConfig):
                raise ValueError(f"Loss '{key}' must be a LossConfig or dict, got {type(loss)}")
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "DistillConfig":
        """Create configuration from dictionary"""
        config_dict = config_dict.copy()
        
        # Handle kd_configuration wrapper (for unified config format)
        if 'kd_configuration' in config_dict:
            config_dict = config_dict['kd_configuration'].copy()
        
        # Extract loss_1, loss_2, ... configurations
        losses = {}
        keys_to_remove = []
        for key, value in config_dict.items():
            if key.startswith("loss_") and isinstance(value, dict):
                losses[key] = LossConfig(**value)
                keys_to_remove.append(key)
        
        # Remove processed loss configurations
        for key in keys_to_remove:
            del config_dict[key]
        
        config_dict['losses'] = losses
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "DistillConfig":
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "DistillConfig":
        """Load configuration from JSON file"""
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = OrderedDict()
        result['method'] = self.method
        
        # Add task loss alpha
        if self.loss_task_alpha != 1.0:
            result['loss_task_alpha'] = self.loss_task_alpha
        
        # Add loss configurations
        for key, loss in self.losses.items():
            result[key] = asdict(loss)
        
        # Add MoE-related parameters
        if self.router_aux_loss_coef != 0.01:
            result['router_aux_loss_coef'] = self.router_aux_loss_coef
        if self.router_z_loss_coef != 0.001:
            result['router_z_loss_coef'] = self.router_z_loss_coef
        
        return result
    
    def to_yaml(self, yaml_path: Union[str, Path]):
        """Save as YAML file"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True
            )
    
    def to_json(self, json_path: Union[str, Path]):
        """Save as JSON file"""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        """String representation"""
        loss_info = ", ".join([f"{k}={v.loss_function}" for k, v in self.losses.items()])
        return f"DistillConfig(method={self.method}, losses=[{loss_info}])"
