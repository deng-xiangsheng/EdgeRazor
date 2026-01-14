"""
EdgeRazor unified configuration module.

This module provides a unified configuration class that can load both QAT and KD
configurations from a single file or separate sources.
"""
# ruff: noqa: UP035

import json
from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .kd.util import DistillConfig
from .qat.util import QuantConfig

_original_encoder_default = json.JSONEncoder.default

def _patched_default(self, obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    return _original_encoder_default(self, obj)

json.JSONEncoder.default = _patched_default


class EdgeRazorConfig:
    """
    Unified configuration class for EdgeRazor framework.
    
    This class can load both QAT (Quantization-Aware Training) and KD (Knowledge Distillation)
    configurations from a single YAML/JSON file or from separate sources.
    
    Configuration file format (unified):
        ```yaml
        # QAT Configuration
        method: QAT
        qat_configuration:
          select: ...
          function: ...
          training: all
        
        # KD Configuration
        method: KD
        kd_configuration:
          loss_task_alpha: 1.0
          loss_1: ...
          loss_2: ...
        ```
    
    Examples:
        >>> # Load from unified config file
        >>> config = EdgeRazorConfig.from_yaml("config.yaml")
        >>> print(config.qat_config)  # QuantConfig object
        >>> print(config.kd_config)   # DistillConfig object
        
        >>> # Load from separate files
        >>> config = EdgeRazorConfig.from_yaml(
        ...     qat_yaml="qat_config.yaml",
        ...     kd_yaml="kd_config.yaml"
        ... )
        
        >>> # Create from dict
        >>> config = EdgeRazorConfig.from_dict({
        ...     'qat_configuration': {...},
        ...     'kd_configuration': {...}
        ... })
    """
    
    def __init__(
        self,
        qat_config: QuantConfig | None = None,
        kd_config: DistillConfig | None = None,
    ):
        """
        Initialize EdgeRazorConfig with QAT and/or KD configurations.
        
        Args:
            qat_config: QuantConfig object for QAT (optional)
            kd_config: DistillConfig object for KD (optional)
        
        Raises:
            ValueError: If both qat_config and kd_config are None
        """
        if qat_config is None and kd_config is None:
            raise ValueError(
                "At least one of qat_config or kd_config must be provided. "
                "Cannot initialize EdgeRazorConfig with both disabled."
            )
        
        self.qat_config = qat_config
        self.kd_config = kd_config
    
    @classmethod
    def load(
        cls,
        config: "EdgeRazorConfig | str | Path | dict | None" = None,
        qat_config: "str | Path | dict | None" = None,
        kd_config: "str | Path | dict | None" = None,
    ) -> "EdgeRazorConfig":
        """
        Universal loader for EdgeRazorConfig from any source.
        
        This is the recommended entry point for loading configurations.
        Automatically detects file format and config type.
        
        Args:
            config: Unified config (file path, dict, or EdgeRazorConfig)
            qat_config: QAT-only config (file path or dict)
            kd_config: KD-only config (file path or dict)
        
        Returns:
            EdgeRazorConfig instance
        
        Examples:
            >>> # From unified YAML/JSON file
            >>> config = EdgeRazorConfig.load("unified_config.yaml")
            
            >>> # From separate files
            >>> config = EdgeRazorConfig.load(
            ...     qat_config="qat.yaml",
            ...     kd_config="kd.yaml"
            ... )
            
            >>> # From dict
            >>> config = EdgeRazorConfig.load({
            ...     'qat_configuration': {...},
            ...     'kd_configuration': {...}
            ... })
            
            >>> # Mixed: QAT from file, KD from dict
            >>> config = EdgeRazorConfig.load(
            ...     qat_config="qat.yaml",
            ...     kd_config={'method': 'KD', 'loss_1': {...}}
            ... )
        """
        # Already an EdgeRazorConfig
        if isinstance(config, EdgeRazorConfig):
            return config
        
        # Unified configuration
        if config is not None:
            if isinstance(config, dict):
                return cls.from_dict(config)
            elif isinstance(config, (str, Path)):
                path = Path(config)
                if path.suffix in ['.yaml', '.yml']:
                    return cls.from_yaml(yaml_path=path)
                elif path.suffix == '.json':
                    return cls.from_json(json_path=path)
                else:
                    raise ValueError(f"Unsupported file format: {path.suffix}")
            else:
                raise TypeError(f"Unsupported config type: {type(config)}")
        
        # Separate configurations
        if qat_config is not None or kd_config is not None:
            qat_cfg = None
            kd_cfg = None
            
            # Load QAT config
            if qat_config is not None:
                if isinstance(qat_config, dict):
                    qat_dict = qat_config.copy()
                    if 'method' not in qat_dict:
                        qat_dict['method'] = 'QAT'
                    elif qat_dict['method'].upper() != 'QAT':
                        raise ValueError(
                            f"Invalid method in qat_config: '{qat_dict['method']}'. "
                            f"Expected 'QAT' but got '{qat_dict['method']}'"
                        )
                    qat_cfg = QuantConfig(qat_dict)
                elif isinstance(qat_config, (str, Path)):
                    path = Path(qat_config)
                    with open(path, encoding='utf-8') as f:
                        qat_data = yaml.safe_load(f)
                    # Check if it has qat_configuration wrapper
                    if 'qat_configuration' in qat_data:
                        qat_data = qat_data['qat_configuration']
                    if 'method' not in qat_data:
                        qat_data['method'] = 'QAT'
                    elif qat_data['method'].upper() != 'QAT':
                        raise ValueError(
                            f"Invalid method in qat_config file '{path}': '{qat_data['method']}'. "
                            f"Expected 'QAT' but got '{qat_data['method']}'"
                        )
                    qat_cfg = QuantConfig(qat_data)
                else:
                    raise TypeError(f"Unsupported qat_config type: {type(qat_config)}")
            
            # Load KD config
            if kd_config is not None:
                if isinstance(kd_config, dict):
                    kd_dict = kd_config.copy()
                    if 'method' not in kd_dict:
                        kd_dict['method'] = 'KD'
                    elif kd_dict['method'].upper() != 'KD':
                        raise ValueError(
                            f"Invalid method in kd_config: '{kd_dict['method']}'. "
                            f"Expected 'KD' but got '{kd_dict['method']}'"
                        )
                    kd_cfg = DistillConfig.from_dict(kd_dict)
                elif isinstance(kd_config, (str, Path)):
                    path = Path(kd_config)
                    with open(path, encoding='utf-8') as f:
                        kd_data = yaml.safe_load(f)
                    # Check if it has kd_configuration wrapper
                    if 'kd_configuration' in kd_data:
                        kd_data = kd_data['kd_configuration']
                    if 'method' not in kd_data:
                        kd_data['method'] = 'KD'
                    elif kd_data['method'].upper() != 'KD':
                        raise ValueError(
                            f"Invalid method in kd_config file '{path}': '{kd_data['method']}'. "
                            f"Expected 'KD' but got '{kd_data['method']}'"
                        )
                    kd_cfg = DistillConfig.from_dict(kd_data)
                else:
                    raise TypeError(f"Unsupported kd_config type: {type(kd_config)}")
            
            return cls(qat_config=qat_cfg, kd_config=kd_cfg)
        
        raise ValueError("No configuration provided")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EdgeRazorConfig":
        """
        Create EdgeRazorConfig from a dictionary.
        
        Supports three formats:
        1. Unified: {'qat_configuration': {...}, 'kd_configuration': {...}}
        2. QAT-only: {'method': 'QAT', 'select': {...}, 'function': {...}}
        3. KD-only: {'method': 'KD', 'loss_1': {...}, 'loss_2': {...}}
        
        Args:
            config_dict: Configuration dictionary
        
        Returns:
            EdgeRazorConfig instance
        """
        qat_config = None
        kd_config = None
        
        # Check for unified format (has wrapper keys)
        has_qat_wrapper = 'qat_configuration' in config_dict
        has_kd_wrapper = 'kd_configuration' in config_dict
        
        if has_qat_wrapper or has_kd_wrapper:
            # Unified configuration format
            if has_qat_wrapper:
                qat_dict = config_dict['qat_configuration'].copy()
                if 'method' not in qat_dict:
                    qat_dict['method'] = 'QAT'
                elif qat_dict['method'].upper() != 'QAT':
                    raise ValueError(
                        f"Invalid method in qat_configuration: '{qat_dict['method']}'. "
                        f"Expected 'QAT' but got '{qat_dict['method']}'"
                    )
                qat_config = QuantConfig(qat_dict)
            
            if has_kd_wrapper:
                kd_dict = config_dict['kd_configuration'].copy()
                if 'method' not in kd_dict:
                    kd_dict['method'] = 'KD'
                elif kd_dict['method'].upper() != 'KD':
                    raise ValueError(
                        f"Invalid method in kd_configuration: '{kd_dict['method']}'. "
                        f"Expected 'KD' but got '{kd_dict['method']}'"
                    )
                kd_config = DistillConfig.from_dict(kd_dict)
        else:
            # Single configuration - auto-detect type
            method = config_dict.get('method', '').upper()
            has_qat_keys = any(k in config_dict for k in ['select', 'function', 'training'])
            has_kd_keys = any(k.startswith('loss_') for k in config_dict.keys())
            
            if method == 'QAT' or has_qat_keys:
                # QAT configuration
                qat_dict = config_dict.copy()
                if 'method' not in qat_dict:
                    qat_dict['method'] = 'QAT'
                qat_config = QuantConfig(qat_dict)
            elif method == 'KD' or has_kd_keys:
                # KD configuration
                kd_dict = config_dict.copy()
                if 'method' not in kd_dict:
                    kd_dict['method'] = 'KD'
                kd_config = DistillConfig.from_dict(kd_dict)
        
        return cls(qat_config=qat_config, kd_config=kd_config)
    
    @classmethod
    def from_yaml(
        cls,
        yaml_path: Union[str, Path] | None = None,
        qat_yaml: Union[str, Path] | None = None,
        kd_yaml: Union[str, Path] | None = None,
    ) -> "EdgeRazorConfig":
        """
        Load EdgeRazorConfig from YAML file(s).
        
        Args:
            yaml_path: Path to unified YAML file containing both QAT and KD configurations
            qat_yaml: Path to separate QAT YAML file (alternative to yaml_path)
            kd_yaml: Path to separate KD YAML file (alternative to yaml_path)
        
        Returns:
            EdgeRazorConfig instance
        
        Raises:
            ValueError: If neither yaml_path nor (qat_yaml or kd_yaml) is provided
            FileNotFoundError: If specified file does not exist
        
        Examples:
            >>> # From unified config
            >>> config = EdgeRazorConfig.from_yaml("unified_config.yaml")
            
            >>> # From separate configs
            >>> config = EdgeRazorConfig.from_yaml(
            ...     qat_yaml="qat_config.yaml",
            ...     kd_yaml="kd_config.yaml"
            ... )
            
            >>> # Only QAT
            >>> config = EdgeRazorConfig.from_yaml(qat_yaml="qat_config.yaml")
        """
        qat_config = None
        kd_config = None
        
        if yaml_path is not None:
            # Load from unified configuration file
            yaml_path = Path(yaml_path)
            if not yaml_path.exists():
                raise FileNotFoundError(f"Config file not found: {yaml_path}")
            
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            return cls.from_dict(config_dict)
        
        elif qat_yaml is not None or kd_yaml is not None:
            # Load from separate configuration files
            if qat_yaml is not None:
                qat_yaml = Path(qat_yaml)
                if not qat_yaml.exists():
                    raise FileNotFoundError(f"QAT config file not found: {qat_yaml}")
                qat_config = QuantConfig.from_yaml(qat_yaml)
            
            if kd_yaml is not None:
                kd_yaml = Path(kd_yaml)
                if not kd_yaml.exists():
                    raise FileNotFoundError(f"KD config file not found: {kd_yaml}")
                kd_config = DistillConfig.from_yaml(kd_yaml)
            
            return cls(qat_config=qat_config, kd_config=kd_config)
        
        else:
            raise ValueError(
                "Must provide either 'yaml_path' (unified config) or "
                "'qat_yaml'/'kd_yaml' (separate configs)"
            )
    
    @classmethod
    def from_json(
        cls,
        json_path: Union[str, Path] | None = None,
        qat_json: Union[str, Path] | None = None,
        kd_json: Union[str, Path] | None = None,
    ) -> "EdgeRazorConfig":
        """
        Load EdgeRazorConfig from JSON file(s).
        
        Args:
            json_path: Path to unified JSON file containing both QAT and KD configurations
            qat_json: Path to separate QAT JSON file (alternative to json_path)
            kd_json: Path to separate KD JSON file (alternative to json_path)
        
        Returns:
            EdgeRazorConfig instance
        
        Raises:
            ValueError: If neither json_path nor (qat_json or kd_json) is provided
            FileNotFoundError: If specified file does not exist
        
        Examples:
            >>> # From unified config
            >>> config = EdgeRazorConfig.from_json("unified_config.json")
            
            >>> # From separate configs
            >>> config = EdgeRazorConfig.from_json(
            ...     qat_json="qat_config.json",
            ...     kd_json="kd_config.json"
            ... )
        """
        qat_config = None
        kd_config = None
        
        if json_path is not None:
            # Load from unified configuration file
            json_path = Path(json_path)
            if not json_path.exists():
                raise FileNotFoundError(f"Config file not found: {json_path}")
            
            with open(json_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
        
        elif qat_json is not None or kd_json is not None:
            # Load from separate configuration files
            if qat_json is not None:
                qat_json = Path(qat_json)
                if not qat_json.exists():
                    raise FileNotFoundError(f"QAT config file not found: {qat_json}")
                qat_config = QuantConfig.from_json(qat_json)
            
            if kd_json is not None:
                kd_json = Path(kd_json)
                if not kd_json.exists():
                    raise FileNotFoundError(f"KD config file not found: {kd_json}")
                kd_config = DistillConfig.from_json(kd_json)
            
            return cls(qat_config=qat_config, kd_config=kd_config)
        
        else:
            raise ValueError(
                "Must provide either 'json_path' (unified config) or "
                "'qat_json'/'kd_json' (separate configs)"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert EdgeRazorConfig to dictionary.
        
        Returns:
            Dictionary with 'qat_configuration' and/or 'kd_configuration' keys
        """
        result = {}
        
        if self.qat_config is not None:
            result['qat_configuration'] = self.qat_config.to_dict()
        
        if self.kd_config is not None:
            result['kd_configuration'] = self.kd_config.to_dict()
        
        return result
    
    def to_yaml(self, yaml_path: Union[str, Path]):
        """
        Save EdgeRazorConfig to a unified YAML file.
        
        Args:
            yaml_path: Path where to save the YAML configuration
        """
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
        """
        Save EdgeRazorConfig to a unified JSON file.
        
        Args:
            json_path: Path where to save the JSON configuration
        """
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @property
    def has_qat(self) -> bool:
        """Check if QAT configuration is present"""
        return self.qat_config is not None
    
    @property
    def has_kd(self) -> bool:
        """Check if KD configuration is present"""
        return self.kd_config is not None
    
    def __repr__(self) -> str:
        """String representation of EdgeRazorConfig"""
        parts = []
        if self.has_qat:
            parts.append("QAT=enabled")
        else:
            parts.append("QAT=disabled")
        
        if self.has_kd:
            parts.append("KD=enabled")
        else:
            parts.append("KD=disabled")
        
        return f"EdgeRazorConfig({', '.join(parts)})"
    
    class JSONEncoder(json.JSONEncoder):
        """Custom JSON encoder that handles EdgeRazorConfig objects."""
        def default(self, obj):
            if isinstance(obj, EdgeRazorConfig):
                return obj.to_dict()
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            return super().default(obj)
    
    def __reduce__(self):
        """Support for pickle serialization."""
        return (self.__class__.from_dict, (self.to_dict(),))
