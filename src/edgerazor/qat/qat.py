from pathlib import Path

import torch.nn as nn
from transformers import PreTrainedModel

from ..log import get_logger
from .block import (
    QKVCacheLlamaAttention,
    QKVCacheOlmoeAttention,
    QKVCacheOlmoeFlashAttention2,
    QKVCacheOlmoeSdpaAttention,
    QKVCacheQwen2_5OmniAttention,
    QKVCacheQwen3Attention,
    QKVCacheQwen3MoeAttention,
    QMultiheadAttention,
)
from .module import QConv1d, QConv2d, QConv3d, QEmbedding, QLinear
from .quantize import apply_quantization, replace_applied_quantized_weights
from .util import QuantConfig, QuantSelector


class QAT:
    """
    Quantization Aware Training (QAT) implementation for EdgeRazor framework.
    
    This class provides quantization-aware training functionality for PyTorch/Transformers neural networks,
    enabling deployment-ready quantized models for edge devices.
    
    Quantized model weights:
    - `1-bit`: {-1, 1} * scaling_factor
    - `1.58-bit` (ternary): {-1, 0, 1} * scaling_factor
    - `2-bit`: {-2, -1, 0, 1} * scaling_factor
    - `4-bit`: {-8, -6, ..., 0, ..., 6, 7} * scaling_factor
    - `8-bit`: {-128, -127, ..., 0, ..., 127} * scaling_factor
    """
    # Default quantized module classes
    qlinear_cls_default = QLinear
    qembedding_cls_default = QEmbedding
    qconv1d_cls_default = QConv1d
    qconv2d_cls_default = QConv2d
    qconv3d_cls_default = QConv3d
    qmultiheadattention_cls_default = QMultiheadAttention
    qkvcacheolmoeattention_cls_default = QKVCacheOlmoeAttention
    qkvcacheolmoeflashattention2_cls_default = QKVCacheOlmoeFlashAttention2
    qkvcacheolmoesdpaattention_cls_default = QKVCacheOlmoeSdpaAttention
    qkvcacheqwen2_5omniattention_cls_default = QKVCacheQwen2_5OmniAttention
    qkvcacheqwen3attention_cls_default = QKVCacheQwen3Attention
    qkvcacheqwen3moeattention_cls_default = QKVCacheQwen3MoeAttention
    qkvcachellamaattention_cls_default = QKVCacheLlamaAttention

    def __init__(self, config: dict | str | Path | QuantConfig):
        """
        Initialize QAT with configuration.
        
        Args:
            config: Configuration for quantization. Can be:
                - dict: Python dictionary containing configuration
                - str/Path: Path to YAML (.yaml/.yml) or JSON (.json) configuration file
                - QuantConfig: Pre-constructed QuantConfig object
                
        Examples:
            >>> # From YAML file
            >>> qat = QAT("configs/q_resnet_w1.58_a16.yaml")
            
            >>> # From JSON file
            >>> qat = QAT("configs/q_resnet_w4_a8.json")
            
            >>> # From Python dict
            >>> qat = QAT({"method": "QAT", "select": {...}, "function": {...}})
            
            >>> # From QuantConfig object
            >>> config = QuantConfig.from_yaml("config.yaml")
            >>> qat = QAT(config)
        """
        # Get component logger
        self.logger = get_logger('QAT')

        # Log initialization
        self.logger.info("Initializing Quantization Aware Training (QAT)")

        # Load configuration
        self.config = self._load_configuration(config)

        # Log configuration details
        self._log_configuration()

        # Initialize quantization selector
        self.selector = QuantSelector(self.config.select)
        self.logger.info("Quantization selector initialized")

        self.logger.info("QAT initialization completed")

    def _load_configuration(self, config: dict | str | Path | QuantConfig) -> QuantConfig:
        """
        Load and parse configuration from various input types.
        Do not support JSON or YAML strings using QAT.
        
        Args:
            config: Configuration input in various formats:
                - QuantConfig: Pre-constructed configuration object
                - dict: Python dictionary with configuration parameters
                - str/Path: File path to YAML (.yaml/.yml) or JSON (.json) configuration file
        
        Returns:
            QuantConfig: Parsed configuration object
            
        Raises:
            TypeError: If config type is not supported
            ValueError: If file format is not supported
            FileNotFoundError: If configuration file does not exist
        """
        try:
            if isinstance(config, QuantConfig):
                # Already a QuantConfig object - use directly
                self.logger.info("Using provided QuantConfig object")
                return config
                
            elif isinstance(config, dict):
                # Python dictionary - construct QuantConfig
                self.logger.info("Loading configuration from Python dictionary")
                return QuantConfig(config)
                
            elif isinstance(config, (str, Path)):
                # File path - detect format by extension
                config_path = Path(config) if isinstance(config, str) else config
                self.logger.info(f"Loading configuration from: {config_path}")
                
                suffix = config_path.suffix.lower()
                if suffix in ['.yaml', '.yml']:
                    loaded_config = QuantConfig.from_yaml(config_path)
                    self.logger.info("Configuration loaded from YAML file")
                    return loaded_config
                elif suffix == '.json':
                    loaded_config = QuantConfig.from_json(config_path)
                    self.logger.info("Configuration loaded from JSON file")
                    return loaded_config
                else:
                    raise ValueError(
                        f"Unsupported configuration file format: {suffix}. "
                        f"Supported formats: .yaml, .yml, .json"
                    )
            else:
                raise TypeError(
                    f"Invalid configuration type: {type(config).__name__}. "
                    f"Expected: dict, str, Path, or QuantConfig"
                )
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def _log_configuration(self):
        """Log detailed configuration information."""
        self.logger.info("=== QAT Configuration Details ===")

        # Log method
        self.logger.info(f"Method: {self.config.method}")

        # Log function configuration
        self.logger.info("Function Configuration (Global Defaults):")
        self.logger.info(f"  Weight: {self.config.function.weight_function}")
        self.logger.info(f"    Scale Factor: {self.config.function.w_scale_factor}")
        self.logger.info(f"    Block Size: {self.config.function.w_block_size}")
        self.logger.info(f"    Mixed Precision Prop: {self.config.function.w_mixed_precision_prop}")
        self.logger.info(f"    Is Quantized: {self.config.function.is_w_quantized}")
        self.logger.info(f"  State (Activation): {self.config.function.activation_function}")
        self.logger.info(f"    Block Size: {self.config.function.a_block_size}")
        self.logger.info(f"    Mixed Precision Prop: {self.config.function.a_mixed_precision_prop}")
        self.logger.info(f"  State (KV Cache): {self.config.function.kv_cache_function}")
        self.logger.info(f"    Block Size: {self.config.function.kv_block_size}")
        self.logger.info(f"    Mixed Precision Prop: {self.config.function.kv_mixed_precision_prop}")
        self.logger.info(f"  Epsilon: {self.config.function.epsilon}")

        # Log overrides if any
        if hasattr(self.config, 'overrides') and self.config.overrides:
            self.logger.info(f"Per-Layer Overrides: {len(self.config.overrides)} rule(s) defined")
            for idx, override in enumerate(self.config.overrides, 1):
                override_desc = []
                if override.module_type:
                    override_desc.append(f"type={override.module_type}")
                if override.module_name:
                    override_desc.append(f"name={override.module_name}")
                self.logger.info(f"  Override {idx}: [{', '.join(override_desc)}]")
                for key, value in override.overrides.items():
                    self.logger.info(f"    {key}: {value}")
        else:
            self.logger.info("Per-Layer Overrides: None")

        # Log selection configuration
        self.logger.info("Selection Configuration:")
        self.logger.info(f"  Target Types: {self.config.select.target_types}")
        self.logger.info(f"  Target Names: {self.config.select.target_names}")
        self.logger.info(f"  Exclude Types: {self.config.select.exclude_types}")
        self.logger.info(f"  Exclude Names: {self.config.select.exclude_names}")

        # Log training configuration
        self.logger.info(f"Training: {self.config.training}")

        self.logger.info("=== End Configuration Details ===")

    def _check_provided_qclass(
        self,
        qlinear_cls: nn.Module = None,
        qembedding_cls: nn.Module = None,
        qconv1d_cls: nn.Module = None,
        qconv2d_cls: nn.Module = None,
        qconv3d_cls: nn.Module = None,
        qmultiheadattention_cls: nn.Module = None,
        qkvcacheolmoeattention_cls: nn.Module = None,
        qkvcacheolmoeflashattention2_cls: nn.Module = None,
        qkvcacheolmoesdpaattention_cls: nn.Module = None,
        qkvcacheqwen2_5omniattention_cls: nn.Module = None,
        qkvcacheqwen3attention_cls: nn.Module = None,
        qkvcacheqwen3moeattention_cls: nn.Module = None,
        qkvcachellamaattention_cls: nn.Module = None,
    ) -> dict:
        """
        Check and use default quantized classes if not provided.
        
        Args:
            qlinear_cls: Custom quantized Linear class
            qembedding_cls: Custom quantized Embedding class
            qconv1d_cls: Custom quantized Conv1d class
            qconv2d_cls: Custom quantized Conv2d class
            qconv3d_cls: Custom quantized Conv3d class
            qmultiheadattention_cls: Custom quantized MultiheadAttention class
            qkvcacheolmoeattention_cls: Custom quantized QKVCacheOlmoeAttention class
            qkvcacheolmoeflashattention2_cls: Custom quantized QKVCacheOlmoeFlashAttention2 class
            qkvcacheolmoesdpaattention_cls: Custom quantized QKVCacheOlmoeSdpaAttention class
            qkvcacheqwen2_5omniattention_cls: Custom quantized QKVCacheQwen2_5OmniAttention class
            qkvcacheqwen3attention_cls: Custom quantized QKVCacheQwen3Attention class
            qkvcacheqwen3moeattention_cls: Custom quantized QKVCacheQwen3MoeAttention class
            
        Returns:
            Dictionary mapping class parameter names to their final classes
        """
        qclass_map = {}
        
        # Check QLinear
        if qlinear_cls is None:
            qclass_map['qlinear_cls'] = self.qlinear_cls_default
            self.logger.info("Using default QLinear class for Linear layers")
        else:
            qclass_map['qlinear_cls'] = qlinear_cls
            self.logger.info(f"Using custom {qlinear_cls.__name__} class for Linear layers")
        
        # Check QEmbedding
        if qembedding_cls is None:
            qclass_map['qembedding_cls'] = self.qembedding_cls_default
            self.logger.info("Using default QEmbedding class for Embedding layers")
        else:
            qclass_map['qembedding_cls'] = qembedding_cls
            self.logger.info(f"Using custom {qembedding_cls.__name__} class for Embedding layers")
        
        # Check QConv1d
        if qconv1d_cls is None:
            qclass_map['qconv1d_cls'] = self.qconv1d_cls_default
            self.logger.info("Using default QConv1d class for Conv1d layers")
        else:
            qclass_map['qconv1d_cls'] = qconv1d_cls
            self.logger.info(f"Using custom {qconv1d_cls.__name__} class for Conv1d layers")
        
        # Check QConv2d
        if qconv2d_cls is None:
            qclass_map['qconv2d_cls'] = self.qconv2d_cls_default
            self.logger.info("Using default QConv2d class for Conv2d layers")
        else:
            qclass_map['qconv2d_cls'] = qconv2d_cls
            self.logger.info(f"Using custom {qconv2d_cls.__name__} class for Conv2d layers")
        
        # Check QConv3d
        if qconv3d_cls is None:
            qclass_map['qconv3d_cls'] = self.qconv3d_cls_default
            self.logger.info("Using default QConv3d class for Conv3d layers")
        else:
            qclass_map['qconv3d_cls'] = qconv3d_cls
            self.logger.info(f"Using custom {qconv3d_cls.__name__} class for Conv3d layers")
        
        # Check QMultiheadAttention
        if qmultiheadattention_cls is None:
            qclass_map['qmultiheadattention_cls'] = self.qmultiheadattention_cls_default
            self.logger.info("Using default QMultiheadAttention class for MultiheadAttention layers")
        else:
            qclass_map['qmultiheadattention_cls'] = qmultiheadattention_cls
            self.logger.info(f"Using custom {qmultiheadattention_cls.__name__} class for MultiheadAttention layers")
        
        # Check QKVCacheOlmoeAttention
        if qkvcacheolmoeattention_cls is None:
            qclass_map['qkvcacheolmoeattention_cls'] = self.qkvcacheolmoeattention_cls_default
            self.logger.info("Using default QKVCacheOlmoeAttention class for OlmoeAttention layers")
        else:
            qclass_map['qkvcacheolmoeattention_cls'] = qkvcacheolmoeattention_cls
            self.logger.info(f"Using custom {qkvcacheolmoeattention_cls.__name__} class for OlmoeAttention layers")
        
        # Check QKVCacheOlmoeFlashAttention2
        if qkvcacheolmoeflashattention2_cls is None:
            qclass_map['qkvcacheolmoeflashattention2_cls'] = self.qkvcacheolmoeflashattention2_cls_default
            self.logger.info("Using default QKVCacheOlmoeFlashAttention2 class for OlmoeFlashAttention2 layers")
        else:
            qclass_map['qkvcacheolmoeflashattention2_cls'] = qkvcacheolmoeflashattention2_cls
            self.logger.info(f"Using custom {qkvcacheolmoeflashattention2_cls.__name__} class for OlmoeFlashAttention2 layers")
        
        # Check QKVCacheOlmoeSdpaAttention
        if qkvcacheolmoesdpaattention_cls is None:
            qclass_map['qkvcacheolmoesdpaattention_cls'] = self.qkvcacheolmoesdpaattention_cls_default
            self.logger.info("Using default QKVCacheOlmoeSdpaAttention class for OlmoeSdpaAttention layers")
        else:
            qclass_map['qkvcacheolmoesdpaattention_cls'] = qkvcacheolmoesdpaattention_cls
            self.logger.info(f"Using custom {qkvcacheolmoesdpaattention_cls.__name__} class for OlmoeSdpaAttention layers")
        
        # Check QKVCacheQwen2_5OmniAttention
        if qkvcacheqwen2_5omniattention_cls is None:
            qclass_map['qkvcacheqwen2_5omniattention_cls'] = self.qkvcacheqwen2_5omniattention_cls_default
            self.logger.info("Using default QKVCacheQwen2_5OmniAttention class for Qwen2_5OmniAttention layers")
        else:
            qclass_map['qkvcacheqwen2_5omniattention_cls'] = qkvcacheqwen2_5omniattention_cls
            self.logger.info(f"Using custom {qkvcacheqwen2_5omniattention_cls.__name__} class for Qwen2_5OmniAttention layers")
        
        # Check QKVCacheQwen3Attention
        if qkvcacheqwen3attention_cls is None:
            qclass_map['qkvcacheqwen3attention_cls'] = self.qkvcacheqwen3attention_cls_default
            self.logger.info("Using default QKVCacheQwen3Attention class for Qwen3Attention layers")
        else:
            qclass_map['qkvcacheqwen3attention_cls'] = qkvcacheqwen3attention_cls
            self.logger.info(f"Using custom {qkvcacheqwen3attention_cls.__name__} class for Qwen3Attention layers")
        
        # Check QKVCacheQwen3MoeAttention
        if qkvcacheqwen3moeattention_cls is None:
            qclass_map['qkvcacheqwen3moeattention_cls'] = self.qkvcacheqwen3moeattention_cls_default
            self.logger.info("Using default QKVCacheQwen3MoeAttention class for Qwen3MoeAttention layers")
        else:
            qclass_map['qkvcacheqwen3moeattention_cls'] = qkvcacheqwen3moeattention_cls
            self.logger.info(f"Using custom {qkvcacheqwen3moeattention_cls.__name__} class for Qwen3MoeAttention layers")
        
        # Check QKVCacheLlamaAttention
        if qkvcachellamaattention_cls is None:
            qclass_map['qkvcachellamaattention_cls'] = self.qkvcachellamaattention_cls_default
            self.logger.info("Using default QKVCacheLlamaAttention class for LlamaAttention layers")
        else:
            qclass_map['qkvcachellamaattention_cls'] = qkvcachellamaattention_cls
            self.logger.info(f"Using custom {qkvcachellamaattention_cls.__name__} class for LlamaAttention layers")
        return qclass_map

    def quantize(
        self,
        model: nn.Module,
        qlinear_cls: nn.Module = None,
        qembedding_cls: nn.Module = None,
        qconv1d_cls: nn.Module = None,
        qconv2d_cls: nn.Module = None,
        qconv3d_cls: nn.Module = None,
        qmultiheadattention_cls: nn.Module = None,
        qkvcacheolmoeattention_cls: nn.Module = None,
        qkvcacheolmoeflashattention2_cls: nn.Module = None,
        qkvcacheolmoesdpaattention_cls: nn.Module = None,
        qkvcacheqwen2_5omniattention_cls: nn.Module = None,
        qkvcacheqwen3attention_cls: nn.Module = None,
        qkvcacheqwen3moeattention_cls: nn.Module = None,
        qkvcachellamaattention_cls: nn.Module = None,
    ) -> nn.Module:
        """
        Apply quantization to the model.
        
        Args:
            model: PyTorch model to quantize
            qlinear_cls: Custom quantized Linear class (default: QLinear)
            qembedding_cls: Custom quantized Embedding class (default: QEmbedding)
            qconv1d_cls: Custom quantized Conv1d class (default: QConv1d)
            qconv2d_cls: Custom quantized Conv2d class (default: QConv2d)
            qconv3d_cls: Custom quantized Conv3d class (default: QConv3d)
            qmultiheadattention_cls: Custom quantized MultiheadAttention class (default: QMultiheadAttention)
            qkvcacheolmoeattention_cls: Custom quantized QKVCacheOlmoeAttention class (default: QKVCacheOlmoeAttention)
            qkvcacheolmoeflashattention2_cls: Custom quantized QKVCacheOlmoeFlashAttention2 class (default: QKVCacheOlmoeFlashAttention2)
            qkvcacheolmoesdpaattention_cls: Custom quantized QKVCacheOlmoeSdpaAttention class (default: QKVCacheOlmoeSdpaAttention)
            qkvcacheqwen2_5omniattention_cls: Custom quantized QKVCacheQwen2_5OmniAttention class (default: QKVCacheQwen2_5OmniAttention)
            qkvcacheqwen3attention_cls: Custom quantized QKVCacheQwen3Attention class (default: QKVCacheQwen3Attention)
            qkvcacheqwen3moeattention_cls: Custom quantized QKVCacheQwen3MoeAttention class (default: QKVCacheQwen3MoeAttention)
            qkvcachellamaattention_cls: Custom quantized QKVCacheLlamaAttention class (default: QKVCacheLlamaAttention)
            
        Returns:
            Quantized model
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting model quantization")
        self.logger.info("=" * 80)

        # Log model information
        self.logger.info(f"Model type: {type(model).__name__}")

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")

        # Analyze model structure and generate quantization plan
        self.logger.info("Analyzing model structure...")
        quant_map = self.selector.analyze_model(model)

        # Log analysis results
        total_modules = len(quant_map)
        modules_to_quantize = len(self.selector.get_modules_to_quantize())
        self.logger.info(f"Total modules analyzed: {total_modules}")
        self.logger.info(f"Modules to quantize: {modules_to_quantize}")
        self.logger.info(f"Modules to skip: {total_modules - modules_to_quantize}")

        # Print detailed quantization plan
        self.logger.info("")
        self.logger.info("Detailed quantization plan:")
        self.logger.info("-" * 80)
        for name, info in sorted(quant_map.items()):
            status = "✓ QUANT" if info.should_quant else "✗ SKIP"
            module_type = info.module_type.__name__
            self.logger.info(f"  {status:<10} {name:<30} [{module_type}]")
        self.logger.info("-" * 80)

        if modules_to_quantize == 0:
            self.logger.warning("No modules selected for quantization!")
            self.logger.warning("Please check your selection configuration")
            return model

        # Check and use default quantized classes if not provided
        qclass_map = self._check_provided_qclass(
            qlinear_cls=qlinear_cls,
            qembedding_cls=qembedding_cls,
            qconv1d_cls=qconv1d_cls,
            qconv2d_cls=qconv2d_cls,
            qconv3d_cls=qconv3d_cls,
            qmultiheadattention_cls=qmultiheadattention_cls,
            qkvcacheolmoeattention_cls=qkvcacheolmoeattention_cls,
            qkvcacheolmoeflashattention2_cls=qkvcacheolmoeflashattention2_cls,
            qkvcacheolmoesdpaattention_cls=qkvcacheolmoesdpaattention_cls,
            qkvcacheqwen2_5omniattention_cls=qkvcacheqwen2_5omniattention_cls,
            qkvcacheqwen3attention_cls=qkvcacheqwen3attention_cls,
            qkvcacheqwen3moeattention_cls=qkvcacheqwen3moeattention_cls,
            qkvcachellamaattention_cls=qkvcachellamaattention_cls,
        )
        
        # Extract classes from map
        qlinear_cls = qclass_map['qlinear_cls']
        qembedding_cls = qclass_map['qembedding_cls']
        qconv1d_cls = qclass_map['qconv1d_cls']
        qconv2d_cls = qclass_map['qconv2d_cls']
        qconv3d_cls = qclass_map['qconv3d_cls']
        qmultiheadattention_cls = qclass_map['qmultiheadattention_cls']
        qkvcacheolmoeattention_cls = qclass_map['qkvcacheolmoeattention_cls']
        qkvcacheolmoeflashattention2_cls = qclass_map['qkvcacheolmoeflashattention2_cls']
        qkvcacheolmoesdpaattention_cls = qclass_map['qkvcacheolmoesdpaattention_cls']
        qkvcacheqwen2_5omniattention_cls = qclass_map['qkvcacheqwen2_5omniattention_cls']
        qkvcacheqwen3attention_cls = qclass_map['qkvcacheqwen3attention_cls']
        qkvcacheqwen3moeattention_cls = qclass_map['qkvcacheqwen3moeattention_cls']
        qkvcachellamaattention_cls = qclass_map['qkvcachellamaattention_cls']

        # Apply quantization
        self.logger.info("")
        self.logger.info("Applying quantization to selected modules...")

        try:
            quantized_model = apply_quantization(
                model=model,
                quant_config=self.config,
                selector=self.selector,
                qlinear_cls=qlinear_cls,
                qembedding_cls=qembedding_cls,
                qconv1d_cls=qconv1d_cls,
                qconv2d_cls=qconv2d_cls,
                qconv3d_cls=qconv3d_cls,
                qmultiheadattention_cls=qmultiheadattention_cls,
                qkvcacheolmoeattention_cls=qkvcacheolmoeattention_cls,
                qkvcacheolmoeflashattention2_cls=qkvcacheolmoeflashattention2_cls,
                qkvcacheolmoesdpaattention_cls=qkvcacheolmoesdpaattention_cls,
                qkvcacheqwen2_5omniattention_cls=qkvcacheqwen2_5omniattention_cls,
                qkvcacheqwen3attention_cls=qkvcacheqwen3attention_cls,
                qkvcacheqwen3moeattention_cls=qkvcacheqwen3moeattention_cls,
                qkvcachellamaattention_cls=qkvcachellamaattention_cls,
            )

            self.logger.info("Quantization applied successfully!")

        except Exception as e:
            self.logger.error(f"Failed to apply quantization: {e}")
            raise

        # Log quantization results
        self.logger.info("")
        self.logger.info("Quantization results:")
        self.logger.info("-" * 80)

        # Count quantized modules
        qlinear_count = 0
        qembedding_count = 0
        qconv1d_count = 0
        qconv2d_count = 0
        qconv3d_count = 0
        qmultiheadattention_count = 0
        qkvcacheolmoeattention_count = 0
        qkvcacheolmoeflashattention2_count = 0
        qkvcacheolmoesdpaattention_count = 0
        qkvcacheqwen2_5omniattention_count = 0
        qkvcacheqwen3attention_count = 0
        qkvcacheqwen3moeattention_count = 0
        qkvcachellamaattention_count = 0
        for _, module in quantized_model.named_modules():
            if isinstance(module, qlinear_cls):
                qlinear_count += 1
            elif isinstance(module, qembedding_cls):
                qembedding_count += 1
            elif isinstance(module, qconv1d_cls):
                qconv1d_count += 1
            elif isinstance(module, qconv2d_cls):
                qconv2d_count += 1
            elif isinstance(module, qconv3d_cls):
                qconv3d_count += 1
            elif isinstance(module, qmultiheadattention_cls):
                qmultiheadattention_count += 1
            # Check subclasses first before checking parent class OlmoeAttention
            elif isinstance(module, qkvcacheolmoeflashattention2_cls):
                qkvcacheolmoeflashattention2_count += 1
            elif isinstance(module, qkvcacheolmoesdpaattention_cls):
                qkvcacheolmoesdpaattention_count += 1
            elif isinstance(module, qkvcacheolmoeattention_cls):
                qkvcacheolmoeattention_count += 1
            elif isinstance(module, qkvcacheqwen2_5omniattention_cls):
                qkvcacheqwen2_5omniattention_count += 1
            elif isinstance(module, qkvcacheqwen3attention_cls):
                qkvcacheqwen3attention_count += 1
            elif isinstance(module, qkvcacheqwen3moeattention_cls):
                qkvcacheqwen3moeattention_count += 1
            elif isinstance(module, qkvcachellamaattention_cls):
                qkvcachellamaattention_count += 1

        self.logger.info(f"  Quantized Linear modules: {qlinear_count}")
        self.logger.info(f"  Quantized Embedding modules: {qembedding_count}")
        self.logger.info(f"  Quantized Conv1d modules: {qconv1d_count}")
        self.logger.info(f"  Quantized Conv2d modules: {qconv2d_count}")
        self.logger.info(f"  Quantized Conv3d modules: {qconv3d_count}")
        self.logger.info(f"  Quantized MultiheadAttention modules: {qmultiheadattention_count}")
        self.logger.info(f"  Quantized OlmoeAttention modules: {qkvcacheolmoeattention_count}")
        self.logger.info(f"  Quantized OlmoeFlashAttention2 modules: {qkvcacheolmoeflashattention2_count}")
        self.logger.info(f"  Quantized OlmoeSdpaAttention modules: {qkvcacheolmoesdpaattention_count}")
        self.logger.info(f"  Quantized Qwen2_5OmniAttention modules: {qkvcacheqwen2_5omniattention_count}")
        self.logger.info(f"  Quantized Qwen3Attention modules: {qkvcacheqwen3attention_count}")
        self.logger.info(f"  Quantized Qwen3MoeAttention modules: {qkvcacheqwen3moeattention_count}")
        self.logger.info(f"  Quantized LlamaAttention modules: {qkvcachellamaattention_count}")
        self.logger.info(f"  Total quantized modules: {qlinear_count + qembedding_count + qconv1d_count + qconv2d_count + qconv3d_count + qmultiheadattention_count + qkvcacheolmoeattention_count + qkvcacheolmoeflashattention2_count + qkvcacheolmoesdpaattention_count + qkvcacheqwen2_5omniattention_count + qkvcacheqwen3attention_count + qkvcacheqwen3moeattention_count + qkvcachellamaattention_count}")
        
        # Fix `tie_word_embeddings=True` issue
        model_class_name = quantized_model.__class__.__name__
        if isinstance(quantized_model, (PreTrainedModel, )) and ('CausalLM' in model_class_name or 'GPT' in model_class_name):
            if quantized_model.config.tie_word_embeddings:
                # quantized_model.model.embed_tokens.weight = quantized_model.lm_head.weight # Manual handling
                quantized_model.tie_weights() # PreTrainedModel API handling
            self.logger.info(f"  Quantized model tie_word_embeddings: {quantized_model.config.tie_word_embeddings}")
        else:
            self.logger.info("  Quantized model is not a CausalLM/GPT model or does not use tie_word_embeddings")

        # Calculate quantized parameters using weight object ID deduplication
        # This prevents double-counting shared weights (e.g., tied embeddings)
        # qkvcache_xxxxxx_cls is state quantization only, so we do not count its parameters (weights) here
        counted_weight_ids = set()
        quantized_params = 0
        for _, module in quantized_model.named_modules():
            if isinstance(module, (qlinear_cls, qembedding_cls, qconv1d_cls, qconv2d_cls, qconv3d_cls, qmultiheadattention_cls)):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_id = id(module.weight)
                    if weight_id not in counted_weight_ids:
                        quantized_params += module.weight.numel()
                        counted_weight_ids.add(weight_id)
        
        self.logger.info(f"  Quantized parameters: {quantized_params:,} ({quantized_params/total_params*100:.10f}%)")
        self.logger.info(f"  Unique weight objects: {len(counted_weight_ids)}")

        self.logger.info("-" * 80)
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Model quantization completed successfully!")
        self.logger.info("=" * 80)

        return quantized_model

    def replace_quantized_weights(
        self,
        model: nn.Module,
        qlinear_cls: nn.Module = None,
        qembedding_cls: nn.Module = None,
        qconv1d_cls: nn.Module = None,
        qconv2d_cls: nn.Module = None,
        qconv3d_cls: nn.Module = None,
        qmultiheadattention_cls: nn.Module = None,
    ) -> nn.Module:
        """
        Replace model weights with their quantized versions.
        
        Args:
            model: PyTorch model with quantized modules
            
        Returns:
            Model with quantized weights
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting replacement of quantized weights")
        self.logger.info("=" * 80)
        
        # Check and use default quantized classes if not provided
        qclass_map = self._check_provided_qclass(
            qlinear_cls=qlinear_cls,
            qembedding_cls=qembedding_cls,
            qconv1d_cls=qconv1d_cls,
            qconv2d_cls=qconv2d_cls,
            qconv3d_cls=qconv3d_cls,
            qmultiheadattention_cls=qmultiheadattention_cls,
        )
        
        # Extract classes from map
        qlinear_cls = qclass_map['qlinear_cls']
        qembedding_cls = qclass_map['qembedding_cls']
        qconv1d_cls = qclass_map['qconv1d_cls']
        qconv2d_cls = qclass_map['qconv2d_cls']
        qconv3d_cls = qclass_map['qconv3d_cls']
        qmultiheadattention_cls = qclass_map['qmultiheadattention_cls']

        try:
            updated_model = replace_applied_quantized_weights(
                model=model,
                selector=self.selector,
                qlinear_cls=qlinear_cls,
                qembedding_cls=qembedding_cls,
                qconv1d_cls=qconv1d_cls,
                qconv2d_cls=qconv2d_cls,
                qconv3d_cls=qconv3d_cls,
                qmultiheadattention_cls=qmultiheadattention_cls,
                replace_weights=True,
            )

            self.logger.info("Quantized weights replaced successfully!")

        except Exception as e:
            self.logger.error(f"Failed to replace quantized weights: {e}")
            raise

        self.logger.info("=" * 80)
        self.logger.info("Replacement of quantized weights completed successfully!")
        self.logger.info("=" * 80)

        return updated_model
