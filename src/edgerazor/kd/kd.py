"""
Knowledge Distillation (KD) module implementation for EdgeRazor.
- total_loss = loss_task_alpha * task_loss + distill_loss
- distill_loss = alpha_1 * loss_1 + alpha_2 * loss_2 + ... + alpha_n * loss_n

Model forward details:
model_inputs:
- input_ids: torch.Size([1, seq_len])
- attention_mask: torch.Size([1, seq_len])
- labels: torch.Size([1, seq_len])
```
output = model(
    **model_inputs,
    return_dict=True,
    output_hidden_states=True,
    output_attentions=True,
)
```

Based on model output:
- transformers.modeling_outputs.CausalLMOutputWithPast
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
- transformers.modeling_outputs.MoeCausalLMOutputWithPast
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None

Details:
- output['loss']: Task-specific loss (e.g., CrossEntropyLoss)
- output['logits'].shape = (batch_size, seq_len, vocab_size)
- output['past_key_values'][decoder_layer_index] = (key_states, value_states)
  - key_states.shape = (batch_size, num_key_value_heads, seq_length, head_dim=hidden_size/num_attention_heads)
  - value_states.shape = (batch_size, num_key_value_heads, seq_length, head_dim=hidden_size/num_attention_heads)
- output['hidden_states'][layer_index].shape = (batch_size, seq_len, hidden_size)
- output['attentions'][decoder_layer_index].shape = (batch_size, num_heads, query_seq_len, key_seq_len)

API format:
- compute_loss(student_outputs, teacher_outputs, labels): calculate total_loss, arrange all distill losses
  - student_outputs: model output (dict or ModelOutput) with 'loss' field containing task_loss
  - teacher_outputs: model output (dict, ModelOutput, or Tensor) for distillation
  - labels: ground truth labels
  - kd_config: loaded in __init__, contains all loss_i configurations
- compute_xxx(student_inputs, teacher_inputs, labels, kd_config_loss): calculate individual distill loss
  - student_inputs/teacher_inputs: logits, hidden_states, attentions, past_key_values, etc.
  - labels: ground truth labels
  - kd_config_loss: LossConfig object containing all parameters (alpha, temperature, padding_id, etc.)

Distill function format: `compute_xxx(...)`
- kldf: Kullback-Leibler Divergence Forward
- kldr: Kullback-Leibler Divergence Reverse
- kldc: Kullback-Leibler Divergence Confidence
- fd: Feature Distillation (MSE Loss)
"""

from pathlib import Path

import torch

from ..log import get_logger
from .util import DistillConfig, get_distill_function


class KD:
    """
    Knowledge Distillation (KD) class for EdgeRazor.
    
    Implements knowledge distillation with flexible multi-loss configuration.
    Formula: total_loss = loss_task_alpha * task_loss + distill_loss
             distill_loss = alpha_1 * loss_1 + alpha_2 * loss_2 + ... + alpha_n * loss_n
    
    Args:
        config: Configuration for KD, can be:
            - DistillConfig object
            - dict: Python dictionary
            - str/Path: Path to YAML/JSON configuration file
    
    Examples:
        >>> # From YAML file
        >>> kd = KD("configs/kd_kldc_fd.yaml")
        >>>
        >>> # From dict
        >>> kd = KD({
        ...     "method": "KD",
        ...     "loss_1": {
        ...         "loss_type": "logits",
        ...         "loss_function": "kldc",
        ...         "alpha": 0.7,
        ...         "temperature": 2.0
        ...     }
        ... })
        >>>
        >>> # Compute loss
        >>> total_loss, loss_dict = kd.compute_loss(
        ...     student_outputs=student_outputs,
        ...     teacher_outputs=teacher_outputs,
        ...     labels=labels
        ... )
    """
    
    def __init__(self, config):
        """
        Initialize KD with configuration.
        
        Args:
            config: Configuration (DistillConfig, dict, or file path)
        """
        self.logger = get_logger('KD')
        self.logger.info('Initializing Knowledge Distillation (KD)')
        
        # Load configuration
        self.config = self._load_configuration(config)
        self._log_configuration()
        
        # Initialize loss functions
        self.loss_functions = {}
        for loss_key, loss_config in self.config.losses.items():
            loss_fn = get_distill_function(loss_config.loss_function)
            self.loss_functions[loss_key] = loss_fn
            self.logger.info(
                f'Registered {loss_key}: '
                f'type={loss_config.loss_type}, '
                f'function={loss_config.loss_function}, '
                f'alpha={loss_config.alpha}'
            )
        
        self.logger.info('KD initialization completed')
    
    def _load_configuration(self, config):
        """
        Load configuration from various formats.
        
        Args:
            config: DistillConfig object, dict, or file path
        
        Returns:
            DistillConfig: Loaded configuration
        
        Raises:
            ValueError: If file format is unsupported
            TypeError: If config type is invalid
        """
        try:
            # DistillConfig object
            if isinstance(config, DistillConfig):
                self.logger.info('Using provided DistillConfig object')
                return config
            
            # Python dictionary
            elif isinstance(config, dict):
                self.logger.info('Loading configuration from dictionary')
                return DistillConfig.from_dict(config)
            
            # File path (YAML or JSON)
            elif isinstance(config, (str, Path)):
                config_path = Path(config)
                self.logger.info(f'Loading configuration from: {config_path}')
                
                suffix = config_path.suffix.lower()
                if suffix in ['.yaml', '.yml']:
                    return DistillConfig.from_yaml(config_path)
                elif suffix == '.json':
                    return DistillConfig.from_json(config_path)
                else:
                    raise ValueError(
                        f'Unsupported file format: {suffix}. '
                        f'Supported formats: .yaml, .yml, .json'
                    )
            
            # Invalid type
            else:
                raise TypeError(
                    f'Invalid config type: {type(config).__name__}. '
                    f'Expected: DistillConfig, dict, str, or Path'
                )
        
        except Exception as e:
            self.logger.error(f'Failed to load configuration: {e}')
            raise
    
    def _log_configuration(self):
        """Log configuration details."""
        self.logger.info('=' * 80)
        self.logger.info('KD Configuration')
        self.logger.info('=' * 80)
        self.logger.info(f'Method: {self.config.method}')
        self.logger.info(f'Task loss alpha: {self.config.loss_task_alpha}')
        self.logger.info(f'Number of losses: {len(self.config.losses)}')
        self.logger.info('')
        
        for loss_key, loss_config in self.config.losses.items():
            self.logger.info(f'{loss_key}:')
            self.logger.info(f'  loss_type:     {loss_config.loss_type}')
            self.logger.info(f'  loss_function: {loss_config.loss_function}')
            self.logger.info(f'  alpha:         {loss_config.alpha}')
            
            if loss_config.loss_type == 'logits':
                self.logger.info(f'  temperature:   {loss_config.temperature}')
                self.logger.info(f'  use_entropy:   {loss_config.use_entropy}')
            
            self.logger.info(f'  reduction:     {loss_config.reduction}')
            self.logger.info('')
        
        self.logger.info('=' * 80)
    
    def compute_loss(
        self,
        student_outputs,
        teacher_outputs,
        labels
    ):
        """
        Compute total loss with knowledge distillation.

        Formula:
            total_loss = loss_task_alpha * task_loss + distill_loss
            distill_loss = alpha_1 * loss_1 + alpha_2 * loss_2 + ... + alpha_n * loss_n

        Args:
            student_outputs: Student model outputs, can be:
                - dict: {'loss': ..., 'logits': ..., 'hidden_states': ..., 'attentions': ...}
                - ModelOutput: transformers output object with 'loss' attribute
            teacher_outputs: Teacher model outputs, can be:
                - torch.Tensor: logits only
                - dict: {'logits': ..., 'hidden_states': ..., 'attentions': ...}
                - ModelOutput: transformers output object
            labels: Ground truth labels (torch.Tensor)

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: torch.Tensor, sum of task_loss and distill_loss
                - loss_dict: dict with keys:
                    - 'task_loss': float, task-specific loss value
                    - 'distill_loss': float, total distillation loss (Σ alpha_i * loss_i)
                    - 'distill_loss_details': dict, individual loss values {'loss_1': float, 'loss_2': float, ...}
                    - 'total_loss': float, final total loss

        Examples:
            >>> # With ModelOutput
            >>> student_outputs = student_model(**inputs, labels=labels, return_dict=True)
            >>> teacher_outputs = teacher_model(**inputs, return_dict=True)
            >>> total_loss, loss_dict = kd.compute_loss(
            ...     student_outputs, teacher_outputs, labels
            ... )
            >>> # loss_dict = {
            >>> #     'task_loss': 2.5,
            >>> #     'distill_loss': 0.85,
            >>> #     'total_loss': 3.35,
            >>> #     'distill_loss_details': {'loss_1': 0.7, 'loss_2': 0.5},
            >>> # }
            >>>
            >>> # With dict outputs
            >>> student_outputs = {
            ...     'loss': task_loss,
            ...     'logits': student_logits,
            ...     'hidden_states': student_hidden_states
            ... }
            >>> teacher_outputs = {
            ...     'logits': teacher_logits,
            ...     'hidden_states': teacher_hidden_states
            ... }
            >>> total_loss, loss_dict = kd.compute_loss(
            ...     student_outputs, teacher_outputs, labels
            ... )
        """
        # Extract task loss from student outputs
        if isinstance(student_outputs, dict):
            task_loss = student_outputs.get('loss')
        else:
            # ModelOutput object
            task_loss = getattr(student_outputs, 'loss', None)

        if task_loss is None:
            raise ValueError(
                "task_loss not found in student_outputs. "
                "student_outputs must contain 'loss' (dict) or have 'loss' attribute (ModelOutput)."
            )

        # Convert outputs to dict format for unified handling
        if not isinstance(student_outputs, dict):
            student_outputs = {
                'loss': getattr(student_outputs, 'loss', None),
                'logits': getattr(student_outputs, 'logits', None),
                'hidden_states': getattr(student_outputs, 'hidden_states', None),
                'attentions': getattr(student_outputs, 'attentions', None),
            }

        if isinstance(teacher_outputs, torch.Tensor):
            teacher_outputs = {'logits': teacher_outputs}
        elif not isinstance(teacher_outputs, dict):
            teacher_outputs = {
                'logits': getattr(teacher_outputs, 'logits', None),
                'hidden_states': getattr(teacher_outputs, 'hidden_states', None),
                'attentions': getattr(teacher_outputs, 'attentions', None),
            }
        
        # Initialize loss dictionary
        loss_dict = {
            'total_loss': 0.0,
            'task_loss': task_loss.item() if isinstance(task_loss, torch.Tensor) else task_loss,
            'distill_loss': 0.0,
            'distill_loss_details': {},
        }
        
        # Compute distillation losses: distill_loss = Σ(alpha_i * loss_i)
        distill_loss = 0.0
        
        for loss_key, loss_config in self.config.losses.items():
            loss_fn = self.loss_functions[loss_key]
            
            # Logits distillation (KLD-based)
            if loss_config.loss_type == 'logits':
                student_logits = student_outputs.get('logits')
                teacher_logits = teacher_outputs.get('logits')
                
                if student_logits is None or teacher_logits is None:
                    self.logger.warning(
                        f'{loss_key}: logits not found in outputs, skipping'
                    )
                    continue
                
                loss_value = loss_fn(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    kd_config_loss=loss_config
                )
                
                weighted_loss = loss_config.alpha * loss_value
                distill_loss += weighted_loss
                loss_dict['distill_loss_details'][loss_key] = loss_value.item()
            
            # Feature/Hidden states distillation (MSE-based)
            elif loss_config.loss_type == 'hidden_states':
                student_features = student_outputs.get('hidden_states')
                teacher_features = teacher_outputs.get('hidden_states')
                
                if student_features is None or teacher_features is None:
                    self.logger.warning(
                        f'{loss_key}: features/hidden_states not found in outputs, skipping'
                    )
                    continue
                
                # Handle layer selection
                # hidden_states can be:
                # - Single tensor: (batch_size, seq_len, hidden_size)
                # - Tuple of tensors: (layer_0, layer_1, ..., layer_n)
                #   where each layer_i has shape (batch_size, seq_len, hidden_size)
                
                if loss_config.layer_index is not None:
                    # If hidden_states is tuple, select specific layers
                    if isinstance(student_features, tuple):
                        # Convert layer_index to list for unified handling
                        if isinstance(loss_config.layer_index, int):
                            layer_indices = [loss_config.layer_index]
                        elif isinstance(loss_config.layer_index, str):
                            layer_indices = [loss_config.layer_index]
                        else:
                            layer_indices = loss_config.layer_index
                        
                        # Get total number of layers
                        num_layers = len(student_features)
                        
                        # Resolve string layer names to actual indices
                        resolved_indices = []
                        for idx in layer_indices:
                            if isinstance(idx, str):
                                # Map predefined string choices to actual layer indices
                                if idx == "low":
                                    actual_idx = 1 if num_layers > 1 else 0
                                elif idx == "mid":
                                    actual_idx = num_layers // 2
                                elif idx == "high":
                                    actual_idx = num_layers - 1
                                else:
                                    self.logger.warning(
                                        f'{loss_key}: unknown layer_index string "{idx}", skipping'
                                    )
                                    continue
                                resolved_indices.append(actual_idx)
                                self.logger.debug(
                                    f'{loss_key}: resolved "{idx}" to layer {actual_idx}'
                                )
                            else:
                                # Handle negative indexing for integer indices
                                actual_idx = idx if idx >= 0 else num_layers + idx
                                resolved_indices.append(actual_idx)
                        
                        # Compute loss for each selected layer and accumulate
                        layer_loss = 0.0
                        num_valid_layers = 0
                        
                        for actual_idx in resolved_indices:
                            if actual_idx < 0 or actual_idx >= num_layers:
                                self.logger.warning(
                                    f'{loss_key}: layer_index {actual_idx} out of range '
                                    f'(total {num_layers} layers), skipping this layer'
                                )
                                continue
                            
                            if actual_idx >= len(teacher_features):
                                self.logger.warning(
                                    f'{loss_key}: teacher layer {actual_idx} out of range '
                                    f'(total {len(teacher_features)} layers), skipping this layer'
                                )
                                continue
                            
                            # Compute loss for this layer
                            loss_value = loss_fn(
                                student_features=student_features[actual_idx],
                                teacher_features=teacher_features[actual_idx],
                                labels=labels,
                                kd_config_loss=loss_config
                            )
                            
                            layer_loss += loss_value
                            num_valid_layers += 1
                        
                        if num_valid_layers == 0:
                            self.logger.warning(
                                f'{loss_key}: no valid layers found for distillation, skipping'
                            )
                            continue
                        
                        # Average loss across selected layers
                        loss_value = layer_loss / num_valid_layers
                    else:
                        # If hidden_states is a single tensor, ignore layer_index
                        self.logger.warning(
                            f'{loss_key}: layer_index specified but hidden_states is not a tuple, '
                            f'using the single tensor for distillation'
                        )
                        loss_value = loss_fn(
                            student_features=student_features,
                            teacher_features=teacher_features,
                            labels=labels,
                            kd_config_loss=loss_config
                        )
                else:
                    # No layer_index specified, use all features
                    loss_value = loss_fn(
                        student_features=student_features,
                        teacher_features=teacher_features,
                        labels=labels,
                        kd_config_loss=loss_config
                    )
                
                weighted_loss = loss_config.alpha * loss_value
                distill_loss += weighted_loss
                loss_dict['distill_loss_details'][loss_key] = loss_value.item()
            
            # Attention distillation (future support)
            elif loss_config.loss_type == 'attention':
                student_attentions = student_outputs.get('attentions')
                teacher_attentions = teacher_outputs.get('attentions')
                
                if student_attentions is None or teacher_attentions is None:
                    self.logger.warning(
                        f'{loss_key}: attentions not found in outputs, skipping'
                    )
                    continue
                
                # TODO: Implement attention distillation
                self.logger.warning(
                    f'{loss_key}: attention distillation not implemented yet'
                )
            
            # Unknown loss type
            else:
                self.logger.warning(
                    f'{loss_key}: unknown loss_type "{loss_config.loss_type}", skipping'
                )
        
        # Finalize loss dictionary
        loss_dict['distill_loss'] = (
            distill_loss.item() if isinstance(distill_loss, torch.Tensor)
            else distill_loss
        )
        
        # Compute total loss with task_loss alpha weighting
        # total_loss = loss_task_alpha * task_loss + distill_loss
        weighted_task_loss = self.config.loss_task_alpha * task_loss
        total_loss = weighted_task_loss + distill_loss
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def __repr__(self):
        """String representation of KD object."""
        num_losses = len(self.config.losses)
        loss_types = [cfg.loss_type for cfg in self.config.losses.values()]
        return (
            f'KD(method={self.config.method}, '
            f'num_losses={num_losses}, '
            f'loss_types={loss_types})'
        )
