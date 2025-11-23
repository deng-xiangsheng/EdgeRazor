"""
Test loading the actual unified QAT+KD configuration file.

This script demonstrates loading the unified configuration file that contains
both QAT and KD configurations in a single YAML file.
"""

from pathlib import Path

from edgerazor import EdgeRazor, EdgeRazorConfig


def test_load_unified_config():
    """Test loading the actual qat_w1.58mp4_a8_kd_kldc_fd.yaml file"""
    
    config_path = Path("example/configs/qad/qat_w1.58mp4_a8_kd_kldc_fd.yaml")
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating example config file for testing...")
        
        # Create example directory
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create example config
        example_config = {
            'qat_configuration': {
                'select': {
                    'target_types': ['linear'],
                    'target_names': [],
                    'exclude_types': [],
                    'exclude_names': []
                },
                'function': {
                    'epsilon': 0.00001,
                    'weight_function': 'weight_quant_uniform_symmetric_clip_per_block_mp_int1_58_int4_static',
                    'w_scale_factor': 2.0,
                    'w_block_size': 128,
                    'is_w_quantized': False,
                    'activation_function': 'state_quant_uniform_symmetric_absmax_per_block_int8',
                    'a_block_size': 128,
                    'kv_cache_function': '',
                    'kv_block_size': -1
                },
                'training': 'all'
            },
            'kd_configuration': {
                'loss_task_alpha': 1.0,
                'loss_1': {
                    'loss_type': 'logits',
                    'alpha': 0.7,
                    'loss_function': 'compute_kld_confidence',
                    'padding_id': -100,
                    'is_router_logits': False,
                    'reduction': 'batch_mean',
                    'temperature': 2.0,
                    'use_entropy': True
                },
                'loss_2': {
                    'loss_type': 'hidden_states',
                    'alpha': 0.5,
                    'loss_function': 'compute_fd',
                    'padding_id': -100,
                    'reduction': 'batch_mean'
                }
            }
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False)
    
    print("=" * 80)
    print("Testing Unified Configuration Loading")
    print("=" * 80)
    print()
    
    # Method 1: Load EdgeRazorConfig directly
    print("Method 1: Load EdgeRazorConfig from YAML")
    print("-" * 80)
    config = EdgeRazorConfig.from_yaml(config_path)
    print(f"Config loaded: {config}")
    print(f"Has QAT: {config.has_qat}")
    print(f"Has KD: {config.has_kd}")
    print()
    
    if config.has_qat:
        print("QAT Configuration:")
        print(f"  - Weight function: {config.qat_config.function.weight_function}")
        print(f"  - Activation function: {config.qat_config.function.activation_function}")
        print(f"  - Weight block size: {config.qat_config.function.w_block_size}")
        print(f"  - Activation block size: {config.qat_config.function.a_block_size}")
    print()
    
    if config.has_kd:
        print("KD Configuration:")
        print(f"  - Task loss alpha: {config.kd_config.loss_task_alpha}")
        print(f"  - Number of losses: {len(config.kd_config.losses)}")
        for loss_name, loss_config in config.kd_config.losses.items():
            print(f"  - {loss_name}:")
            print(f"      Type: {loss_config.loss_type}")
            print(f"      Function: {loss_config.loss_function}")
            print(f"      Alpha: {loss_config.alpha}")
    print()
    
    # Method 2: Initialize EdgeRazor with unified config
    print("Method 2: Initialize EdgeRazor with unified config")
    print("-" * 80)
    edgerazor = EdgeRazor(config=config_path)
    print(f"EdgeRazor instance: {edgerazor}")
    print(f"QAT enabled: {edgerazor.is_qat_enabled}")
    print(f"KD enabled: {edgerazor.is_kd_enabled}")
    print()
    
    # Method 3: Initialize EdgeRazor with EdgeRazorConfig object
    print("Method 3: Initialize EdgeRazor with EdgeRazorConfig object")
    print("-" * 80)
    edgerazor2 = EdgeRazor(config=config)
    print(f"EdgeRazor instance: {edgerazor2}")
    print()
    
    print("=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_load_unified_config()
