config = {
    "qat_configuration": {
        "method": "QAT",
        "select": {
            "target_types": ["linear"],
            "target_names": [],
            "exclude_types": [],
            "exclude_names": []
        },
        "function": {
            "epsilon": 1e-05,
            "weight_function": "weight_quant_uniform_symmetric_absmax_per_block_int4",
            "w_block_size": 128,
            "is_w_quantized": False,
            "activation_function": "state_quant_uniform_symmetric_absmax_per_block_int8",
            "a_block_size": 128,
            "kv_cache_function": "",
            "kv_block_size": -1
        },
        "training": "all"
    },
    "kd_configuration": {
        "method": "KD",
        "loss_1": {
            "loss_type": "hidden_states",
            "loss_function": "compute_fd",
            "alpha": 0.5,
            "layer_index": ["low", "mid", "high"],
            "padding_id": -100,
            "reduction": "batch_mean"
        }
    }
}
