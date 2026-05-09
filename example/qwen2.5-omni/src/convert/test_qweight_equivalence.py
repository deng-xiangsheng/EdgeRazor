"""
Test equivalence between quantized model (training mode) and quantized weight model (inference mode).

This script verifies that a model with quantization applied during forward pass (is_w_quantized=False)
produces the same outputs as a model with pre-quantized weights loaded from disk (is_w_quantized=True).

Usage:
    python test_qweight_equivalence.py \\
        --quant_config ../src/edgerazor_qwen3_w2a8kv16.yaml \\
        --original_model /path/to/original/model \\
        --quantized_model /path/to/quantized/model \\
        --dtype bfloat16
"""

import argparse
import sys

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers import Qwen2_5OmniForConditionalGeneration


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test equivalence between quantized and quantized-weight models"
    )
    parser.add_argument(
        "--quant_config",
        type=str,
        required=True,
        help="Path to the quantization config YAML file"
    )
    parser.add_argument(
        "--original_model",
        type=str,
        required=True,
        help="Path to the original unquantized model"
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="Path to the model with quantized weights"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model loading (default: bfloat16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to load model on (default: cuda:0)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for comparison (default: 1e-2)"
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for comparison (default: 1e-2)"
    )
    
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_and_quantize_model(
    model_path: str,
    quant_config: str,
    dtype: torch.dtype,
    device: str,
    is_w_quantized: bool
):
    """
    Load model and apply quantization with EdgeRazor.
    
    Args:
        model_path: Path to the model
        quant_config: Path to quantization config
        dtype: Model dtype
        device: Device to load on
        is_w_quantized: Whether weights are already quantized
        
    Returns:
        Quantized model and EdgeRazor instance
    """
    try:
        from edgerazor import EdgeRazor
    except ImportError:
        print("✗ Failed to import EdgeRazor. Please ensure it's installed.", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading model from: {model_path}")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")
    print(f"  is_w_quantized: {is_w_quantized}")

    try:
        # Try loading with Qwen2_5OmniForConditionalGeneration first
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        print("✓ Model loaded successfully (Qwen2_5OmniForConditionalGeneration)")
    except Exception as e:
        print(f"[Qwen2_5OmniForConditionalGeneration] Failed: {e}", file=sys.stderr)
        # Fall back to AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            print("✓ Model loaded successfully (AutoModelForCausalLM)")
        except Exception as e2:
            print(f"✗ Failed to load and quantize model: {e2}", file=sys.stderr)
            raise

    # Apply quantization with EdgeRazor
    erazor = EdgeRazor(config=quant_config)

    # Override is_w_quantized setting
    erazor.qat.config.function.is_w_quantized = is_w_quantized
    print(f"  Set is_w_quantized = {is_w_quantized}")

    qmodel = erazor.quantize(model)
    print("✓ Quantization applied successfully")

    # Set model to eval mode for inference
    qmodel.eval()

    return qmodel, erazor


def get_first_and_last_linear(model: nn.Module) -> tuple[nn.Linear, nn.Linear]:
    """
    Extract the first and last Linear layers from the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (first_linear, last_linear)
    """
    linear_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))
    
    if len(linear_layers) < 2:
        raise ValueError(f"Model must have at least 2 Linear layers, found {len(linear_layers)}")
    
    first_name, first_linear = linear_layers[0]
    last_name, last_linear = linear_layers[-1]
    
    print(f"  First Linear: {first_name} - in_features={first_linear.in_features}, out_features={first_linear.out_features}")
    print(f"  Last Linear:  {last_name} - in_features={last_linear.in_features}, out_features={last_linear.out_features}")
    
    return first_linear, last_linear


def test_layer_equivalence(
    layer1: nn.Linear,
    layer2: nn.Linear,
    layer_name: str,
    input_tensor: torch.Tensor,
    rtol: float,
    atol: float
) -> bool:
    """
    Test if two layers produce equivalent outputs for the same input.
    
    Args:
        layer1: First layer (original quantized)
        layer2: Second layer (quantized weights)
        layer_name: Name of the layer for logging
        input_tensor: Input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if outputs are equivalent, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Testing {layer_name}")
    print(f"{'='*80}")
    
    with torch.no_grad():
        output1 = layer1(input_tensor)
        output2 = layer2(input_tensor)
    
    # Check if outputs are close
    outputs_close = torch.allclose(output1, output2, rtol=rtol, atol=atol)
    max_diff = torch.max(torch.abs(output1 - output2)).item()
    mean_diff = torch.mean(torch.abs(output1 - output2)).item()
    
    print(f"Output shape: {output1.shape}")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")
    print(f"Outputs are close (rtol={rtol}, atol={atol}): {outputs_close}")
    
    # Show first 5 weights
    print(f"\nLayer 1 weights (first 5): {layer1.weight.flatten()[:5]}")
    print(f"Layer 2 weights (first 5): {layer2.weight.flatten()[:5]}")
    
    # Show sample outputs
    print(f"\nModel 1 output (first 5): {output1.flatten()[:5]}")
    print(f"Model 2 output (first 5): {output2.flatten()[:5]}")
    
    if outputs_close:
        print(f"✓ {layer_name} outputs are equivalent")
    else:
        print(f"✗ {layer_name} outputs differ by max {max_diff:.6e}")
    
    return outputs_close


def main():
    """Main testing workflow."""
    args = parse_args()
    
    print("=" * 80)
    print("EdgeRazor Quantized Weight Equivalence Test")
    print("=" * 80)
    
    torch_dtype = get_torch_dtype(args.dtype)
    
    # Load Model 1: Original model with quantization (is_w_quantized=False)
    print("\n" + "="*80)
    print("Loading Model 1: Original + Quantization (is_w_quantized=False)")
    print("="*80)
    model1, erazor1 = load_and_quantize_model(
        args.original_model,
        args.quant_config,
        torch_dtype,
        args.device,
        is_w_quantized=False
    )
    
    # Load Model 2: Quantized weights model (is_w_quantized=True)
    print("\n" + "="*80)
    print("Loading Model 2: Quantized Weights (is_w_quantized=True)")
    print("="*80)
    model2, erazor2 = load_and_quantize_model(
        args.quantized_model,
        args.quant_config,
        torch_dtype,
        args.device,
        is_w_quantized=True
    )
    
    # Test text llm part
    model1, model2, = model1.thinker.model, model2.thinker.model
    model1, model2 = model1.eval(), model2.eval()
    
    # Extract first and last linear layers
    print("\n" + "="*80)
    print("Extracting Linear Layers")
    print("="*80)
    print("\nModel 1:")
    first_linear_1, last_linear_1 = get_first_and_last_linear(model1)
    print("\nModel 2:")
    first_linear_2, last_linear_2 = get_first_and_last_linear(model2)
    
    # Prepare test inputs
    print("\n" + "="*80)
    print("Preparing Test Inputs")
    print("="*80)
    
    # Input for first linear layer
    first_input_size = first_linear_1.in_features
    first_input = torch.randn(1, first_input_size, dtype=torch_dtype).to(model1.device)
    print(f"First layer input shape: {first_input.shape}")
    
    # Input for last linear layer
    last_input_size = last_linear_1.in_features
    last_input = torch.randn(1, last_input_size, dtype=torch_dtype).to(model1.device)
    print(f"Last layer input shape: {last_input.shape}")
    
    # Test first linear layer
    first_result = test_layer_equivalence(
        first_linear_1,
        first_linear_2,
        "First Linear Layer",
        first_input,
        args.rtol,
        args.atol
    )
    
    # Test last linear layer
    last_result = test_layer_equivalence(
        last_linear_1,
        last_linear_2,
        "Last Linear Layer",
        last_input,
        args.rtol,
        args.atol
    )
    
    # Final summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    if first_result and last_result:
        print("✓ All tests passed! Models are equivalent.")
        print("\nConclusion:")
        print("  The model with quantized weights (is_w_quantized=True) produces")
        print("  the same outputs as the model with on-the-fly quantization")
        print("  (is_w_quantized=False) for both first and last Linear layers.")
        return 0
    else:
        print("✗ Some tests failed! Models are NOT equivalent.")
        print("\nFailed tests:")
        if not first_result:
            print("  - First Linear Layer")
        if not last_result:
            print("  - Last Linear Layer")
        return 1


if __name__ == "__main__":
    sys.exit(main())
