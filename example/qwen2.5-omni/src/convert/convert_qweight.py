"""
Convert unquantized model to quantized model with EdgeRazor.

This script loads an unquantized model, applies quantization using EdgeRazor,
replaces weights with their quantized versions, and saves only the weight files
(safetensors) to the output directory.

Usage:
    python convert_qweight.py \\
        --quant_config /path/to/edgerazor_qwen3_w2a8kv16.yaml \\
        --unquantized_model /path/to/original/model \\
        --quantized_model /path/to/save/quantized/model \\
        --dtype bfloat16
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers import Qwen2_5OmniForConditionalGeneration


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert unquantized model to quantized model with EdgeRazor"
    )
    parser.add_argument(
        "--quant_config",
        type=str,
        required=True,
        help="Path to the quantization config YAML file"
    )
    parser.add_argument(
        "--unquantized_model",
        type=str,
        required=True,
        help="Path to the original unquantized model"
    )
    parser.add_argument(
        "--quantized_model",
        type=str,
        required=True,
        help="Path to save the quantized model weights"
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
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="Trust remote code when loading model (default: False)"
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


def load_model(model_path: str, dtype: torch.dtype, device: str, trust_remote_code: bool):
    """Load model from path."""
    print(f"Loading model from: {model_path}")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")
    
    try:
        # Try loading with Qwen2_5OmniForConditionalGeneration first
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
        )
        print("✓ Model loaded successfully (Qwen2_5OmniForConditionalGeneration)")
        return model
    except Exception as e:
        print(f"[Qwen2_5OmniForConditionalGeneration] Failed: {e}", file=sys.stderr)
        # Fall back to AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=trust_remote_code,
                torch_dtype=dtype,
            )
            print("✓ Model loaded successfully (AutoModelForCausalLM)")
            return model
        except Exception as e2:
            print(f"✗ Failed to load model: {e2}", file=sys.stderr)
            raise


def quantize_model(model, quant_config_path: str):
    """Apply quantization to model using EdgeRazor."""
    try:
        from edgerazor import EdgeRazor
    except ImportError:
        print("✗ Failed to import EdgeRazor. Please ensure it's installed.", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nApplying quantization with config: {quant_config_path}")
    
    if not os.path.exists(quant_config_path):
        print(f"✗ Quantization config not found: {quant_config_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        erazor = EdgeRazor(config=quant_config_path)
        qmodel = erazor.quantize(model)
        print("✓ Quantization applied successfully")
        return qmodel, erazor
    except Exception as e:
        print(f"✗ Failed to apply quantization: {e}", file=sys.stderr)
        raise


def replace_weights(qmodel, erazor):
    """Replace model weights with quantized versions."""
    print("\nReplacing weights with quantized versions...")
    
    try:
        quantized_model = erazor.replace_quantized_weights(qmodel)
        print("✓ Weights replaced successfully")
        return quantized_model
    except Exception as e:
        print(f"✗ Failed to replace weights: {e}", file=sys.stderr)
        raise


def save_quantized_weights(quantized_model, output_dir: str):
    """Save only the quantized weight files (safetensors) to output directory."""
    print(f"\nSaving quantized weights to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use temporary directory to save full model, then copy only safetensors
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  Using temporary directory: {temp_dir}")
            
            # Save model to temporary directory
            quantized_model.save_pretrained(
                temp_dir,
                state_dict=quantized_model.state_dict(),
                safe_serialization=True
            )
            
            # Copy safetensors files to output directory
            safetensors_files = list(Path(temp_dir).glob("*.safetensors"))
            index_file = Path(temp_dir) / "model.safetensors.index.json"
            
            if not safetensors_files:
                print("✗ No safetensors files found in temporary directory", file=sys.stderr)
                return False
            
            for file in safetensors_files:
                dest_file = os.path.join(output_dir, file.name)
                shutil.copy2(file, dest_file)
                print(f"  ✓ Copied: {file.name}")
            
            # Also copy index.json if it exists (for sharded models)
            if index_file.exists():
                dest_index = os.path.join(output_dir, index_file.name)
                shutil.copy2(index_file, dest_index)
                print(f"  ✓ Copied: {index_file.name}")
            
            print(f"✓ Successfully saved {len(safetensors_files)} weight file(s)")
            return True
            
    except Exception as e:
        print(f"✗ Failed to save quantized weights: {e}", file=sys.stderr)
        raise


def main():
    """Main conversion workflow."""
    args = parse_args()
    
    print("=" * 80)
    print("EdgeRazor Model Weight Quantization Converter")
    print("=" * 80)
    
    # Convert dtype string to torch.dtype
    torch_dtype = get_torch_dtype(args.dtype)
    
    # Step 1: Load unquantized model
    model = load_model(
        args.unquantized_model,
        torch_dtype,
        args.device,
        args.trust_remote_code
    )
    
    # Step 2: Apply quantization
    qmodel, erazor = quantize_model(model, args.quant_config)
    
    # Step 3: Replace weights with quantized versions
    quantized_model = replace_weights(qmodel, erazor)
    
    # Step 4: Save quantized weights
    success = save_quantized_weights(quantized_model, args.quantized_model)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Conversion completed successfully!")
        print("=" * 80)
        print(f"\nQuantized weights saved to: {args.quantized_model}")
        print("\nNext steps:")
        print("  1. Copy other model files (config.json, tokenizer, etc.) to the output directory")
        print("  2. Update the quantization config to set is_w_quantized=True")
        print("  3. Load the quantized model with the updated config")
    else:
        print("\n" + "=" * 80)
        print("✗ Conversion failed")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
