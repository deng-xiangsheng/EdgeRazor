"""
Calculate quantization proportion for ResNet-18 with QAT.

Usage: python quant_prop.py [--quant_config CONFIG.yaml]
"""

import argparse
import sys
from pathlib import Path

import torch.nn as nn
from torchvision.models import resnet18

# Add EdgeRazor to path
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from edgerazor.qat import QAT


class ResNet18ForMNIST(nn.Module):
    """ResNet-18 model for MNIST."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)


def main():
    parser = argparse.ArgumentParser(description="Calculate quantization proportion for ResNet-18")
    parser.add_argument("--quant_config", type=str, default="q_resnet_w1.58_a16.yaml",
                        help="Path to quantization YAML file")
    args = parser.parse_args()
    
    # Create model
    model = ResNet18ForMNIST(num_classes=10)
    
    print("=" * 80)
    print("ResNet-18 Quantization Proportion Analysis")
    print("=" * 80)
    print(f"Model: ResNet-18 (MNIST, 10 classes)")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Apply QAT quantization
    quant_config_path = Path(args.quant_config)
    if not quant_config_path.exists():
        print(f"Error: {quant_config_path} not found")
        sys.exit(1)
    
    print(f"Applying QAT with config: {quant_config_path}")
    qat = QAT(config_path=str(quant_config_path))
    model = qat.quantize(model)
    print(model)


if __name__ == "__main__":
    main()
