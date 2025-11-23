# Quantize ResNet-18 on MNIST

This example demonstrates how to train ResNet-18 on MNIST dataset with EdgeRazor's Quantization Aware Training (QAT).

## Project Structure

```
example/qat/resnet/
├── src/
│   ├── __init__.py            # Package initialization
│   ├── __main__.py            # Module entry point
│   ├── arg.py                 # Argument parser with all parameters
│   ├── train.py               # Main training script with rich logging
│   └── ResNet-18.json         # ResNet-18 model configuration
├── q_resnet_w1.58_a16.yaml    # QAT config: W1.58-A16
├── q_resnet_w1.58_a8.yaml     # QAT config: W1.58-A8
├── q_resnet_w1.58_a4.yaml     # QAT config: W1.58-A4
├── q_resnet_w4_a16.yaml       # QAT config: W4-A16
├── q_resnet_w4_a8.yaml        # QAT config: W4-A8
├── q_resnet_w4_a4.yaml        # QAT config: W4-A4
├── run.sh                     # Training commands
└── README.md                  # Comprehensive documentation
```

## Model Configuration

**ResNet-18**:
- Architecture: ResNet-18 from torchvision
- Input channels: 3 (RGB)
- Image size: 224x224
- Number of classes: 10 (MNIST digits)
- Parameters: 11,181,642 (~11.18M)
  - Conv2d: 11,166,912 (99.87%)
  - BatchNorm2d: 9,600 (0.09%)
  - Linear: 5,130 (0.05%)

## Quantization Configurations

### Weight Quantization
- **W1.58 (Ternary)**: {-1, 0, 1} * scaling_factor
  - Function: `weight_quant_uniform_symmetric_clip_per_block_int1_58`
  - Granularity: Per-block
  
- **W4 (4-bit)**: {-7, ..., 0, ..., 7} * scaling_factor
  - Function: `weight_quant_uniform_symmetric_absmax_per_block_int4`
  - Granularity: Per-block

### State Quantization (Activation)
- **A16**: No quantization (full precision)
- **A8**: INT8 quantization using `state_quant_uniform_symmetric_absmax_per_token_int8`
- **A4**: INT4 quantization using `state_quant_uniform_symmetric_absmax_per_token_int4`

## Target Layers

ResNet-18 quantization targets:
- **Conv2d layers**: All convolutional layers in residual blocks
- **Linear layers**: Final fully-connected classification layer

## Usage

### Install Dependencies

```bash
# Install EdgeRazor package from the root directory
pip install -e ../../../

# Install project-specific dependencies
pip install -r requirements.txt
```

### Run Training

Execute all training configurations:

```bash
mkdir -p runs && bash run.sh | tee ./runs/$(date +%Y_%m_%d_%H_%M).log
```

This trains 7 models: Baseline (W16-A16), W1.58-A16, W1.58-A8, W1.58-A4, W4-A16, W4-A8, W4-A4

## Hyperparameters

| Parameter               | Value    | Description                             |
| :---------------------- | :------- | :-------------------------------------- |
| Batch size              | 256      | Training batch size                     |
| Epochs                  | 10       | Maximum training epochs                 |
| Learning rate           | 1e-4     | Initial learning rate                   |
| Weight decay            | 0.2      | L2 regularization coefficient           |
| Warmup steps            | 150      | Linear warmup steps (~10% of total)     |
| Min LR                  | 1e-6     | Minimum learning rate after annealing   |
| Early stopping patience | 3        | Stop after 3 epochs without improvement |
| Data type               | bfloat16 | Weight and computation dtype            |

## Experimental Results

Under same hyperparameters:

| Model     | Training | Params | W    | A    | Prop   | Comp   | Accuracy | Recall | Precision | F1     |
| :-------- | :------- | :----- | :--- | :--- | :----- | :----- | :------- | :----- | :-------- | :----- |
| ResNet-18 | 3.41 min | ~11M   | 16   | 16   | 0%     | 1.00x  | 98.74%   | 98.72% | 98.74%    | 98.73% |
| ResNet-18 | 4.21 min | ~11M   | 4    | 16   | 99.91% | 3.99x  | 98.63%   | 98.62% | 98.63%    | 98.62% |
| ResNet-18 | 4.21 min | ~11M   | 4    | 8    | 99.91% | 3.99x  | 98.67%   | 98.66% | 98.66%    | 98.66% |
| ResNet-18 | 4.17 min | ~11M   | 4    | 4    | 99.91% | 3.99x  | 98.46%   | 98.45% | 98.45%    | 98.45% |
| ResNet-18 | 3.42 min | ~11M   | 1.58 | 16   | 99.91% | 10.04x | 95.73%   | 95.70% | 95.72%    | 95.70% |
| ResNet-18 | 4.26 min | ~11M   | 1.58 | 8    | 99.91% | 10.04x | 95.70%   | 95.65% | 95.69%    | 95.66% |
| ResNet-18 | 4.21 min | ~11M   | 1.58 | 4    | 99.91% | 10.04x | 95.55%   | 95.52% | 95.53%    | 95.52% |

**Notes:**
- **Params**: Approximate parameter count
- **Prop**: Proportion of quantized parameters
- **Comp**: Compression ratio = 16 / ((1-Prop) * 16 + Prop * W_q)
- **W**: Weight bit-width (1.58 = ternary)
- **A**: Activation (state) bit-width
