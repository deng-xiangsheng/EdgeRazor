"""Implementation of Quantization Modules
Supported Modules:
- Linear
- Embedding
- Conv1d
- Conv2d
- Conv3d
"""
# ruff: noqa: F401

from .qconv1d import QConv1d, copy_conv1d_to_qconv1d
from .qconv2d import QConv2d, copy_conv2d_to_qconv2d
from .qconv3d import QConv3d, copy_conv3d_to_qconv3d
from .qembedding import QEmbedding, copy_embedding_to_qembedding
from .qlinear import QLinear, copy_linear_to_qlinear
