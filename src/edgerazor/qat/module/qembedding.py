
import torch.nn as nn
from torch import Tensor

from ..util.quant_config import QuantConfig


class QEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
        # Additional QAT hyperparameters
        quant_config: QuantConfig = None,   # Quantization configuration
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse, _weight, _freeze, device, dtype)

        if quant_config is None:
            raise ValueError("quant_config must be provided for QEmbedding.")

        # Small value to prevent division by zero
        self.epsilon = quant_config.function.epsilon
        # Whether the weights are already quantized: {-2^(n-1), 0, 2^(n-1)} * w_scale
        self.is_w_quantized = quant_config.function.is_w_quantized

        # Quantization configuration
        ## Weight
        self.w_quant_function = quant_config.function.weight_function
        self.w_scale_factor = quant_config.function.w_scale_factor
        self.w_block_size = quant_config.function.w_block_size
        # Embedding's input is LongInt, so no need to quantize the activation.

    def _weight_quant(self, replace_self: bool = False) -> Tensor:
        # Quantize weight into quantized format
        W = self.weight.data.clone()
        if self.w_scale_factor > 0 and self.w_block_size > 0:
            w_quant = self.w_quant_function(w=W, epsilon=self.epsilon, w_scale_factor=self.w_scale_factor, block_size=self.w_block_size)
        elif self.w_scale_factor > 0:
            w_quant = self.w_quant_function(w=W, epsilon=self.epsilon, w_scale_factor=self.w_scale_factor)
        elif self.w_block_size > 0:
            w_quant = self.w_quant_function(w=W, epsilon=self.epsilon, block_size=self.w_block_size)
        else:
            w_quant = self.w_quant_function(w=W, epsilon=self.epsilon)

        if replace_self:
            if not self.is_w_quantized:
                # IF need to replace self.weight with quantized weights
                self.weight.data = w_quant.clone()
                self.is_w_quantized = True
            else:
                raise RuntimeError("Weights are already ternarized. Cannot replace self again.")
        return w_quant

    def forward(self, x: Tensor) -> Tensor:
        W = self.weight

        if self.training:
            # Straight-Through Estimator for training
            w_quant = self._weight_quant(replace_self=False)
            w_quant = W + (w_quant - W).detach()
        else: # is_inference_mode
            if self.is_w_quantized:
                w_quant = W
            else:
                w_quant = self._weight_quant(replace_self=False)

        # Use standard embedding during training to ensure correct gradient propagation
        output = nn.functional.embedding(
            x, w_quant, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

        return output


def copy_embedding_to_qembedding(
    embedding: nn.Embedding,
    qembedding_cls: nn.Module = QEmbedding,
    quant_config: QuantConfig = None
):
    """Copy Embedding to quantized Embedding (adjust according to your QEmbedding implementation)"""
    # Adjust according to your QEmbedding implementation
    qembedding = qembedding_cls(
        num_embeddings=embedding.num_embeddings,
        embedding_dim=embedding.embedding_dim,
        padding_idx=embedding.padding_idx,
        max_norm=embedding.max_norm,
        norm_type=embedding.norm_type,
        scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse,
        _weight=None,   # will copy weight later
        _freeze=False,  # will not freeze the weights
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
        quant_config=quant_config
    )
    # Copy weights
    qembedding.weight.data = embedding.weight.data.clone()
    return qembedding
