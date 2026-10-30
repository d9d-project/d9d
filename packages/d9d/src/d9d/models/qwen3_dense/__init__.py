"""Qwen3 Dense model, its Hugging Face state mappers and its parallelization."""

from .decoder_layer import Qwen3DenseLayer
from .huggingface import (
    mapper_from_huggingface_qwen3_dense,
    mapper_from_huggingface_qwen3_dense_for_causal_lm,
    mapper_from_huggingface_qwen3_dense_for_classification,
    mapper_from_huggingface_qwen3_dense_for_embedding,
    mapper_to_huggingface_qwen3_dense,
    mapper_to_huggingface_qwen3_dense_for_causal_lm,
    mapper_to_huggingface_qwen3_dense_for_classification,
    mapper_to_huggingface_qwen3_dense_for_embedding,
)
from .model import Qwen3DenseModel
from .parallelize import parallelize_qwen3_dense_model
from .params import (
    Qwen3DenseLayerParameters,
    Qwen3DenseParameters,
)

__all__ = [
    "Qwen3DenseLayer",
    "Qwen3DenseLayerParameters",
    "Qwen3DenseModel",
    "Qwen3DenseParameters",
    "mapper_from_huggingface_qwen3_dense",
    "mapper_from_huggingface_qwen3_dense_for_causal_lm",
    "mapper_from_huggingface_qwen3_dense_for_classification",
    "mapper_from_huggingface_qwen3_dense_for_embedding",
    "mapper_to_huggingface_qwen3_dense",
    "mapper_to_huggingface_qwen3_dense_for_causal_lm",
    "mapper_to_huggingface_qwen3_dense_for_classification",
    "mapper_to_huggingface_qwen3_dense_for_embedding",
    "parallelize_qwen3_dense_model",
]
