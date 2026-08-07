"""Task heads that turn backbone hidden states into a typed output, plus their IO contracts."""

from .base import TaskHead
from .classification import ClassificationHead
from .embedding import EmbeddingHead
from .io import (
    SequenceCausalLMHeadShared,
    SequenceCausalLMOutput,
    SequenceClassificationOutput,
    SequenceEmbeddingOutput,
    SequencePoolingHeadShared,
)
from .language_modelling import LM_IGNORE_INDEX, SplitLanguageModellingHead

__all__ = [
    "LM_IGNORE_INDEX",
    "ClassificationHead",
    "EmbeddingHead",
    "SequenceCausalLMHeadShared",
    "SequenceCausalLMOutput",
    "SequenceClassificationOutput",
    "SequenceEmbeddingOutput",
    "SequencePoolingHeadShared",
    "SplitLanguageModellingHead",
    "TaskHead",
]
