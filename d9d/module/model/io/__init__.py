"""Typed pipeline IO dataclasses for the model catalogue."""

from .sequence import (
    SequenceCausalLMOutput,
    SequenceCausalLMShared,
    SequenceClassificationOutput,
    SequenceEmbeddingOutput,
    SequenceInput,
    SequencePoolingShared,
    SequenceShared,
    SequenceTransfer,
)

__all__ = [
    "SequenceCausalLMOutput",
    "SequenceCausalLMShared",
    "SequenceClassificationOutput",
    "SequenceEmbeddingOutput",
    "SequenceInput",
    "SequencePoolingShared",
    "SequenceShared",
    "SequenceTransfer",
]
