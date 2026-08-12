"""Typed pipeline IO dataclasses for the model catalogue."""

from d9d.module.block.attention import SequencePacking

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
    "SequencePacking",
    "SequencePoolingShared",
    "SequenceShared",
    "SequenceTransfer",
]
