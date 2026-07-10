"""Typed pipeline IO dataclasses for the model catalogue."""

from .sequence import (
    SequenceCausalLMHeadShared,
    SequenceCausalLMOutput,
    SequenceClassificationOutput,
    SequenceEmbeddingOutput,
    SequenceHeadsOutput,
    SequenceHeadsShared,
    SequenceInput,
    SequencePoolingHeadShared,
    SequenceShared,
    SequenceTransfer,
)

__all__ = [
    "SequenceCausalLMHeadShared",
    "SequenceCausalLMOutput",
    "SequenceClassificationOutput",
    "SequenceEmbeddingOutput",
    "SequenceHeadsOutput",
    "SequenceHeadsShared",
    "SequenceInput",
    "SequencePoolingHeadShared",
    "SequenceShared",
    "SequenceTransfer",
]
