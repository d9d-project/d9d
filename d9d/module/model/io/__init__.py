"""Typed pipeline IO dataclasses for the model catalogue."""

from .multimodal import MediaSegments, MultimodalSequenceInput
from .sequence import (
    SequenceHeadShared,
    SequenceHeadsOutput,
    SequenceHeadsShared,
    SequenceInput,
    SequenceShared,
    SequenceTransfer,
)

__all__ = [
    "MediaSegments",
    "MultimodalSequenceInput",
    "SequenceHeadShared",
    "SequenceHeadsOutput",
    "SequenceHeadsShared",
    "SequenceInput",
    "SequenceShared",
    "SequenceTransfer",
]
