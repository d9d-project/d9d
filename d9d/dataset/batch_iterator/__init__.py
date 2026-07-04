"""Composable building blocks for the batch-iterator stack."""

from .maths import num_microbatches_for_global_batch
from .packer import FixedCountMicrobatchPacker

__all__ = [
    "FixedCountMicrobatchPacker",
    "num_microbatches_for_global_batch",
]
