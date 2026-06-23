"""Common type definitions used throughout the framework."""

from .data import CollateFn, Microbatches
from .pytree import PyTree, ScalarTree, TensorTree

__all__ = [
    "CollateFn",
    "Microbatches",
    "PyTree",
    "ScalarTree",
    "TensorTree",
]
