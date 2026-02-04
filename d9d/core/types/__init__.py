"""
Common type definitions used throughout the framework.
"""
from .data import CollateFn
from .pytree import PyTree, ScalarTree, TensorTree

__all__ = [
    "CollateFn",
    "PyTree",
    "ScalarTree",
    "TensorTree"
]
