"""
Horizontal parallelism strategies and utilities for d9d modules.

This package provides high-level helper functions to apply specific distributed
parallelism strategies to PyTorch modules compatible with the d9d ecosystem.
"""

from .expert_parallel import parallelize_expert_parallel
from .fully_sharded import parallelize_fsdp
from .replicate_parallel import parallelize_replicate

__all__ = [
    "parallelize_expert_parallel",
    "parallelize_fsdp",
    "parallelize_replicate"
]
