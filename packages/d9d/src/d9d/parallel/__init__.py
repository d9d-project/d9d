"""Horizontal parallelism: placing parameters as ``DTensor`` on a device mesh."""

from .fully_sharded import parallelize_fsdp
from .hybrid_sharded import parallelize_hsdp
from .placement import distribute_parameters
from .replicate_parallel import parallelize_replicate

__all__ = [
    "distribute_parameters",
    "parallelize_fsdp",
    "parallelize_hsdp",
    "parallelize_replicate",
]
