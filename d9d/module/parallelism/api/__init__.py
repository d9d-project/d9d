"""Functions that apply horizontal parallelism strategies to modules."""

from .expert_parallel import parallelize_expert_parallel
from .fully_sharded import parallelize_fsdp
from .hybrid_sharded import parallelize_hsdp
from .replicate_parallel import parallelize_replicate

__all__ = [
    "parallelize_expert_parallel",
    "parallelize_fsdp",
    "parallelize_hsdp",
    "parallelize_replicate",
]
