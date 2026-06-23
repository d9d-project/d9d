"""Package providing protocol definitions for standard PyTorch objects."""

from .data import BatchIterator
from .training import LRSchedulerProtocol, OptimizerProtocol

__all__ = [
    "BatchIterator",
    "LRSchedulerProtocol",
    "OptimizerProtocol",
]
