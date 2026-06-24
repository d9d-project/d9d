"""Package providing protocol definitions for standard PyTorch objects."""

from .data import DataLoaderProtocol, MicrobatchPackIterator
from .training import LRSchedulerProtocol, OptimizerProtocol

__all__ = [
    "DataLoaderProtocol",
    "LRSchedulerProtocol",
    "MicrobatchPackIterator",
    "OptimizerProtocol",
]
