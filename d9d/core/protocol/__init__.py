"""Package providing protocol definitions for standard PyTorch objects."""

from .data import DataLoaderProtocol, MicrobatchPackStream
from .training import LRSchedulerProtocol, OptimizerProtocol

__all__ = [
    "DataLoaderProtocol",
    "LRSchedulerProtocol",
    "MicrobatchPackStream",
    "OptimizerProtocol",
]
