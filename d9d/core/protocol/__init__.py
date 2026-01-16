"""Package providing protocol definitions for standard PyTorch objects."""

from .training import LRSchedulerProtocol, OptimizerProtocol

__all__ = [
    "LRSchedulerProtocol",
    "OptimizerProtocol"
]
