"""Package for gradient bucketing and asynchronous gradient reduction.

It works like ``DistributedDataParallel``, but on ``DTensor`` parameters and for internal use.
"""

from .synchronizer import GradientSynchronizer

__all__ = [
    "GradientSynchronizer",
]
