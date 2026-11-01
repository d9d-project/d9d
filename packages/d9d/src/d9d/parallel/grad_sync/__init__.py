"""Gradient bucketing and asynchronous gradient reduction for ``DTensor`` parameters.

It works like ``DistributedDataParallel``, but reduces over the replicated dimensions of the placements.
"""

from .synchronizer import GradientSynchronizer

__all__ = [
    "GradientSynchronizer",
]
