"""
Gradient synchronization utilities.

This package provides the infrastructure for manual gradient bucketing and
asynchronous reduction, similar to DistributedDataParallel but exposed
for internal framework usage with DTensors.
"""


from .synchronizer import GradientSynchronizer

__all__ = [
    "GradientSynchronizer"
]
