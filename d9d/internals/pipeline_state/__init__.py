"""
Pipeline State management package.

This package provides mechanisms to store, retrieve, and synchronize state
across different stages of a distributed pipeline, providing global and sharded view for these states.
"""

from .api import PipelineState
from .handler import PipelineStateHandler

__all__ = [
    "PipelineState",
    "PipelineStateHandler"
]
