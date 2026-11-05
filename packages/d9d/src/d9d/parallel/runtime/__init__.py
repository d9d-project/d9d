"""Explicit data parallelism that the pipeline program drives: weight gathers, gradient reductions and memory."""

from .coordinator import ParameterRuntime
from .types import BucketId, ReductionPlan

__all__ = ["BucketId", "ParameterRuntime", "ReductionPlan"]
