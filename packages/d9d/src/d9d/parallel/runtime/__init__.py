"""Explicit data parallelism that the pipeline program drives: weight gathers, gradient reductions and buffers."""

from .coordinator import ParameterRuntime
from .stage import BackwardPass, StageParameters

__all__ = ["BackwardPass", "ParameterRuntime", "StageParameters"]
