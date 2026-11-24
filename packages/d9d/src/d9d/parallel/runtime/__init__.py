"""Data parallelism with weight gathers and gradient reductions around the regions of a module."""

from .bound import BoundParameters
from .coordinator import ParameterRuntime
from .module import BackwardPass, ModuleParameters

__all__ = ["BackwardPass", "BoundParameters", "ModuleParameters", "ParameterRuntime"]
