"""Operators that run the user task over the microbatches of a pack."""

from .inference import InferenceTaskOperator
from .train import TrainTaskOperator

__all__ = [
    "InferenceTaskOperator",
    "TrainTaskOperator",
]
