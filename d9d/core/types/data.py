from collections.abc import Callable, Sequence
from typing import TypeAlias, TypeVar

from .pytree import PyTree

TDataTree = TypeVar("TDataTree", bound=PyTree)

CollateFn: TypeAlias = Callable[[Sequence[TDataTree]], TDataTree]
"""
Type alias for a function that collates a sequence of samples into a batch.

The function receives a sequence of individual data point structures (PyTrees)
and is responsible for stacking or merging them into a single batched structure.
"""
