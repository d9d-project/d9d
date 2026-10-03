from typing import TypeAlias, TypeVar

import torch

TLeaf = TypeVar("TLeaf")

PyTree: TypeAlias = TLeaf | list["PyTree[TLeaf]"] | dict[str, "PyTree[TLeaf]"] | tuple["PyTree[TLeaf]", ...]
"""Type alias for a recursive tree of data.

The tree nests standard Python containers (dicts, lists, tuples) to any depth. Its leaves have type ``TLeaf``.
It describes nested state dicts and arguments of functions that traverse them recursively, similar to
``torch.utils._pytree``.
"""

TensorTree: TypeAlias = PyTree[torch.Tensor]
"""Type alias for a ``PyTree`` whose leaves are tensors."""

ScalarTree: TypeAlias = PyTree[str | float | int | bool]
"""Type alias for a ``PyTree`` whose leaves are Python scalars (``str``, ``float``, ``int``, ``bool``)."""
