from typing import Any, ClassVar, Protocol, TypeAlias, TypeVar

import torch

TLeaf = TypeVar("TLeaf")


# The same protocol as _typeshed.DataclassInstance, which exists only for type checkers.
class _DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]


PyTree: TypeAlias = (
    TLeaf | list["PyTree[TLeaf]"] | dict[str, "PyTree[TLeaf]"] | tuple["PyTree[TLeaf]", ...] | _DataclassInstance
)
"""Type alias for a recursive tree of data.

The tree nests standard Python containers (dicts, lists, tuples) and dataclasses to any depth. Its leaves have type
``TLeaf``. The type does not check the fields of a dataclass.
It describes nested state dicts and arguments of functions that traverse them recursively, similar to
``torch.utils._pytree``.
"""

TensorTree: TypeAlias = PyTree[torch.Tensor]
"""Type alias for a ``PyTree`` whose leaves are tensors."""

ScalarTree: TypeAlias = PyTree[str | float | int | bool]
"""Type alias for a ``PyTree`` whose leaves are Python scalars (``str``, ``float``, ``int``, ``bool``)."""
