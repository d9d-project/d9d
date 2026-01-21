from typing import TypeAlias, TypeVar

import torch

TLeaf = TypeVar("TLeaf")

PyTree: TypeAlias = TLeaf | list["PyTree[TLeaf]"] | dict[str, "PyTree[TLeaf]"] | tuple["PyTree[TLeaf]", ...]
"""
A recursive type definition representing a tree of data.

This type alias covers standard Python containers (dictionaries, lists, tuples)
nested arbitrarily deep, terminating in a leaf node of type `TLeaf`.

This is commonly used for handling nested state dictionaries or arguments
passed to functions that support recursive traversal (similar to `torch.utils._pytree`).
"""

TensorTree: TypeAlias = PyTree[torch.Tensor]
"""
A recursive tree structure where the leaf nodes are PyTorch Tensors.
"""
