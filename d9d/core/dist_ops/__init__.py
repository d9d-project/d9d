"""Wrappers around ``torch.distributed`` collectives that allocate the output buffers."""

from .object import all_gather_object, gather_object
from .tensor import all_gather, all_gather_variadic_shape, gather, gather_variadic_shape

__all__ = [
    "all_gather",
    "all_gather_object",
    "all_gather_variadic_shape",
    "gather",
    "gather_object",
    "gather_variadic_shape",
]
