"""
This module provides high-level wrappers around `torch.distributed` collective operations.
"""


from .object import all_gather_object, gather_object
from .tensor import all_gather, all_gather_variadic_shape, gather, gather_variadic_shape

__all__ = [
    "all_gather",
    "all_gather_object",
    "all_gather_variadic_shape",
    "gather",
    "gather_object",
    "gather_variadic_shape"
]
