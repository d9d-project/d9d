"""
This module provides high-level wrappers around `torch.distributed` collective operations.
"""


from .object import gather_object, all_gather_object
from .tensor import gather, all_gather, gather_variadic_shape, all_gather_variadic_shape

__all__ = [
    "gather_object",
    "all_gather_object",
    "gather",
    "all_gather",
    "gather_variadic_shape",
    "all_gather_variadic_shape"
]
