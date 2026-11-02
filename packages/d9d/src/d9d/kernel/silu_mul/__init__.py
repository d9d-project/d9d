"""Fused SiLU-and-multiply kernels, the activation of SwiGLU."""

from .function import silu_mul

__all__ = [
    "silu_mul",
]
