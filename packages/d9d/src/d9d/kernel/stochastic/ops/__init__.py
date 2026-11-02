"""Triton device functions for stochastic rounding from fp32 to bf16."""

from .round import fp32_to_bf16_kernel

__all__ = [
    "fp32_to_bf16_kernel",
]
