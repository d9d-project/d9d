"""Autograd extensions that let custom autograd functions skip gradient work in split backward passes."""

from .grad_context import GLOBAL_GRAD_CONTEXT, GlobalGradContext, GradDirection

__all__ = [
    "GLOBAL_GRAD_CONTEXT",
    "GlobalGradContext",
    "GradDirection",
]
