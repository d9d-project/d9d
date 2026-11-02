"""Split backward passes for input and weight gradients, and the context that custom autograd functions check."""

from .backward import (
    BackwardInputResult,
    ParamGroup,
    stage_backward_full,
    stage_backward_input,
    stage_backward_weight,
)
from .context import GLOBAL_GRAD_CONTEXT, GlobalGradContext, GradDirection, needs_input_grad

__all__ = [
    "GLOBAL_GRAD_CONTEXT",
    "BackwardInputResult",
    "GlobalGradContext",
    "GradDirection",
    "ParamGroup",
    "needs_input_grad",
    "stage_backward_full",
    "stage_backward_input",
    "stage_backward_weight",
]
