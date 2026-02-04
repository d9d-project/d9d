from contextlib import contextmanager
from enum import StrEnum


class GradDirection(StrEnum):
    """
    Enum representing the specific gradient edges to compute.

    This is used to manually control gradient flow in custom autograd functions
    during split backward passes.

    Attributes:
        inputs: Mark gradient edge as pointing to the module's inputs (activations).
        weight: Mark gradient edge as pointing to the module's parameters (weights).
    """

    inputs = "inputs"
    weight = "weights"


class GlobalGradContext:
    """
    Global state manager for controlling gradient computation in custom autograd functions.

    This context addresses a limitation in PyTorch where custom `torch.autograd.Function`
    implementations set `ctx.needs_input_grad` to True for all edges requiring grad,
    even during partial backward passes (e.g., `torch.autograd.backward(inputs=...)`).

    For additional information on this limitation, please refer to a
        [related issue](https://github.com/pytorch/pytorch/issues/174017).

    This class allows:

    1. For the training code - to explicitly signal which gradient edges (inputs vs weights)
        should currently be computed, allowing custom ops to skip unnecessary computations.
    2. For module code - to check whether it's required to compute a gradient edge.
    """

    def __init__(self):
        """Constructs a GlobalGradContext object with all directions enabled by default."""

        # both directions by default
        self._enabled_directions: set[GradDirection] = {GradDirection.inputs, GradDirection.weight}

    def check_direction(self, direction: GradDirection | None) -> bool:
        """
        Checks if the gradient calculation for the given direction is currently enabled.

        Args:
            direction: The direction to check (inputs or weights). If None,
                returns True.

        Returns:
            True if the direction is enabled or None is passed, False otherwise.
        """

        if direction is None:
            return True

        return direction in self._enabled_directions

    @contextmanager
    def with_directions(self, *directions: GradDirection):
        """
        Sets the enabled gradient directions, overriding the current state.

        Args:
            *directions: Variable number of GradDirection enums to enable.
        """
        prev_directions = self._enabled_directions
        self._enabled_directions = set(directions)
        yield
        self._enabled_directions = prev_directions


GLOBAL_GRAD_CONTEXT = GlobalGradContext()
"""
The singleton instance of GlobalGradContext.

This should be used by custom autograd functions to check `GLOBAL_GRAD_CONTEXT.check_direction()`
during their backward pass.
"""
