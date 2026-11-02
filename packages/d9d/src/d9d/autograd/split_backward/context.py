from contextlib import contextmanager
from enum import StrEnum

from torch.autograd.function import BackwardCFunction


class GradDirection(StrEnum):
    """Gradient edges that a custom autograd function can compute.

    Custom autograd functions use it to skip gradient work during split backward passes.

    Attributes:
        inputs: The gradient edges to the module inputs (activations).
        weight: The gradient edges to the module parameters (weights).
    """

    inputs = "inputs"
    weight = "weights"


class GlobalGradContext:
    """Global state that controls gradient computation in custom autograd functions.

    PyTorch sets ``ctx.needs_input_grad`` to ``True`` for every edge that requires grad in a custom
    ``torch.autograd.Function``. It does so even in a partial backward pass, such as
    ``torch.autograd.backward(inputs=...)``. See the
    [related issue](https://github.com/pytorch/pytorch/issues/174017).

    This class works around the limitation:

    1.  Training code sets which gradient edges (inputs or weights) to compute now.
    2.  Module code checks whether it must compute a gradient edge, and skips the work otherwise.
    """

    def __init__(self):
        """Constructs the ``GlobalGradContext`` object with both directions enabled."""
        self._enabled_directions: set[GradDirection] = {GradDirection.inputs, GradDirection.weight}

    def check_direction(self, direction: GradDirection | None) -> bool:
        """Checks whether gradient computation for the given direction is enabled.

        Args:
            direction: The direction to check.

        Returns:
            ``True`` if the direction is enabled or ``direction`` is ``None``, ``False`` otherwise.
        """
        if direction is None:
            return True

        return direction in self._enabled_directions

    @contextmanager
    def with_directions(self, *directions: GradDirection):
        """Enables only the given gradient directions inside the context.

        The previous directions are restored when the context exits, even if the block raises an exception.

        Args:
            *directions: The gradient directions to enable.
        """
        prev_directions = self._enabled_directions
        self._enabled_directions = set(directions)
        try:
            yield
        finally:
            self._enabled_directions = prev_directions


GLOBAL_GRAD_CONTEXT = GlobalGradContext()
"""The singleton ``GlobalGradContext``.

Custom autograd functions check it in their backward pass through ``needs_input_grad()``.
"""


def needs_input_grad(ctx: BackwardCFunction, index: int, direction: GradDirection) -> bool:
    """Checks whether the backward pass of a custom autograd function must compute the gradient of an input.

    Use it in ``backward`` instead of ``ctx.needs_input_grad``, so that split backward passes skip the gradients of
    the other direction.

    Args:
        ctx: The autograd context passed to ``backward``.
        index: The position of the input in the arguments of ``forward``, without ``ctx``.
        direction: The gradient direction of the input.

    Returns:
        ``True`` if autograd needs the gradient of the input and ``GLOBAL_GRAD_CONTEXT`` enables its direction.
    """
    return ctx.needs_input_grad[index] and GLOBAL_GRAD_CONTEXT.check_direction(direction)
