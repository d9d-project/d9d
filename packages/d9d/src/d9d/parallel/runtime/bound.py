from types import TracebackType
from typing import Self

import torch
from torch import nn

from .bucket import BucketGradients, SideStreams
from .module import BoundModule, ModuleParameters


class BoundParameters:
    """The parameters of the registered modules between ``bind()`` and ``close()``, and the gradients of a step.

    As a context manager, it calls ``close()`` on exit.
    """

    def __init__(self, streams: SideStreams, modules: dict[nn.Module, BoundModule]):
        """Constructs the ``BoundParameters`` object.

        Args:
            streams: The side streams of the modules.
            modules: The parameters of each registered module.
        """
        self._streams = streams
        self._modules = modules
        self._closed = False
        # Whether wait() finished the gradients of the current step.
        self._finished = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ):
        self.close()

    @property
    def closed(self) -> bool:
        """Whether ``close()`` ran."""
        return self._closed

    def _require_open(self):
        if self._closed:
            raise ValueError("The bound parameters are closed. Call bind() again.")

    def _gradients(self) -> list[BucketGradients]:
        return [item for bound in self._modules.values() for item in bound.gradients]

    def parameters(self, module: nn.Module) -> ModuleParameters:
        """Returns the parameters of a registered module.

        Args:
            module: The module.

        Returns:
            The parameters of the module.

        Raises:
            ValueError: If the bound parameters are closed, or if ``module`` is not registered.
        """
        self._require_open()
        if module not in self._modules:
            raise ValueError(f"The module ({type(module).__name__}) is not registered. Call register() before bind().")
        return self._modules[module].parameters

    def wait(self):
        """Finishes the gradients of the step and runs the post-accumulate-grad hooks of the sharded parameters.

        Afterwards, the current stream sees the reduced gradients.

        Raises:
            ValueError: If the bound parameters are closed, if the step is already finished, if the final backward
                passes did not reduce a group, if a gradient of a sharded parameter was replaced, or if a hook set a
                gradient to ``None``.
        """
        self._require_open()
        if self._finished:
            raise ValueError("The gradients of this step are already finished. Call zero_grad() before the next step.")

        for module, bound in self._modules.items():
            missing = bound.unreduced_groups()
            if missing:
                raise ValueError(
                    f"The backward passes of the module ({type(module).__name__}) did not reduce the parameter "
                    f"groups ({missing}). Run a backward pass with final=True over every region of the module before "
                    "wait()."
                )

        gradients = self._gradients()
        for item in gradients:
            item.check()

        self._streams.wait_reductions()
        # The optimizer writes the sharded weights after wait(), and a gather that no region used can still read them.
        for bound in self._modules.values():
            bound.invalidate_weights()
        # From here on, a retry would run the hooks that already ran.
        self._finished = True
        for item in gradients:
            item.finish()

    @torch.no_grad()
    def scale_grads(self, factor: torch.Tensor):
        """Multiplies all gradients by a factor in place.

        Args:
            factor: The scalar factor.

        Raises:
            ValueError: If the bound parameters are closed, or if ``wait()`` did not finish the step.
        """
        self._require_open()
        # Before wait(), a reduction on the reduce stream can still overwrite the scaled gradients.
        if not self._finished:
            raise ValueError("The gradients of this step are not finished. Call wait() before scale_grads().")
        tensors = [item.tensor for item in self._gradients()]
        if tensors:
            torch._foreach_mul_(tensors, factor)

    def zero_grad(self):
        """Starts a step: zeroes the gradients and drops the unsharded weights.

        Raises:
            ValueError: If the bound parameters are closed.
        """
        self._require_open()
        # A failed step can leave reductions queued on the reduce stream.
        self._streams.wait_reductions()
        for bound in self._modules.values():
            bound.zero_grad()
            # A gather that started before the optimizer step holds the weights from before it.
            bound.invalidate_weights()
        self._finished = False

    def invalidate_weights(self):
        """Drops the unsharded weights, so the next region gathers the sharded weights again.

        Raises:
            ValueError: If the bound parameters are closed.
        """
        self._require_open()
        for bound in self._modules.values():
            bound.invalidate_weights()

    def close(self):
        """Releases the buffers, the hooks and the unsharded parameters.

        It sets the gradients of the sharded parameters to ``None``. Afterwards, the methods of this object and of its
        ``ModuleParameters`` raise. A second call does nothing.
        """
        if self._closed:
            return
        self._streams.close()
        for bound in self._modules.values():
            bound.close()
        self._modules = {}
        self._closed = True
