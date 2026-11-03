from contextlib import contextmanager

import torch
from torch.profiler import record_function

from d9d.dist_context import FLAT_DOMAIN, DistributedContext
from d9d.metric.catalogue.aggregation import WeightedMeanMetric
from d9d.offload import Offloadable, OffloadContext, OnloadContext
from d9d.parallel.runtime import ParameterRuntime

from ..config import GradientManagerConfig, dtype_from_name
from .tracked_modules import TrackedModules


class GradientManager(Offloadable):
    """Lifecycle manager for gradients during the training loop.

    It sets the gradient dtype, binds the ``ParameterRuntime`` that reduces gradients across ranks, and divides
    the gradients by the accumulated loss weight before the optimizer step.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        tracked_modules: TrackedModules,
        config: GradientManagerConfig,
        parameter_runtime: ParameterRuntime,
    ):
        """Constructs the ``GradientManager`` object.

        Args:
            dist_context: The distributed context.
            tracked_modules: The model stages whose gradients are managed.
            config: The gradient handling configuration.
            parameter_runtime: The runtime that holds the gradients and reduces them from the pipeline program.
        """
        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._config = config
        self._loss = WeightedMeanMetric()
        self._loss.to("cuda")

        self._runtime = parameter_runtime

        self._installed = False
        self._offloaded = False
        self._in_flight_count = 0

    def _bind(self):
        if self._config.grad_dtype is not None:
            grad_dtype = dtype_from_name(self._config.grad_dtype)
            for module in self._tracked_modules.modules:
                for parameter in module.parameters():
                    if parameter.requires_grad:
                        parameter.grad_dtype = grad_dtype
        self._runtime.bind()

    @contextmanager
    def install(self):
        """Activates gradient handling while the context is open.

        It sets the gradient dtype and binds the ``ParameterRuntime``, which allocates the gradient buffers.
        """
        self._bind()
        self._installed = True
        try:
            yield
        finally:
            self._installed = False
            self._runtime.unbind()

    def add_loss_with_weight(self, loss: torch.Tensor, loss_weight: torch.Tensor):
        """Accumulates a loss value and its corresponding weight into the internal metric.

        Args:
            loss: The computed loss scalar.
            loss_weight: The weight associated with this loss.
        """
        self._loss.update(loss, loss_weight)
        self._in_flight_count += 1

    def sync_and_scale(self):
        """Finalizes gradients to be ready for the optimizer step.

        This method performs the following operations:

        1.  Waits for the gradient reductions of the pipeline program.
        2.  Synchronizes the accumulated loss/weights across the distributed context.
        3.  Scales the gradients by the inverse of the total accumulated weight to
            normalize them.
        """
        with record_function("Wait & Scale Gradients"):
            self._runtime.wait()

            if self._dist_context.mesh_params.is_distributed:
                self._loss.sync(self._dist_context.mesh_for(FLAT_DOMAIN).get_group())
            self._runtime.scale_grads(1.0 / self._loss.accumulated_weight)

    def compute_global_loss(self) -> torch.Tensor:
        """Calculates the final weighted mean loss.

        Returns:
            The averaged loss scalar across all accumulation steps and ranks.
        """
        return self._loss.compute()

    def zero_grad(self):
        """Resets the internal state for the next training step.

        This clears the gradients in the runtime and resets the loss metrics.
        """
        self._runtime.zero_grad()
        self._loss.reset()
        self._in_flight_count = 0

    @property
    def has_in_flight_gradients(self) -> bool:
        """Whether a gradient accumulation is in flight.

        It becomes ``True`` after ``add_loss_with_weight`` and ``False`` after ``zero_grad``. During an
        accumulation, partial gradients live in the runtime buffers, so offloading would lose them.
        """
        return self._in_flight_count > 0

    def offload(self, ctx: OffloadContext) -> None:
        """Releases the GPU memory held by the gradient state.

        The runtime buffers are released and the residual loss accumulator is reset.

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the gradient state is already offloaded, or if the manager is not installed.
        """
        if self._offloaded:
            raise RuntimeError("GradientManager is already offloaded.")
        if not self._installed:
            raise RuntimeError("GradientManager must be installed before it can be offloaded.")

        self._runtime.unbind()
        self._loss.reset()
        self._offloaded = True

    def onload(self, ctx: OnloadContext) -> None:
        """Reallocates on the GPU the gradient state released by ``offload``.

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the gradient state is not offloaded.
        """
        if not self._offloaded:
            raise RuntimeError("GradientManager is not offloaded.")

        if self._installed:
            self._bind()

        self._offloaded = False

    def is_offloaded(self) -> bool:
        """Reports whether the gradient state is offloaded.

        Returns:
            ``True`` if the gradient state is offloaded, otherwise ``False``.
        """
        return self._offloaded
