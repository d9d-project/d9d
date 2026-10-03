from contextlib import contextmanager

import torch
from torch.distributed.tensor import DTensor
from torch.profiler import record_function

from d9d.core.dist_context import DistributedContext
from d9d.core.offload import Offloadable, OffloadContext, OnloadContext
from d9d.internals.grad_sync import GradientSynchronizer
from d9d.loop.config import GradientManagerConfig
from d9d.metric.impl.aggregation import WeightedMeanMetric

from .model_stage_factory import TrackedModules


class GradientManager(Offloadable):
    """Manages the lifecycle of gradients during the training loop.

    It synchronizes gradients across ranks through a ``GradientSynchronizer``, sets the gradient dtype
    and divides gradients by the accumulated loss weight before the optimizer step.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        tracked_modules: TrackedModules,
        config: GradientManagerConfig,
    ):
        """Constructs the ``GradientManager`` object.

        Args:
            dist_context: The distributed context.
            tracked_modules: The model stages whose gradients are managed.
            config: The gradient handling configuration.
        """
        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._config = config
        self._loss = WeightedMeanMetric()
        self._loss.to("cuda")

        self._grad_sync = GradientSynchronizer(
            [list(module.parameters()) for module in self._tracked_modules.modules],
            bucket_size_mb=self._config.bucket_size_mb,
        )
        self._grads_to_scale: list[torch.Tensor] | None = None

        self._installed = False
        self._offloaded = False
        self._in_flight_count = 0

    def _setup_grad_dtype(self):
        if self._config.grad_dtype is None:
            return

        for mod in self._tracked_modules.modules:
            for param in mod.parameters():
                if param.requires_grad:
                    param.grad_dtype = getattr(torch, self._config.grad_dtype)

    def _bind_grads_to_scale(self):
        grads_to_scale: list[torch.Tensor] = []

        for mod in self._tracked_modules.modules:
            for param in mod.parameters():
                if param.grad is None:
                    continue
                grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                grads_to_scale.append(grad)

        self._grads_to_scale = grads_to_scale

    def _unbind_grads_to_scale(self):
        self._grads_to_scale = None

    def _scale_grads(self):
        if self._grads_to_scale is None:
            raise ValueError("Gradients are not bound. Call sync_and_scale() inside the install() context.")

        scale_factor = 1.0 / self._loss.accumulated_weight
        if len(self._grads_to_scale) > 0:
            torch._foreach_mul_(self._grads_to_scale, scale_factor)

    def _bind(self):
        self._setup_grad_dtype()
        self._grad_sync.bind()
        self._bind_grads_to_scale()

    def _unbind(self):
        self._unbind_grads_to_scale()
        self._grad_sync.unbind()

    @contextmanager
    def install(self):
        """Activates gradient handling while the context is open.

        It sets the gradient dtype, installs the backward hooks that synchronize gradients
        and binds the gradients for later scaling.
        """
        self._bind()
        self._installed = True
        yield
        self._installed = False
        self._unbind()

    def set_required_accumulations(self, require_accumulations: int):
        """Sets how many backward passes are accumulated before gradients are reduced this step.

        It must be called before the backward passes of each step, because the pack length (and so the
        number of accumulations) can vary between steps.

        Args:
            require_accumulations: Number of backward passes in the current step.
        """
        self._grad_sync.set_required_accumulations(require_accumulations)

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

        1.  Waits for all gradient synchronization hooks to complete.
        2.  Synchronizes the accumulated loss/weights across the distributed context.
        3.  Scales the gradients by the inverse of the total accumulated weight to
            normalize them.
        """
        with record_function("Wait & Scale Gradients"):
            self._grad_sync.wait()

            if self._dist_context.mesh_params.is_distributed:
                self._loss.sync(self._dist_context)
            self._scale_grads()

    def compute_global_loss(self) -> torch.Tensor:
        """Calculates the final weighted mean loss.

        Returns:
            The averaged loss scalar across all accumulation steps and ranks.
        """
        return self._loss.compute()

    def zero_grad(self):
        """Resets the internal state for the next training step.

        This clears the accumulated gradients in the synchronizer and resets the
        loss metrics.
        """
        self._grad_sync.zero_grad()
        self._loss.reset()
        self._in_flight_count = 0

    @property
    def has_in_flight_gradients(self) -> bool:
        """Whether a gradient accumulation is in flight.

        It becomes ``True`` after ``add_loss_with_weight`` and ``False`` after ``zero_grad``. During an
        accumulation, partial gradients live in the synchronizer buckets, so offloading would lose them.
        """
        return self._in_flight_count > 0

    def offload(self, ctx: OffloadContext) -> None:
        """Releases the GPU memory held by the gradient state.

        The synchronizer bucket buffers are released and the residual loss accumulator is reset.

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the gradient state is already offloaded, or if the manager is not installed.
        """
        if self._offloaded:
            raise RuntimeError("GradientManager is already offloaded.")
        if not self._installed:
            raise RuntimeError("GradientManager must be installed before it can be offloaded.")

        self._unbind()
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
