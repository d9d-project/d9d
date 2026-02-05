from contextlib import contextmanager

import torch
from torch.distributed.tensor import DTensor

from d9d.core.dist_context import DistributedContext
from d9d.internals.grad_sync import GradientSynchronizer
from d9d.loop.config import GradientManagerConfig
from d9d.metric.impl import WeightedMeanMetric

from .batch_maths import BatchMaths
from .model_stage_factory import TrackedModules


class GradientManager:
    """
    Manages the lifecycle of gradients during the training loop.

    This class handles gradient synchronization across ranks,
    gradient data type configuration, and loss scaling based on accumulated weights.
    It orchestrates the `GradientSynchronizer` and ensures gradients are correctly
    prepared before the optimizer step.
    """

    def __init__(
            self,
            dist_context: DistributedContext,
            tracked_modules: TrackedModules,
            batch_maths: BatchMaths,
            config: GradientManagerConfig
    ):
        """
        Constructs the GradientManager and initializes the internal synchronizer.

        Args:
            dist_context: The distributed context.
            tracked_modules: Container of model modules to manage gradients for.
            batch_maths: Calculation utility for batch sizes and accumulation steps.
            config: Configuration for gradient handling.
        """

        self._dist_context = dist_context
        self._tracked_modules = tracked_modules
        self._batch_maths = batch_maths
        self._config = config
        self._loss = WeightedMeanMetric()
        self._loss.to("cuda")

        self._grad_sync = GradientSynchronizer(
            [list(module.parameters()) for module in self._tracked_modules.modules],
            bucket_size_mb=self._config.bucket_size_mb,
            require_accumulations=self._batch_maths.num_backward_calls
        )
        self._grads_to_scale: list[torch.Tensor] | None = None

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
            raise ValueError("You should bind the manager first.")

        scale_factor = 1.0 / self._loss.accumulated_weight
        if len(self._grads_to_scale) > 0:
            torch._foreach_mul_(self._grads_to_scale, scale_factor)

    @contextmanager
    def install(self):
        """
        Context manager to activate gradient handling for a forward/backward pass.

        This sets up gradient dtypes, install backward hooks for synchronization via
        the `GradientSynchronizer`, and binds gradients for later scaling. It acts
        as the boundary for the accumulation phase.
        """

        self._setup_grad_dtype()
        self._grad_sync.bind()
        self._bind_grads_to_scale()
        yield
        self._unbind_grads_to_scale()
        self._grad_sync.unbind()

    def add_loss_with_weight(self, loss: torch.Tensor, loss_weight: torch.Tensor):
        """
        Accumulates a loss value and its corresponding weight into the internal metric.

        Args:
            loss: The computed loss scalar.
            loss_weight: The weight asscociated with this loss.
        """

        self._loss.update(loss, loss_weight)

    def sync_and_scale(self):
        """
        Finalizes gradients to be ready for the optimizer step.

        This method performs the following operations:

        1. Waits for all gradient synchronization hooks to complete.
        2. Synchronizes the accumulated loss/weights across the distributed context.
        3. Scales the gradients by the inverse of the total accumulated weight to
           normalize them.
        """

        self._grad_sync.wait()

        if self._dist_context.mesh_params.is_distributed:
            self._loss.trigger_sync(self._dist_context)
            self._loss.wait_sync(self._dist_context)
        self._scale_grads()

    def compute_global_loss(self) -> torch.Tensor:
        """
        Calculates the final weighted mean loss.

        Returns:
            The averaged loss scalar across all accumulation steps and ranks.
        """

        return self._loss.compute()

    def zero_grad(self):
        """
        Resets the internal state for the next training step.

        This clears the accumulated gradients in the synchronizer and resets the
        loss metrics.
        """

        self._grad_sync.zero_grad()
        self._loss.reset()
