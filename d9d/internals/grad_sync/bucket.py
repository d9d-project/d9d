import abc
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.utils.hooks import RemovableHandle

from .placement_helper import dist_grad_from_local


class AbstractGradientBucket(abc.ABC):
    """Abstract base class for a bucket that holds a subset of the model parameters.

    A bucket owns the memory layout of the parameter gradients and the lifecycle of their synchronization.
    """

    @abc.abstractmethod
    def bind(self):
        """Prepares the bucket for gradient accumulation.

        Implementations can allocate a contiguous gradient buffer and register backward hooks.
        """

    @abc.abstractmethod
    def unbind(self):
        """Releases the bucket state: hooks, buffers and gradients."""

    @abc.abstractmethod
    def zero_grad(self):
        """Zeros out the gradients and resets accumulation counters."""

    @abc.abstractmethod
    def mark_sync(self):
        """Marks this bucket as synchronized."""

    @abc.abstractmethod
    def set_required_accumulations(self, require_accumulations: int):
        """Sets how many accumulations must happen before this bucket reduces gradients.

        Args:
            require_accumulations: The number of accumulations required before the reduction.
        """


class LocalGradientBucket(AbstractGradientBucket):
    """A bucket for parameters that do not require distributed synchronization."""

    def __init__(self, params: list[nn.Parameter]):
        """Constructs the ``LocalGradientBucket`` object.

        Args:
            params: The parameters to manage.
        """
        self._params = params

    def bind(self):
        """Does nothing: local gradients need no buffer."""

    def unbind(self):
        """Does nothing."""

    def wait(self):
        """Does nothing: local buckets do not communicate."""

    @torch.no_grad()
    def zero_grad(self):
        """Sets the gradients of the parameters to ``None``."""
        for param in self._params:
            param.grad = None

    def mark_sync(self):
        """Does nothing."""

    def set_required_accumulations(self, require_accumulations: int):
        """Does nothing: local buckets never reduce."""


class AccumulationCounter:
    """Counter of gradient accumulation steps for a set of parameters."""

    def __init__(self, parameters: list[nn.Parameter]):
        """Constructs the ``AccumulationCounter`` object.

        Args:
            parameters: The parameters to track.
        """
        self._require_accumulations: int | None = None
        self._param_to_sync_count = {param: 0 for param in parameters}

    def reset(self):
        """Resets all counters to zero."""
        self._param_to_sync_count = {param: 0 for param in self._param_to_sync_count}

    def set_required_accumulations(self, require_accumulations: int):
        """Sets the number of accumulations required before the bucket is ready to sync.

        Args:
            require_accumulations: The number of accumulations required before the sync.
        """
        self._require_accumulations = require_accumulations

    def update(self, param: nn.Parameter):
        """Increments the counter for a specific parameter.

        Args:
            param: The parameter that finished a backward step.
        """
        self._param_to_sync_count[param] += 1

    def is_ready(self) -> bool:
        """Checks whether all parameters reached the required number of accumulations.

        Returns:
            ``True`` if the synchronization can start.

        Raises:
            RuntimeError: If the required accumulation count is not set.
        """
        if self._require_accumulations is None:
            raise RuntimeError("The required accumulation count was not set. Call set_required_accumulations() first.")
        return all(x == self._require_accumulations for x in self._param_to_sync_count.values())


class SyncGradientBucket(AbstractGradientBucket):
    """A bucket that keeps its gradients in one contiguous buffer and reduces them asynchronously.

    With one buffer, a single all-reduce per process group covers all gradients of the bucket.
    """

    def __init__(
        self,
        parameters: list[nn.Parameter],
        device: torch.device,
        grad_dtype: torch.dtype,
        reduce_mesh: DeviceMesh,
        communicate_stream: torch.cuda.Stream,
    ):
        """Constructs the ``SyncGradientBucket`` object.

        Args:
            parameters: The parameters to manage.
            device: The device of the parameters.
            grad_dtype: The dtype of the gradients.
            reduce_mesh: The device mesh to reduce the gradients over.
            communicate_stream: The CUDA stream to run the asynchronous communication on.

        Raises:
            ValueError: If any parameter does not hold ``DTensor`` data.
        """
        if not all(isinstance(x.data, DTensor) for x in parameters):
            raise ValueError("All parameters of a SyncGradientBucket must hold DTensor data.")

        self._params = parameters
        self._accum_counter = AccumulationCounter(parameters)
        self._device = device
        self._grad_dtype = grad_dtype
        # Iterate from innermost to outermost group.
        self._reduce_groups: list[dist.ProcessGroup] = reduce_mesh.get_all_groups()[::-1]

        self._buffer: Tensor | None = None
        self._hooks: list[RemovableHandle] | None = None

        self._communicate_stream = communicate_stream
        self._ready_to_sync = False

    def _bind_buffer(self):
        """Allocates the flat buffer and makes the parameter gradients views into it."""
        buffer_size = sum(cast(DTensor, param.data).to_local().numel() for param in self._params)

        self._buffer = torch.zeros((buffer_size,), dtype=self._grad_dtype, device=self._device)

        offset = 0

        for param in self._params:
            data = cast(DTensor, param.data)
            local_param = data.to_local()

            local_grad = self._buffer[offset : offset + local_param.numel()].view(local_param.shape)

            param.grad = dist_grad_from_local(data, local_grad)

            offset += local_param.numel()

    @torch.no_grad()
    def _post_accumulation_hook(self, param: nn.Parameter):
        """Counts an accumulation of ``param`` and starts the asynchronous all-reduce once the bucket is ready.

        Args:
            param: The parameter whose gradient was accumulated.

        Raises:
            ValueError: If the previous reduction of the bucket was not waited for, or if the buffer is not
                allocated.
        """
        self._accum_counter.update(param)

        if not self._accum_counter.is_ready():
            return

        if self._ready_to_sync:
            raise ValueError(
                "The bucket is ready to reduce again, but its previous reduction was not waited for. "
                "Call wait() on the synchronizer after each reduction."
            )

        buffer = self._buffer
        if buffer is None:
            raise ValueError("The gradient buffer is not allocated. Call bind() first.")

        with record_function("Gradient Sync"):
            # The reduction must see the gradients that the backward pass wrote on the current stream.
            self._communicate_stream.wait_stream(torch.cuda.current_stream())
            # A side stream overlaps the reduction with the rest of the backward pass; the all-reduces share the
            # buffer, so they run one after another on that stream.
            with torch.cuda.stream(self._communicate_stream):
                for group in self._reduce_groups:
                    dist.all_reduce(buffer, op=dist.ReduceOp.SUM, group=group)
            self._ready_to_sync = True

    def _bind_hooks(self):
        """Registers post-accumulate hooks on all parameters."""
        hooks = []
        for param in self._params:
            hooks.append(param.register_post_accumulate_grad_hook(self._post_accumulation_hook))
        self._hooks = hooks

    @torch.no_grad()
    def bind(self):
        """Allocates the contiguous buffer and registers the hooks."""
        self._bind_buffer()
        self._bind_hooks()

    def _unbind_buffer(self):
        """Frees the buffer and clears the parameter gradients."""
        self._buffer = None

        for param in self._params:
            param.grad = None

    def _unbind_hooks(self):
        """Removes all registered hooks."""
        if self._hooks is None:
            return

        for hook in self._hooks:
            hook.remove()
        self._hooks = None

    @torch.no_grad()
    def unbind(self):
        """Frees the buffer, clears the gradients and removes the hooks."""
        self._unbind_buffer()
        self._unbind_hooks()

    @torch.no_grad()
    def zero_grad(self):
        """Zeros the gradient buffer and resets the accumulation counters.

        Raises:
            ValueError: If the buffer is not allocated.
        """
        buffer = self._buffer
        if buffer is None:
            raise ValueError("The gradient buffer is not allocated. Call bind() first.")

        buffer.zero_()
        self._accum_counter.reset()

    def mark_sync(self):
        if not self._ready_to_sync:
            raise ValueError(
                "The bucket is not ready to sync: its parameters did not reach the required accumulation count."
            )

        self._ready_to_sync = False

    def set_required_accumulations(self, require_accumulations: int):
        self._accum_counter.set_required_accumulations(require_accumulations)
