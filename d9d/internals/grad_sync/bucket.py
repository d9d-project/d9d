import abc

import torch
from torch import nn, Tensor
from torch.distributed import DeviceMesh
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.utils.hooks import RemovableHandle

from .placement_helper import mark_grad_sync_complete, mark_grad_sync_awaiting, dist_grad_from_local


class AbstractGradientBucket(abc.ABC):
    """
    Interface for a bucket containing a subset of model parameters.

    A bucket manages the memory layout and synchronization lifecycle of the
    gradients associated with its parameters.
    """

    @abc.abstractmethod
    def bind(self):
        """
        Initializes the bucket state.

        This involves allocating contiguous memory buffers (if applicable),
        registering backward hooks, and preparing the gradients for accumulation.
        """

        pass

    @abc.abstractmethod
    def unbind(self):
        """
        Cleans up the bucket state.

        Removes hooks, deallocates buffers, and detaches gradients.
        """

        pass

    @abc.abstractmethod
    def wait(self):
        """
        Waits for any asynchronous synchronization operations to complete.
        """

        pass

    @abc.abstractmethod
    def zero_grad(self):
        """
        Zeros out the gradients and resets accumulation counters.
        """

        pass


class LocalGradientBucket(AbstractGradientBucket):
    """
    A bucket for parameters that do not require distributed synchronization.
    """

    def __init__(self, params: list[nn.Parameter]):
        """
        Constructs a LocalGradientBucket.

        Args:
            params: List of parameters to manage.
        """

        self._params = params

    def bind(self):
        """
        No-op for local buckets as they do not require special buffering.
        """

        pass

    def unbind(self):
        """
        No-op for local buckets.
        """

        pass

    def wait(self):
        """
        No-op as no async communication is performed.
        """

        pass

    @torch.no_grad()
    def zero_grad(self):
        """
        Directly zeros the grad attribute of the parameters.
        """

        for param in self._params:
            param.grad.zero_()


class AccumulationCounter:
    """
    Tracks the number of gradient accumulation steps for a set of parameters.
    """

    def __init__(self, require_accumulations: int, parameters: list[nn.Parameter]):
        """
        Constructs an AccumulationCounter.

        Args:
            require_accumulations: Number of accumulations required before sync.
            parameters: List of parameters to track.
        """

        self._require_accumulations = require_accumulations
        self._param_to_sync_count = {param: 0 for param in parameters}

    def reset(self):
        """
        Resets all counters to zero.
        """

        self._param_to_sync_count = {param: 0 for param in self._param_to_sync_count.keys()}

    def update(self, param: nn.Parameter):
        """
        Increments the counter for a specific parameter.

        Args:
            param: The parameter that finished a backward step.
        """

        self._param_to_sync_count[param] += 1

    def is_ready(self) -> bool:
        """
        Checks if all parameters have reached the required number of accumulations.

        Returns:
            True if synchronization can proceed.
        """

        return all(x == self._require_accumulations for x in self._param_to_sync_count.values())


class SyncGradientBucket(AbstractGradientBucket):
    """
    A bucket that manages a contiguous memory buffer for gradients and performs async reduction.

    This bucket flattens the gradients of its parameters into a single contiguous
    Tensor to enable efficient batched all-reduce operations.
    """

    def __init__(
            self,
            parameters: list[nn.Parameter],
            require_accumulations: int,
            device: torch.device,
            grad_dtype: torch.dtype,
            reduce_mesh: DeviceMesh
    ):
        """
        Constructs a SyncGradientBucket.

        Args:
            parameters: List of parameters to manage.
            require_accumulations: Number of accumulations before triggering reduce.
            device: Device where parameters reside.
            grad_dtype: Data type for the gradients.
            reduce_mesh: DeviceMesh on which reduction happens.
        """

        self._params = parameters
        self._accum_counter = AccumulationCounter(require_accumulations, parameters)
        self._device = device
        self._grad_dtype = grad_dtype
        self._reduce_group: dist.ProcessGroup = reduce_mesh.get_group()


        self._buffer: Tensor | None = None
        self._hooks: list[RemovableHandle] | None = None

        self._reduce_job: dist.Work | None = None

    def _bind_buffer(self):
        """
        Allocates the flat buffer and redirects parameter gradients to view into it.
        """

        buffer_size = sum(param.data.to_local().numel() for param in self._params)

        self._buffer = torch.zeros(
            (buffer_size,),
            dtype=self._grad_dtype,
            device=self._device
        )

        offset = 0

        for param in self._params:
            data: DTensor = param.data
            local_param = data.to_local()

            local_grad = self._buffer[offset:offset + local_param.numel()].view(local_param.shape)

            param.grad = dist_grad_from_local(data, local_grad)

            offset += local_param.numel()

    @torch.no_grad()
    def _post_accumulation_hook(self, param: nn.Parameter):
        """
        Hook executed after backward pass for a parameter.

        Updates the accumulation counter and triggers the asynchronous all-reduce
        if the bucket is ready.

        Args:
            param: The parameter that finished backward pass.
        """

        self._accum_counter.update(param)

        if not self._accum_counter.is_ready():
            return

        self._reduce_job = dist.all_reduce(
            self._buffer,
            op=dist.ReduceOp.SUM,
            group=self._reduce_group,
            async_op=True
        )

    def _bind_hooks(self):
        """
        Registers post-accumulate hooks on all parameters.
        """

        hooks = []
        for param in self._params:
            hooks.append(param.register_post_accumulate_grad_hook(self._post_accumulation_hook))
        self._hooks = hooks

    @torch.no_grad()
    def bind(self):
        """
        Allocates the contiguous buffer and registers hooks.
        """

        self._bind_buffer()
        self._bind_hooks()

    def _unbind_buffer(self):
        """
        Deallocates the buffer and clears parameter gradients.
        """

        self._buffer = None

        for param in self._params:
            param.grad = None

    def _unbind_hooks(self):
        """
        Removes all registered hooks.
        """

        for hook in self._hooks:
            hook.remove()
        self._hooks = None

    @torch.no_grad()
    def unbind(self):
        """
        Cleans up buffer and hooks.
        """

        self._unbind_buffer()
        self._unbind_hooks()

    @torch.no_grad()
    def wait(self):
        """
        Blocks until the async reduction job is complete and marks sync as done.
        """

        self._reduce_job.wait()

        for param in self._params:
            mark_grad_sync_complete(param)

    @torch.no_grad()
    def zero_grad(self):
        """
        Zeros the contiguous buffer, resets counters, and marks params as awaiting sync.

        Raises:
            ValueError: If the buffer is not initialized (call bind first).
        """

        if self._buffer is None:
            raise ValueError('Buffer is not initialized')

        self._buffer.zero_()
        self._accum_counter.reset()
        for param in self._params:
            mark_grad_sync_awaiting(param)
