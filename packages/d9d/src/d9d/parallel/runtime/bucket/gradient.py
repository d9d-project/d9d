"""The gradients of the buckets and their reductions: reduce-scatter for gathered buckets, all-reduce for the rest."""

from collections.abc import Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from d9d.parallel.placement import dist_grad_from_local

from ..plan import BucketPlan, GatherPlan, chunk_places, local_tensor
from .buffer import ManagedBuffer, PendingGradientBuffers
from .parameter import ParameterHandle, run_post_accumulate_hooks


def _check_accumulated(handles: Sequence[ParameterHandle], grads: Sequence[Tensor]):
    for handle, grad in zip(handles, grads, strict=True):
        if handle.unsharded.grad is not grad:
            raise ValueError(
                f"The gradient of the parameter ({handle.plan.name}) was replaced during the backward pass, e.g. by "
                "backward(create_graph=True). Call backward() without create_graph=True."
            )


class BucketGradients:
    """The gradients of a bucket on this rank, which the ``.grad`` of each sharded parameter views."""

    def __init__(
        self,
        handles: Sequence[ParameterHandle],
        tensor: Tensor,
        local_views: Sequence[Tensor],
        replica_process_groups: Sequence[ProcessGroup],
    ):
        """Constructs the ``BucketGradients`` object.

        It zeroes the flat gradients and points the gradients of the sharded parameters at them.

        Args:
            handles: The parameters of the bucket.
            tensor: The flat gradients: the shard of this rank in a gathered bucket, the full gradients otherwise.
            local_views: The view of ``tensor`` that holds the local gradient of each parameter.
            replica_process_groups: The process groups over which the gradients are summed, in reduction order.
        """
        self._tensor = tensor
        self._handles = handles
        self._replica_process_groups = replica_process_groups
        self._reduced = False
        self._views = [
            dist_grad_from_local(h.plan.sharded.data, view) if isinstance(h.plan.sharded.data, DTensor) else view
            for h, view in zip(handles, local_views, strict=True)
        ]
        self.reset()

    @property
    def tensor(self) -> Tensor:
        """The flat gradients."""
        return self._tensor

    @property
    def reduced(self) -> bool:
        """Whether the final reduction of the step ran."""
        return self._reduced

    def reset(self):
        """Zeroes the gradients for a new step and points the gradients of the sharded parameters at them again."""
        self._tensor.zero_()
        self._reduced = False
        for handle, view in zip(self._handles, self._views, strict=True):
            handle.plan.sharded.grad = view

    def reduce_replicas(self):
        """Sums the gradients across replicas on the current stream, which ends the reduction of the step."""
        for process_group in self._replica_process_groups:
            dist.all_reduce(self._tensor, op=dist.ReduceOp.SUM, group=process_group)
        self._reduced = True

    def check(self):
        """Checks that the gradient of every sharded parameter still views the bucket.

        Raises:
            ValueError: If a gradient was replaced since ``reset()``.
        """
        for handle, view in zip(self._handles, self._views, strict=True):
            if handle.plan.sharded.grad is not view:
                raise ValueError(
                    f"The gradient of the parameter ({handle.plan.name}) was replaced, so the runtime can neither "
                    "scale nor clear it. Change the gradient in place."
                )

    @torch.no_grad()
    def finish(self):
        """Runs the post-accumulate-grad hooks of the sharded parameters.

        Call it after the final reduction. A gradient that a hook replaces is copied back into the bucket.

        Raises:
            ValueError: If a hook set a gradient to ``None``.
        """
        for handle, view in zip(self._handles, self._views, strict=True):
            sharded = handle.plan.sharded
            run_post_accumulate_hooks(handle)

            # The hooks can replace the gradient, as PyTorch allows, so its value goes back into the bucket.
            current = sharded.grad
            if current is view:
                continue
            if current is None:
                raise ValueError(
                    f"A hook set the gradient of the parameter ({handle.plan.name}) to None. Zero it in place, e.g. "
                    "with grad.zero_()."
                )
            view.copy_(current)
            sharded.grad = view

    def clear(self):
        """Sets the gradients of the sharded parameters to ``None``."""
        for handle in self._handles:
            handle.plan.sharded.grad = None


class ShardedReduction:
    """The gradient buffer of a trainable gathered bucket and the reduce-scatter into its gradient shard."""

    def __init__(
        self,
        plan: BucketPlan,
        gather: GatherPlan,
        handles: Sequence[ParameterHandle],
        reduce_stream: torch.cuda.Stream,
        pending: PendingGradientBuffers,
    ):
        """Constructs the ``ShardedReduction`` object.

        It points the gradients of the unsharded parameters at the gradient buffer, which has no memory yet. The
        gradient buffer belongs to the current stream, so the backward passes must run on it.

        Args:
            plan: The plan of the bucket.
            gather: The gather of the bucket, whose process group reduce-scatters.
            handles: The parameters, in buffer order.
            reduce_stream: The stream that reduces gradients.
            pending: The gradient buffers of all buckets that the reduce stream may still read.
        """
        self._groups = plan.groups
        self._process_group = gather.process_group
        # The bucket is trainable, and plan_parameters() rejects a trainable parameter without grad_dtype.
        grad_dtype = cast(torch.dtype, plan.grad_dtype)
        self._handles = handles
        self._reduce_stream = reduce_stream
        self._pending = pending
        # Whether a reduce-scatter of this step already wrote the gradient shard.
        self._shard_written = False

        parts = gather.process_group.size()
        widths = [sizes[0] for sizes in gather.shard_sizes]
        self._buffer = ManagedBuffer(parts * sum(widths), grad_dtype, torch.cuda.current_stream())
        self._buffer.acquire()
        self._views: list[Tensor] = []
        self._chunks: list[Tensor] = []
        for handle, place in zip(handles, chunk_places(gather, self._buffer.tensor), strict=True):
            sharded = handle.plan.sharded
            view = place[: sharded.numel()].view(sharded.shape)
            handle.unsharded.grad = view
            self._views.append(view)
            self._chunks.append(place.view(parts, -1))
        self._buffer.release()

        # The reduce-scatter writes the chunks of this rank, which lie in the same order.
        rank = gather.process_group.rank()
        shard = torch.zeros((sum(widths),), dtype=grad_dtype, device=self._buffer.tensor.device)
        local_views, offset = [], 0
        for handle, sizes, width in zip(handles, gather.shard_sizes, widths, strict=True):
            local_shape = local_tensor(handle.plan.sharded.data).shape
            local_views.append(shard[offset : offset + sizes[rank]].view(local_shape))
            offset += width

        self._gradients = BucketGradients(
            handles=handles,
            tensor=shard,
            local_views=local_views,
            replica_process_groups=plan.replica_process_groups,
        )

    @property
    def groups(self) -> tuple[int, ...]:
        """The groups that hold the parameters of the bucket, in ascending order."""
        return self._groups

    @property
    def gradients(self) -> BucketGradients:
        """The gradient shard of this rank."""
        return self._gradients

    def prepare(self):
        """Gives the gradient buffer zeroed memory for a backward pass.

        Raises:
            ValueError: If the current stream is not the stream that constructed the reduction.
        """
        buffer = self._buffer
        # Autograd writes the gradient buffer on the current stream. On another stream than its home, the waits for
        # the reduce stream would order the wrong stream.
        buffer.require_writer()
        # The previous backward pass of this bucket can still have the buffer pending, so it keeps its memory.
        if not self._pending.take_back(buffer):
            self._pending.make_room()
            buffer.acquire()
        # On the compute stream: on the reduce stream, the backward pass would wait for all queued reductions.
        buffer.tensor.zero_()

    @torch.no_grad()
    def reduce(self, final: bool):
        """Reduce-scatters the gradients of this backward pass, and sums them across replicas on the final pass.

        Args:
            final: Whether this is the last backward pass of the stage in the step.

        Raises:
            ValueError: If autograd replaced a gradient instead of accumulating it.
        """
        _check_accumulated(self._handles, self._views)
        stream = self._reduce_stream
        stream.wait_event(torch.cuda.current_stream().record_event())

        with torch.cuda.stream(stream):
            # The reduce-scatter needs the chunks of each rank next to each other. One collective per bucket, as in
            # the gather.
            source = torch.cat(self._chunks, dim=1).view(-1)
            # After this copy, the reduce stream reads only its own temporaries.
            self._pending.push(self._buffer, stream.record_event())
            self._reduce_scatter(source)
            if final:
                self._gradients.reduce_replicas()

    def _reduce_scatter(self, source: Tensor):
        shard = self._gradients.tensor
        # The first reduction of a step lands on zeros, so it needs neither a temporary nor an addition.
        output = torch.empty_like(shard) if self._shard_written else shard
        dist.reduce_scatter_single(output, source, op=dist.ReduceOp.SUM, group=self._process_group)
        if self._shard_written:
            shard.add_(output)
        self._shard_written = True

    def _free_buffer(self):
        if self._buffer.allocated:
            # A pending buffer waits for its reads. A backward pass that raised leaves a buffer that it never reduced.
            self._pending.take_back(self._buffer)
            self._buffer.release()

    def zero_grad(self):
        """Frees the gradient buffer and zeroes the gradient shard for a new step."""
        self._free_buffer()
        self._gradients.reset()
        self._shard_written = False

    def close(self):
        """Frees the gradient buffer and sets the gradients of the sharded parameters to ``None``."""
        self._free_buffer()
        self._gradients.clear()


class ReplicatedReduction:
    """The gradient buffer of trainable parameters without a gather, which autograd accumulates into directly."""

    def __init__(self, plan: BucketPlan, handles: Sequence[ParameterHandle], reduce_stream: torch.cuda.Stream):
        """Constructs the ``ReplicatedReduction`` object.

        It points the gradients of the sharded and unsharded parameters at the gradient buffer.

        Args:
            plan: The plan of the bucket.
            handles: The parameters, in buffer order.
            reduce_stream: The stream that reduces gradients.
        """
        self._groups = plan.groups
        self._handles = handles
        self._reduce_stream = reduce_stream

        device = handles[0].unsharded.device
        # The bucket is trainable, and plan_parameters() rejects a trainable parameter without grad_dtype.
        tensor = torch.zeros((plan.numel,), dtype=cast(torch.dtype, plan.grad_dtype), device=device)
        self._views: list[Tensor] = []
        offset = 0
        for handle in handles:
            size = handle.unsharded.numel()
            view = tensor[offset : offset + size].view(handle.unsharded.shape)
            handle.unsharded.grad = view
            self._views.append(view)
            offset += size

        self._gradients = BucketGradients(
            handles=handles,
            tensor=tensor,
            local_views=self._views,
            replica_process_groups=plan.replica_process_groups,
        )

    @property
    def groups(self) -> tuple[int, ...]:
        """The groups that hold the parameters of the bucket, in ascending order."""
        return self._groups

    @property
    def gradients(self) -> BucketGradients:
        """The full gradients of the parameters."""
        return self._gradients

    @torch.no_grad()
    def reduce(self, final: bool):
        """Sums the gradients across replicas after the final backward pass of the step.

        Args:
            final: Whether this is the last backward pass of the stage in the step.

        Raises:
            ValueError: If autograd replaced a gradient instead of accumulating it.
        """
        _check_accumulated(self._handles, self._views)
        if not final:
            return
        stream = self._reduce_stream
        stream.wait_event(torch.cuda.current_stream().record_event())
        with torch.cuda.stream(stream):
            self._gradients.reduce_replicas()

    def zero_grad(self):
        """Zeroes the gradients for a new step."""
        self._gradients.reset()

    def close(self):
        """Sets the gradients of the sharded parameters to ``None``."""
        self._gradients.clear()


AnyReduction = ShardedReduction | ReplicatedReduction
