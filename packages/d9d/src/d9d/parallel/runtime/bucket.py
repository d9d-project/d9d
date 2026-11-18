"""Buffers and collectives of the buckets: gathered buckets for sharded parameters, replicated buckets for the rest."""

import dataclasses
from collections import deque
from collections.abc import Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DTensor

from d9d.parallel.placement import dist_grad_from_local

from .parameter import ParameterHandle, local_tensor, new_unsharded, run_post_accumulate_hooks
from .plan import BucketPlan, GatherPlan


def _check_accumulated(handles: Sequence[ParameterHandle], grads: Sequence[Tensor]):
    for handle, grad in zip(handles, grads, strict=True):
        if handle.unsharded.grad is not grad:
            raise ValueError(
                f"The gradient of the parameter ({handle.plan.name}) was replaced during the backward pass, e.g. by "
                "backward(create_graph=True). Call backward() without create_graph=True."
            )


class ManagedBuffer:
    """A flat tensor whose ``Storage`` stays the same while its memory comes and goes.

    Saved autograd views point to the ``Storage``, so they stay valid. Replacing ``.data`` would break them.

    The memory comes from the allocator pool of the home stream, which can reuse it at once after ``release()``. So
    the home stream must wait for the reads of other streams before the release.
    """

    def __init__(self, size: int, dtype: torch.dtype, home: torch.cuda.Stream):
        """Constructs the ``ManagedBuffer`` object.

        The buffer holds no memory after construction.

        Args:
            size: The number of elements.
            dtype: The dtype.
            home: The stream that writes the buffer and owns its memory.
        """
        self._home = home
        self._bytes = size * dtype.itemsize
        self._allocated = False
        with torch.cuda.stream(home):
            self.tensor: Tensor = torch.empty(size, dtype=dtype, device="cuda")
        self.tensor.untyped_storage().resize_(0)

    @property
    def home(self) -> torch.cuda.Stream:
        """The stream that writes the buffer and owns its memory."""
        return self._home

    @property
    def allocated(self) -> bool:
        """Whether the buffer currently owns memory."""
        return self._allocated

    def acquire(self):
        """Gives the buffer uninitialized memory.

        Raises:
            ValueError: If the buffer already has memory.
        """
        if self._allocated:
            raise ValueError("The buffer already has memory. Call release() before acquire().")
        with torch.cuda.stream(self._home):
            self.tensor.untyped_storage().resize_(self._bytes)
        self._allocated = True

    def wait(self, read: torch.cuda.Event):
        """Orders the home stream after a read on another stream.

        Args:
            read: The event after the read.
        """
        self._home.wait_event(read)

    def release(self, after: torch.cuda.Event | None = None):
        """Gives the memory back to the allocator pool of the home stream.

        Args:
            after: The event after the last read on another stream, or ``None`` if only the home stream read it.

        Raises:
            ValueError: If the buffer has no memory.
        """
        if not self._allocated:
            raise ValueError("The buffer has no memory. Call acquire() before release().")
        if after is not None:
            self.wait(after)
        self.tensor.untyped_storage().resize_(0)
        self._allocated = False


# The reduce stream copies a gradient buffer behind the reduce-scatter of the previous bucket. So the copy of a buffer
# ends about one bucket later, and the compute stream waits for it only two buckets later.
_PENDING_GRADIENT_BUFFERS = 2


class PendingGradientBuffers:
    """The gradient buffers that the reduce stream may still read, oldest first.

    Autograd writes the gradient buffers on the compute stream, and the reduce stream reads them. A wait for these
    reads right after the reduction would stall the backward pass behind the reductions. So the compute stream frees
    a buffer only when later buckets allocate their own, and the reads of the oldest one have usually ended by then.
    """

    def __init__(self):
        """Constructs the ``PendingGradientBuffers`` object."""
        self._pending: deque[tuple[ManagedBuffer, torch.cuda.Event]] = deque()

    def push(self, buffer: ManagedBuffer, read: torch.cuda.Event):
        """Adds a buffer that the reduce stream reads.

        Args:
            buffer: The buffer.
            read: The event after the last read on the reduce stream.
        """
        self._pending.append((buffer, read))

    def take_back(self, buffer: ManagedBuffer) -> bool:
        """Takes a pending buffer back with its memory, after its reads.

        Args:
            buffer: The buffer.

        Returns:
            Whether the buffer was pending. Its memory then holds the old gradients.
        """
        for index, (pending, read) in enumerate(self._pending):
            if pending is buffer:
                del self._pending[index]
                buffer.wait(read)
                return True
        return False

    def make_room(self):
        """Frees the oldest buffers until fewer than the limit of pending buffers are pending."""
        while len(self._pending) >= _PENDING_GRADIENT_BUFFERS:
            buffer, read = self._pending.popleft()
            buffer.release(after=read)

    def free_all(self):
        """Frees every pending buffer."""
        while self._pending:
            buffer, read = self._pending.popleft()
            buffer.release(after=read)


class BucketGradients:
    """The gradients of a bucket on this rank, which the ``.grad`` of each sharded parameter views.

    ``scale_grads()`` and ``zero_grad()`` of the runtime change this tensor, so a gradient that is replaced by another
    tensor would escape them.

    Attributes:
        tensor: The flat gradients: the shard of this rank in a gathered bucket, the full gradients in a replicated
            one.
    """

    def __init__(
        self,
        handles: Sequence[ParameterHandle],
        tensor: Tensor,
        local_views: Sequence[Tensor],
        replica_process_groups: Sequence[ProcessGroup],
    ):
        """Constructs the ``BucketGradients`` object.

        It points the gradients of the sharded parameters at the flat gradients.

        Args:
            handles: The parameters of the bucket.
            tensor: The flat gradients.
            local_views: The view of ``tensor`` that holds the local gradient of each parameter.
            replica_process_groups: The process groups over which the gradients are summed, in reduction order.
        """
        self.tensor = tensor
        self._handles = handles
        self._replica_process_groups = replica_process_groups
        self._views = [
            dist_grad_from_local(h.plan.sharded.data, view) if isinstance(h.plan.sharded.data, DTensor) else view
            for h, view in zip(handles, local_views, strict=True)
        ]
        self.reset()

    def reset(self):
        """Clears the gradients and points every gradient at them again."""
        self.tensor.zero_()
        for handle, view in zip(self._handles, self._views, strict=True):
            handle.plan.sharded.grad = view

    def reduce_replicas(self):
        """Sums the gradients across replicas. The caller runs it on the reduce stream."""
        for process_group in self._replica_process_groups:
            dist.all_reduce(self.tensor, op=dist.ReduceOp.SUM, group=process_group)

    def check(self):
        """Checks that the gradient of every sharded parameter still views the bucket.

        Raises:
            ValueError: If a gradient was replaced during the step.
        """
        for handle, view in zip(self._handles, self._views, strict=True):
            if handle.plan.sharded.grad is not view:
                raise ValueError(
                    f"The gradient of the parameter ({handle.plan.name}) was replaced, so the runtime can neither "
                    "scale nor clear it. Change the gradient in place."
                )

    @torch.no_grad()
    def finish(self):
        """Runs the post-accumulate-grad hooks of the sharded parameters after the final reduction.

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


@dataclasses.dataclass(slots=True)
class _PendingGather:
    """A started gather of a bucket.

    Attributes:
        work: The running collective.
        gathered: The output of the collective: the chunks of all parameters of one rank, rank after rank.
        source: The input of the collective. It must stay alive until the collective ends.
    """

    work: dist.Work
    gathered: Tensor
    source: Tensor


@dataclasses.dataclass(slots=True)
class _GatheredGrads:
    """The gradients of a trainable gathered bucket.

    Attributes:
        buffer: The full gradient in the layout of the weights, which autograd writes.
        views: The gradient of each unsharded parameter in ``buffer``.
        chunks: The chunks of each parameter in ``buffer``, one chunk per rank.
        gradients: The gradient shard of this rank after the reduce-scatter.
    """

    buffer: ManagedBuffer
    views: list[Tensor]
    chunks: list[Tensor]
    gradients: BucketGradients


class GatheredBucket:
    """Buffers and collectives of parameters that the runtime gathers before the groups that use them.

    The runtime schedules each operation. The bucket owns its buffers, the packing of collectives and the stream
    dependencies.

    Attributes:
        plan: The parameters and process groups of the bucket.
        handles: The parameters, whose unsharded weights view the weight buffer of the bucket.
    """

    def __init__(
        self,
        plan: BucketPlan,
        gather: GatherPlan,
        gather_stream: torch.cuda.Stream,
        reduce_stream: torch.cuda.Stream,
        pending: PendingGradientBuffers,
    ):
        """Constructs the ``GatheredBucket`` object.

        It creates the unsharded parameters without memory for the weights and the gradient buffer. The gradient
        buffer belongs to the current stream, so the backward passes must run on it.

        Args:
            plan: The parameters and process groups of the bucket.
            gather: The gather of the bucket.
            gather_stream: The stream that gathers weights.
            reduce_stream: The stream that reduces gradients.
            pending: The gradient buffers of all buckets that the reduce stream may still read.
        """
        self.plan = plan
        self._gather = gather
        self._parts = gather.process_group.size()
        self._gather_stream = gather_stream
        self._reduce_stream = reduce_stream
        self._pending = pending
        self._weights_ready = False
        # Whether a reduce-scatter of this step already wrote the gradient shard.
        self._shard_written = False
        self._gathering: _PendingGather | None = None

        self._weights = ManagedBuffer(plan.numel, plan.dtype, gather_stream)
        self.handles, self._weight_chunks = self._place_parameters()
        self._grads = self._bind_grads() if plan.trainable else None

    @property
    def gradients(self) -> BucketGradients | None:
        """The gradient shard, or ``None`` for a frozen bucket."""
        return None if self._grads is None else self._grads.gradients

    def _places(self, flat: Tensor) -> list[Tensor]:
        # A parameter takes one padded chunk per rank, and the chunks of one rank lie next to each other in the
        # collective. One copy per bucket converts between this layout and the layout of the parameters.
        places, offset = [], 0
        for sizes in self._gather.shard_sizes:
            width = sizes[0]
            places.append(flat[self._parts * offset : self._parts * (offset + width)])
            offset += width
        return places

    def _place_parameters(self) -> tuple[list[ParameterHandle], list[Tensor]]:
        weights = self._weights
        # The views need memory to be created. The parameters view one Storage, so one resize frees all of them.
        weights.acquire()
        handles, chunks = [], []
        for parameter, place in zip(self.plan.parameters, self._places(weights.tensor), strict=True):
            unsharded = new_unsharded(parameter, place[: parameter.sharded.numel()].view(parameter.sharded.shape))
            handles.append(ParameterHandle(plan=parameter, unsharded=unsharded))
            # Views of .data, because a write to the weights bumps the version that saved graphs check.
            chunks.append(place.data.view(self._parts, -1))
        weights.release()
        return handles, chunks

    def _bind_grads(self) -> _GatheredGrads:
        widths = [sizes[0] for sizes in self._gather.shard_sizes]
        buffer = ManagedBuffer(self._parts * sum(widths), self.plan.grad_dtype, torch.cuda.current_stream())
        buffer.acquire()
        views, chunks = [], []
        for handle, place in zip(self.handles, self._places(buffer.tensor), strict=True):
            sharded = handle.plan.sharded
            view = place[: sharded.numel()].view(sharded.shape)
            handle.unsharded.grad = view
            views.append(view)
            chunks.append(place.view(self._parts, -1))
        buffer.release()

        # The reduce-scatter writes the chunks of this rank, which lie in the same order.
        rank = self._gather.process_group.rank()
        shard = torch.zeros((sum(widths),), dtype=self.plan.grad_dtype, device=buffer.tensor.device)
        local_views, offset = [], 0
        for handle, sizes, width in zip(self.handles, self._gather.shard_sizes, widths, strict=True):
            local_shape = local_tensor(handle.plan.sharded.data).shape
            local_views.append(shard[offset : offset + sizes[rank]].view(local_shape))
            offset += width

        gradients = BucketGradients(
            handles=self.handles,
            tensor=shard,
            local_views=local_views,
            replica_process_groups=self.plan.replica_process_groups,
        )
        return _GatheredGrads(buffer=buffer, views=views, chunks=chunks, gradients=gradients)

    def _trainable_grads(self) -> _GatheredGrads:
        # The stage schedules gradient work only for trainable buckets.
        return cast(_GatheredGrads, self._grads)

    @torch.no_grad()
    def start_gather(self, weights_written: torch.cuda.Event):
        """Starts gathering the shards of the bucket into its unsharded weights.

        Args:
            weights_written: The event after the latest writes to the sharded weights.
        """
        if self._gathering is not None or self._weights_ready:
            return

        self._weights.acquire()

        # The optimizer writes the sharded weights on the compute stream, and the gather reads them.
        self._gather_stream.wait_event(weights_written)
        with torch.cuda.stream(self._gather_stream):
            pieces = []
            for handle, chunks in zip(self.handles, self._weight_chunks, strict=True):
                shard = local_tensor(handle.plan.sharded.data).reshape(-1)
                pieces.append(shard)
                if shard.numel() < chunks.shape[1]:
                    pieces.append(shard.new_zeros(chunks.shape[1] - shard.numel()))

            # One collective per bucket: many small collectives cost more launches and reach less bandwidth than one
            # large one.
            source = torch.cat(pieces)
            gathered = source.new_empty(source.numel() * self._parts)
            work = dist.all_gather_single(gathered, source, group=self._gather.process_group, async_op=True)

        # PyTorch types the result as optional, but async_op=True always returns the work handle.
        self._gathering = _PendingGather(cast(dist.Work, work), gathered, source)

    @torch.no_grad()
    def finish_gather(self):
        """Makes the unsharded weights of the bucket ready for the current stream."""
        if self._gathering is None:
            return
        gathering = self._gathering
        self._gathering = None

        stream = self._gather_stream
        with torch.cuda.stream(stream):
            # Inside the stream: a collective orders only the stream that waits for it.
            gathering.work.wait()
            widths = [chunks.shape[1] for chunks in self._weight_chunks]
            torch.split_with_sizes_copy(
                gathering.gathered.view(self._parts, -1), widths, dim=1, out=self._weight_chunks
            )
            ready = stream.record_event()

        torch.cuda.current_stream().wait_event(ready)
        # The weights stay valid until the runtime releases them, so a later start_gather() does nothing.
        self._weights_ready = True

    def free_weights(self):
        """Frees the unsharded weights after the reads of the current stream.

        The gather stream waits for these reads, so the gathers stay at most one group ahead of the current stream.
        """
        self._weights_ready = False
        self._weights.release(after=torch.cuda.current_stream().record_event())

    def prepare_grads(self):
        """Allocates cleared gradients for the backward pass.

        Raises:
            ValueError: If the current stream is not the stream that bound the bucket.
        """
        buffer = self._trainable_grads().buffer
        # The gradient buffers come from the pool of the stream that bound the bucket. On another stream, the waits
        # for the reduce stream would order the wrong stream.
        if torch.cuda.current_stream() != buffer.home:
            raise ValueError(
                f"The backward pass runs on the stream ({torch.cuda.current_stream()}), but bind() ran on the stream "
                f"({buffer.home}). Run bind() and the steps on one stream."
            )
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
        grads = self._trainable_grads()
        _check_accumulated(self.handles, grads.views)
        stream = self._reduce_stream
        stream.wait_event(torch.cuda.current_stream().record_event())

        with torch.cuda.stream(stream):
            # The reduce-scatter needs the chunks of each rank next to each other. One collective per bucket, as in
            # the gather.
            source = torch.cat(grads.chunks, dim=1).view(-1)
            # After this copy, the reduce stream reads only its own temporaries.
            self._pending.push(grads.buffer, stream.record_event())
            self._reduce_scatter(grads.gradients.tensor, source)
            if final:
                grads.gradients.reduce_replicas()

    def _reduce_scatter(self, shard: Tensor, source: Tensor):
        # The first reduction of a step lands on zeros, so it needs neither a temporary nor an addition.
        output = torch.empty_like(shard) if self._shard_written else shard
        dist.reduce_scatter_single(output, source, op=dist.ReduceOp.SUM, group=self._gather.process_group)
        if self._shard_written:
            shard.add_(output)
        self._shard_written = True

    def _free_grad_buffer(self):
        buffer = self._trainable_grads().buffer
        if buffer.allocated:
            # A pending buffer waits for its reads. A backward pass that raised leaves a buffer that it never reduced.
            self._pending.take_back(buffer)
            buffer.release()

    def zero_grad(self):
        """Clears the gradients."""
        if self._grads is None:
            return
        self._free_grad_buffer()
        self._grads.gradients.reset()
        self._shard_written = False

    def invalidate_weights(self):
        """Drops the unsharded weights, including frozen and inference weights."""
        self.finish_gather()
        if self._weights.allocated:
            self.free_weights()

    def unbind(self):
        """Frees the buffers after their reads and sets the gradients of the sharded parameters to ``None``."""
        self.invalidate_weights()
        if self._grads is not None:
            self._free_grad_buffer()
            self._grads.gradients.clear()


class ReplicatedBucket:
    """The gradient buffer of trainable parameters without a gather, which autograd accumulates into directly.

    Attributes:
        plan: The parameters and process groups of the bucket.
        gradients: The full gradients of the parameters.
    """

    def __init__(self, plan: BucketPlan, handles: Sequence[ParameterHandle], reduce_stream: torch.cuda.Stream):
        """Constructs the ``ReplicatedBucket`` object.

        It points the gradients of its parameters at its buffer.

        Args:
            plan: The parameters and process groups of the bucket.
            handles: The parameters of the bucket, in buffer order.
            reduce_stream: The stream that reduces gradients.
        """
        self.plan = plan
        self._handles = handles
        self._reduce_stream = reduce_stream

        device = handles[0].unsharded.device
        tensor = torch.zeros((plan.numel,), dtype=plan.grad_dtype, device=device)
        self._views: list[Tensor] = []
        offset = 0
        for handle in handles:
            size = handle.unsharded.numel()
            view = tensor[offset : offset + size].view(handle.unsharded.shape)
            handle.unsharded.grad = view
            self._views.append(view)
            offset += size

        self.gradients = BucketGradients(
            handles=handles,
            tensor=tensor,
            local_views=self._views,
            replica_process_groups=plan.replica_process_groups,
        )

    @torch.no_grad()
    def reduce(self, final: bool):
        """Sums the gradients across replicas. Only the final backward pass of a step has work.

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
            self.gradients.reduce_replicas()

    def zero_grad(self):
        """Clears the gradients."""
        self.gradients.reset()

    def unbind(self):
        """Sets the gradients of the sharded parameters to ``None``."""
        self.gradients.clear()


AnyBucket = GatheredBucket | ReplicatedBucket
