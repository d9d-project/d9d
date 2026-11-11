"""Buffers and collectives of the buckets: gathered buckets for sharded parameters, replicated buckets for the rest."""

import dataclasses
from collections.abc import Sequence
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from .memory import ManagedBuffer, MemoryLedger
from .parameter import ParameterHandle, grad_like, local_tensor, new_unsharded, run_post_accumulate_hooks
from .plan import BucketPlan


def _name(handle: ParameterHandle) -> str:
    module, name = handle.plan.owners[0]
    return f"{type(module).__name__}.{name}"


class GradientShard:
    """The gradient shard of a bucket on this rank, which the ``.grad`` of each sharded parameter views.

    ``scale_grads()`` and ``zero_grad()`` of the runtime change the shard, so a gradient that is replaced by another
    tensor would escape them. The shard also finds out which parameters any rank used in the step.

    Attributes:
        tensor: The flat gradient shard.
    """

    def __init__(
        self,
        handles: Sequence[ParameterHandle],
        tensor: Tensor,
        local_views: Sequence[Tensor],
        replica_groups: Sequence[ProcessGroup],
        usage_groups: Sequence[ProcessGroup],
    ):
        """Constructs the ``GradientShard`` object and points the gradients of the sharded parameters at it.

        Args:
            handles: The parameters of the bucket.
            tensor: The flat gradient shard.
            local_views: The view of ``tensor`` that holds the local gradient of each parameter.
            replica_groups: The process groups over which the shard is summed, in reduction order.
            usage_groups: The process groups of every mesh dimension of the parameters.
        """
        self.tensor = tensor
        self._handles = handles
        self._replica_groups = replica_groups
        self._usage_groups = usage_groups
        self._views = [grad_like(h.plan.sharded.data, view) for h, view in zip(handles, local_views, strict=True)]
        self._usage: Tensor | None = None
        self.reset()

    def reset(self):
        """Clears the shard and points every gradient at it again, also those that the previous step left ``None``."""
        self.tensor.zero_()
        for handle, view in zip(self._handles, self._views, strict=True):
            handle.plan.sharded.grad = view
            handle.used = False
        self._usage = None

    def reduce_replicas(self):
        """Sums the shard across replicas and finds out which parameters any rank used. Runs on the reduce stream."""
        for process_group in self._replica_groups:
            dist.all_reduce(self.tensor, op=dist.ReduceOp.SUM, group=process_group)

        # A non-blocking copy from pinned memory, so the host does not wait for the reduce stream.
        local = torch.tensor([handle.used for handle in self._handles], dtype=torch.uint8).pin_memory()
        usage = local.to(self.tensor.device, non_blocking=True)
        for process_group in self._usage_groups:
            dist.all_reduce(usage, op=dist.ReduceOp.MAX, group=process_group)
        self._usage = usage

    @property
    def usage(self) -> Tensor:
        """Whether any rank used each parameter in this step, in the order of the handles.

        It is valid on the reduce stream after ``reduce_replicas()``.

        Raises:
            ValueError: If the shard has not reduced its usage flags.
        """
        if self._usage is None:
            raise ValueError("The gradient shard has no usage. Reduce the final backward pass first.")
        return self._usage

    def check(self):
        """Checks that the gradient of every sharded parameter still views the shard.

        Raises:
            ValueError: If a gradient was replaced during the step.
        """
        for handle, view in zip(self._handles, self._views, strict=True):
            if handle.plan.sharded.grad is not view:
                raise ValueError(
                    f"The gradient of ({_name(handle)}) was replaced, so the runtime can neither scale nor clear it. "
                    "Change the gradient in place."
                )

    @torch.no_grad()
    def finish(self, used: Sequence[bool]):
        """Finishes the gradients of the step after the final reduction.

        A parameter that no rank used gets no gradient, as without the runtime. The others run their
        post-accumulate-grad hooks.

        Args:
            used: Whether any rank used each parameter, in the order of the handles.

        Raises:
            ValueError: If a hook set a gradient to ``None``.
        """
        for handle, view, is_used in zip(self._handles, self._views, used, strict=True):
            sharded = handle.plan.sharded
            if not is_used:
                sharded.grad = None
                continue

            run_post_accumulate_hooks(handle)

            # The hooks can replace the gradient, as PyTorch allows, so its value goes back into the shard.
            current = sharded.grad
            if current is view:
                continue
            if current is None:
                raise ValueError(
                    f"A hook set the gradient of ({_name(handle)}) to None. Zero it in place, e.g. with grad.zero_()."
                )
            view.copy_(current)
            sharded.grad = view

    def detach(self):
        """Removes the gradients from the sharded parameters."""
        for handle in self._handles:
            handle.plan.sharded.grad = None


@dataclasses.dataclass(slots=True)
class _PendingGather:
    """A started gather of a bucket.

    Attributes:
        work: The running collective.
        gathered: The output of the collective: the shards of all parameters of one rank, rank after rank.
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
        rows: The rows of each parameter in ``buffer``, one row per rank.
        shard: The gradient shard of this rank after the reduce-scatter.
    """

    buffer: ManagedBuffer
    rows: list[Tensor]
    shard: GradientShard


class GatheredBucket:
    """Buffers and collectives of parameters that the runtime gathers before the groups that use them.

    The runtime schedules each operation. The bucket owns its buffers, the packing of collectives and the stream
    dependencies.

    Attributes:
        plan: The parameters and process groups of the bucket.
        handles: The parameters, whose unsharded weights view the weight buffer of the bucket.
        gradients: The gradient shard, or ``None`` for a frozen bucket.
        keep_weights: Whether the unsharded weights stay gathered between groups. ``zero_grad()`` still frees
            trainable weights, because the optimizer changes their shards.
    """

    def __init__(
        self,
        plan: BucketPlan,
        gather_group: ProcessGroup,
        gather_stream: torch.cuda.Stream,
        reduce_stream: torch.cuda.Stream,
        ledger: MemoryLedger,
    ):
        """Constructs the ``GatheredBucket`` object and its unsharded parameters, without holding memory.

        Args:
            plan: The parameters and process groups of the bucket.
            gather_group: The process group of the gather axis.
            gather_stream: The stream that gathers weights.
            reduce_stream: The stream that reduces gradients.
            ledger: The shared memory budget.
        """
        self.plan = plan
        self.keep_weights = False
        self._group = gather_group
        self._parts = gather_group.size()
        self._gather_stream = gather_stream
        self._reduce_stream = reduce_stream
        self._ledger = ledger
        self._weights_ready = False
        # Whether a reduce-scatter of this step already wrote the gradient shard.
        self._shard_written = False
        self._gathering: _PendingGather | None = None

        self._weights = ManagedBuffer(plan.numel, plan.compute_dtype, gather_stream, ledger)
        self.handles, self._weight_rows = self._place_parameters(self._weights)

        self._grads = self._bind_grads() if plan.trainable else None
        self.gradients = None if self._grads is None else self._grads.shard

    def _place_parameters(self, weights: ManagedBuffer) -> tuple[list[ParameterHandle], list[Tensor]]:
        # The parameters view one Storage, so one resize frees all of them. The views need memory to be created.
        weights.tensor.untyped_storage().resize_(weights.tensor.nbytes)

        # The rows of one rank lie next to each other in the collective, but a parameter needs its own rows in
        # sequence. One copy per bucket converts between the layouts, as in FSDP2.
        handles, rows = [], []
        rank_offset = 0
        for parameter, sizes in zip(self.plan.parameters, self.plan.shard_sizes, strict=True):
            width = sizes[0]
            place = weights.tensor[self._parts * rank_offset : self._parts * (rank_offset + width)]
            unsharded = new_unsharded(parameter, place[: parameter.sharded.numel()].view(parameter.sharded.shape))
            handles.append(ParameterHandle(plan=parameter, unsharded=unsharded, has_own_copy=False))
            rows.append(place.data.view(self._parts, width))
            rank_offset += width

        weights.tensor.untyped_storage().resize_(0)
        return handles, rows

    def _bind_grads(self) -> _GatheredGrads:
        rank = self._group.rank()
        widths = [sizes[0] for sizes in self.plan.shard_sizes]
        buffer = ManagedBuffer(self._parts * sum(widths), self.plan.grad_dtype, self._reduce_stream, self._ledger)
        grads = buffer.tensor
        grads.untyped_storage().resize_(grads.nbytes)

        # The reduction writes the rows of this rank, which lie in the same order.
        shard = torch.zeros((sum(widths),), dtype=self.plan.grad_dtype, device=grads.device)
        rows, local_views = [], []
        rank_offset = 0
        for handle, sizes, width in zip(self.handles, self.plan.shard_sizes, widths, strict=True):
            sharded = handle.plan.sharded
            place = grads[self._parts * rank_offset : self._parts * (rank_offset + width)]
            handle.unsharded.grad = place[: sharded.numel()].view(sharded.shape)
            rows.append(place.view(self._parts, width))

            local_shape = local_tensor(sharded.data).shape
            local_views.append(shard[rank_offset : rank_offset + sizes[rank]].view(local_shape))
            rank_offset += width

        grads.untyped_storage().resize_(0)
        gradient_shard = GradientShard(
            handles=self.handles,
            tensor=shard,
            local_views=local_views,
            replica_groups=self.plan.replica_groups,
            usage_groups=self.plan.usage_groups,
        )
        return _GatheredGrads(buffer=buffer, rows=rows, shard=gradient_shard)

    def _require_grads(self) -> _GatheredGrads:
        if self._grads is None:
            raise ValueError("The bucket is frozen, so it has no gradients.")
        return self._grads

    @torch.no_grad()
    def start_unshard(self, storage_written: torch.cuda.Event):
        """Starts gathering the shards of the bucket into its unsharded weights.

        Args:
            storage_written: The event after the latest writes to the sharded weights.
        """
        if self._gathering is not None or self._weights_ready:
            return

        self._weights.acquire()
        dtype = self.plan.compute_dtype
        # Both packed shards and the gathered output stay live until the collective finishes.
        temporary_bytes = (self.plan.numel + self.plan.numel // self._parts) * dtype.itemsize
        self._ledger.reserve(temporary_bytes)

        # The optimizer writes the sharded weights on the compute stream, and the gather reads them.
        self._gather_stream.wait_event(storage_written)
        with torch.cuda.stream(self._gather_stream):
            pieces = []
            for handle, rows in zip(self.handles, self._weight_rows, strict=True):
                shard = local_tensor(handle.plan.sharded.data).to(dtype).reshape(-1)
                pieces.append(shard)
                if shard.numel() < rows.shape[1]:
                    pieces.append(shard.new_zeros(rows.shape[1] - shard.numel()))

            # One collective per bucket: many small collectives cost more launches and reach less bandwidth than one
            # large one.
            source = torch.cat(pieces)
            gathered = torch.empty(source.numel() * self._parts, dtype=dtype, device=source.device)
            work = dist.all_gather_single(gathered, source, group=self._group, async_op=True)

        # PyTorch types the result as optional, but async_op=True always returns the work handle.
        self._gathering = _PendingGather(cast(dist.Work, work), gathered, source)

    @torch.no_grad()
    def finish_unshard(self):
        """Makes the unsharded weights of the bucket ready for the current stream."""
        if self._gathering is None:
            return
        gathering = self._gathering
        self._gathering = None

        stream = self._gather_stream
        with torch.cuda.stream(stream):
            # Inside the stream: a collective orders only the stream that waits for it.
            gathering.work.wait()
            widths = [rows.shape[1] for rows in self._weight_rows]
            # The rows are views of .data, because a write to the weights bumps the version that saved graphs check.
            torch.split_with_sizes_copy(gathering.gathered.view(self._parts, -1), widths, dim=1, out=self._weight_rows)
            ready = stream.record_event()

        self._ledger.retire([ready], gathering.gathered.nbytes + gathering.source.nbytes)
        torch.cuda.current_stream().wait_event(ready)
        # The weights stay valid until the runtime releases them, so a later start_unshard() does nothing.
        self._weights_ready = True

    def release_weights(self, compute_stream: torch.cuda.Stream):
        """Frees the unsharded weights after their queued reads.

        Args:
            compute_stream: The stream that last read the weights.
        """
        self._weights_ready = False
        self._weights.release([compute_stream])

    def prepare_grads(self):
        """Allocates cleared gradients and orders compute after their initialization.

        Raises:
            ValueError: If the bucket is frozen.
        """
        buffer = self._require_grads().buffer
        buffer.acquire()
        stream = self._reduce_stream
        with torch.cuda.stream(stream):
            buffer.tensor.zero_()
            ready = stream.record_event()
        torch.cuda.current_stream().wait_event(ready)

    @torch.no_grad()
    def reduce(self, final: bool):
        """Reduce-scatters the gradients of this backward pass, and sums them across replicas on the final pass.

        Args:
            final: Whether this is the last backward pass of the stage in the step.

        Raises:
            ValueError: If the bucket is frozen.
        """
        grads = self._require_grads()
        stream = self._reduce_stream
        compute_stream = torch.cuda.current_stream()
        stream.wait_event(compute_stream.record_event())

        with torch.cuda.stream(stream):
            self._reduce_scatter(grads)
            if final:
                grads.shard.reduce_replicas()

        grads.buffer.release([compute_stream, stream])

    def _reduce_scatter(self, grads: _GatheredGrads):
        shard = grads.shard.tensor
        size = grads.buffer.tensor.nbytes + (shard.nbytes if self._shard_written else 0)
        self._ledger.reserve(size)

        # The rows of each rank next to each other, the layout of the collective. One collective per bucket, as in
        # the gather.
        source = torch.cat(grads.rows, dim=1).view(-1)
        # The first reduction of a step lands on zeros, so it needs neither a temporary nor an addition.
        output = torch.empty_like(shard) if self._shard_written else shard
        dist.reduce_scatter_single(output, source, op=dist.ReduceOp.SUM, group=self._group)
        if self._shard_written:
            shard.add_(output)
        self._shard_written = True

        self._ledger.retire([torch.cuda.current_stream().record_event()], size)

    def zero_grad(self):
        """Clears the gradients and drops the weights that the optimizer changed."""
        compute_stream = torch.cuda.current_stream()
        if self._grads is None:
            if not self.keep_weights:
                self.invalidate_weights(compute_stream)
            return
        # A backward pass that raised leaves a gradient buffer that it never reduced.
        if self._grads.buffer.allocated:
            self._grads.buffer.release([compute_stream])
        self._grads.shard.reset()
        self._shard_written = False
        # A prefetch that crossed the optimizer step holds the weights from before it.
        self.invalidate_weights(compute_stream)

    def invalidate_weights(self, compute_stream: torch.cuda.Stream):
        """Drops the unsharded weights, including frozen and inference weights.

        Args:
            compute_stream: The stream that last read the weights.
        """
        self.finish_unshard()
        if self._weights.allocated:
            self.release_weights(compute_stream)
        self._weights_ready = False

    def unbind(self):
        """Releases the buffers after their final reads and detaches the gradients of the sharded parameters."""
        # Kept buffers can still be read on the compute stream when the runtime is torn down.
        self.finish_unshard()
        buffers = [self._weights] if self._grads is None else [self._weights, self._grads.buffer]
        for buffer in buffers:
            if buffer.allocated:
                buffer.release([torch.cuda.current_stream()])
        self._weights_ready = False

        if self.gradients is not None:
            self.gradients.detach()


class ReplicatedBucket:
    """The gradient buffer of trainable parameters without a gather, which autograd accumulates into directly.

    Attributes:
        plan: The parameters and process groups of the bucket.
        gradients: The gradient shard, which holds the full gradient of each parameter.
    """

    def __init__(self, plan: BucketPlan, handles: Sequence[ParameterHandle], reduce_stream: torch.cuda.Stream):
        """Constructs the ``ReplicatedBucket`` object and points the gradients of its parameters at its buffer.

        Args:
            plan: The parameters and process groups of the bucket.
            handles: The parameters of the bucket, in buffer order.
            reduce_stream: The stream that reduces gradients.
        """
        self.plan = plan
        self._reduce_stream = reduce_stream

        device = handles[0].unsharded.device
        tensor = torch.zeros((plan.numel,), dtype=plan.grad_dtype, device=device)
        views, offset = [], 0
        for handle in handles:
            size = handle.unsharded.numel()
            view = tensor[offset : offset + size].view(handle.unsharded.shape)
            handle.unsharded.grad = view
            views.append(view)
            offset += size

        self.gradients = GradientShard(
            handles=handles,
            tensor=tensor,
            local_views=views,
            replica_groups=plan.replica_groups,
            usage_groups=plan.usage_groups,
        )

    @torch.no_grad()
    def reduce(self, final: bool):
        """Sums the gradients across replicas. Only the final backward pass of a step has work.

        Args:
            final: Whether this is the last backward pass of the stage in the step.
        """
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
        """Detaches the gradients of the sharded parameters."""
        self.gradients.detach()


AnyBucket = GatheredBucket | ReplicatedBucket
