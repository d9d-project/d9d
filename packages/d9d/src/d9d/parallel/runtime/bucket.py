"""Buffers and collectives of one bucket."""

import dataclasses
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed.tensor import DTensor

from d9d.parallel.memory import ManagedBuffer, MemoryLedger

from .parameter import ParameterHandle, grad_like, local_tensor, run_post_accumulate_hooks
from .plan import BucketPlan, shard_sizes


def _name(handle: ParameterHandle) -> str:
    module, name = handle.owners[0]
    return f"{type(module).__name__}.{name}"


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


class Bucket:
    """Buffers and collectives of parameters that share one communication layout.

    The runtime schedules each operation. The bucket owns its buffers, the packing of collectives and the stream
    dependencies.
    """

    def __init__(
        self,
        plan: BucketPlan,
        gather_stream: torch.cuda.Stream,
        reduce_stream: torch.cuda.Stream,
        ledger: MemoryLedger,
    ):
        """Constructs the ``Bucket`` object without allocating its buffers.

        Args:
            plan: The parameters and process groups of the bucket.
            gather_stream: The stream that gathers weights.
            reduce_stream: The stream that reduces gradients.
            ledger: The shared memory budget.
        """
        self.plan = plan
        self._gather_stream = gather_stream
        self._reduce_stream = reduce_stream
        self._ledger = ledger
        self.weights: ManagedBuffer | None = None
        self.grads: ManagedBuffer | None = None
        self.grad_shard: Tensor | None = None
        # The weights are freed after each segment that uses them, unless the bucket is pinned or the next segment
        # uses it too. zero_grad() frees pinned trainable weights, because the optimizer changes their shards.
        self.pinned = False
        self.weights_ready = False
        self.reduced = False
        self._gathering: _PendingGather | None = None
        self._weight_rows: list[Tensor] = []
        self._grad_rows: list[Tensor] = []
        # The gradient of each sharded parameter views the gradient shard. scale_grads() and zero_grad() change the
        # shard, so a gradient that is replaced by another tensor would escape them.
        self._sharded_grads: list[tuple[ParameterHandle, Tensor]] = []

    def bind(self):
        """Allocates the gradient shard and connects the parameters to their buffers."""
        if self.plan.gathers:
            self._bind_gathered()
        else:
            self._bind_plain()

    def _bind_plain(self):
        device = self.plan.handles[0].unsharded.device
        self.grad_shard = torch.zeros((self.plan.numel,), dtype=self.plan.grad_dtype, device=device)
        offset = 0
        for handle in self.plan.handles:
            size = handle.unsharded.numel()
            view = self.grad_shard[offset : offset + size].view(handle.unsharded.shape)
            handle.unsharded.grad = view
            data = handle.sharded.data
            grad = grad_like(data, view) if isinstance(data, DTensor) else view
            handle.sharded.grad = grad
            self._sharded_grads.append((handle, grad))
            offset += size

    def _bind_gathered(self):
        gather_group = cast(dist.ProcessGroup, self.plan.gather_group)
        parts = gather_group.size()
        dtype = self.plan.handles[0].unsharded.dtype

        self.weights = ManagedBuffer(self.plan.numel, dtype, self._gather_stream, self._ledger)
        # The parameters view one Storage, so one resize frees all of them. The views need memory to be created.
        self.weights.tensor.untyped_storage().resize_(self.plan.numel * dtype.itemsize)
        # The rows of one rank lie next to each other in the collective, but a parameter needs its own rows in
        # sequence. One copy per bucket converts between the layouts, as in FSDP2.
        rank_offset = 0
        for handle in self.plan.handles:
            sharded = handle.sharded
            handle.has_own_copy = False
            handle.shard_sizes = shard_sizes(sharded, parts)
            if local_tensor(sharded.data).numel() != handle.shard_sizes[gather_group.rank()]:
                raise ValueError(
                    f"The shard of ({handle.owners[0][1]}) does not match torch.chunk. "
                    "Distribute the parameter with parallelize_fsdp() or parallelize_hsdp()."
                )
            width = handle.shard_sizes[0]
            place = self.weights.tensor[parts * rank_offset : parts * (rank_offset + width)]
            grad_dtype = handle.unsharded.grad_dtype
            handle.unsharded = nn.Parameter(place[: sharded.numel()].view(sharded.shape), sharded.requires_grad)
            handle.unsharded.grad_dtype = grad_dtype
            self._weight_rows.append(place.data.view(parts, width))
            rank_offset += width
        self.weights.tensor.untyped_storage().resize_(0)

        if self.plan.trainable:
            self._bind_gathered_grads(parts)

    def _bind_gathered_grads(self, parts: int):
        rank = cast(dist.ProcessGroup, self.plan.gather_group).rank()
        widths = [handle.shard_sizes[0] for handle in self.plan.handles]
        self.grads = ManagedBuffer(parts * sum(widths), self.plan.grad_dtype, self._reduce_stream, self._ledger)
        grads = self.grads.tensor
        grads.untyped_storage().resize_(grads.numel() * self.plan.grad_dtype.itemsize)
        # The reduction writes the rows of this rank, which lie in the same order.
        self.grad_shard = torch.zeros((sum(widths),), dtype=self.plan.grad_dtype, device=grads.device)
        rank_offset = 0
        for handle, width in zip(self.plan.handles, widths, strict=True):
            sharded = handle.sharded
            place = grads[parts * rank_offset : parts * (rank_offset + width)]
            handle.unsharded.grad = place[: sharded.numel()].view(sharded.shape)
            self._grad_rows.append(place.view(parts, width))
            data = cast(DTensor, sharded.data)
            shard = self.grad_shard[rank_offset : rank_offset + handle.shard_sizes[rank]]
            grad = grad_like(data, shard.view(data.to_local().shape))
            sharded.grad = grad
            self._sharded_grads.append((handle, grad))
            rank_offset += width
        grads.untyped_storage().resize_(0)

    @torch.no_grad()
    def start_unshard(self, storage_written: torch.cuda.Event):
        """Starts gathering the shards of the bucket into its unsharded weights.

        Args:
            storage_written: The event after the latest writes to the sharded weights.
        """
        if self._gathering is not None or self.weights_ready:
            return

        weights = cast(ManagedBuffer, self.weights)
        gather_group = cast(dist.ProcessGroup, self.plan.gather_group)
        weights.acquire()
        dtype = weights.tensor.dtype
        # Both packed shards and the gathered output stay live until the collective finishes.
        temporary_bytes = (self.plan.numel + self.plan.numel // gather_group.size()) * dtype.itemsize
        self._ledger.reserve(temporary_bytes)
        # The optimizer writes the sharded weights on the compute stream, and the gather reads them.
        self._gather_stream.wait_event(storage_written)
        with torch.cuda.stream(self._gather_stream):
            pieces = []
            for handle, rows in zip(self.plan.handles, self._weight_rows, strict=True):
                shard = local_tensor(handle.sharded.data).to(dtype).reshape(-1)
                pieces.append(shard)
                if shard.numel() < rows.shape[1]:
                    pieces.append(shard.new_zeros(rows.shape[1] - shard.numel()))
            # One collective per bucket: a collective per parameter is small enough that NCCL picks its
            # low-latency protocol, which loses most of the bandwidth between nodes.
            source = torch.cat(pieces)
            gathered = torch.empty(source.numel() * gather_group.size(), dtype=dtype, device=source.device)
            work = dist.all_gather_single(gathered, source, group=gather_group, async_op=True)
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
            parts = cast(dist.ProcessGroup, self.plan.gather_group).size()
            widths = [rows.shape[1] for rows in self._weight_rows]
            # The rows are views of .data, because a write to the weights bumps the version that saved graphs check.
            torch.split_with_sizes_copy(gathering.gathered.view(parts, -1), widths, dim=1, out=self._weight_rows)
            ready = stream.record_event()
        self._ledger.retire([ready], gathering.gathered.nbytes + gathering.source.nbytes)
        torch.cuda.current_stream().wait_event(ready)
        # The weights stay valid until the runtime releases them, so a later start_unshard() does nothing.
        self.weights_ready = True

    def release_weights(self, compute_stream: torch.cuda.Stream):
        """Frees the unsharded weights after their queued reads.

        Args:
            compute_stream: The stream that last read the weights.
        """
        self.weights_ready = False
        cast(ManagedBuffer, self.weights).release([compute_stream])

    def prepare_grads(self):
        """Allocates cleared gradients and orders compute after their initialization."""
        grads = cast(ManagedBuffer, self.grads)
        stream = self._reduce_stream
        grads.acquire()
        with torch.cuda.stream(stream):
            grads.tensor.zero_()
            ready = stream.record_event()
        torch.cuda.current_stream().wait_event(ready)

    @torch.no_grad()
    def reduce(self, final: bool):
        """Reduces the gradients of this backward pass, and finishes the gradients on the final pass.

        The final pass sums the gradient shards across replicas and runs the post-accumulate-grad hooks of the
        sharded parameters.

        Args:
            final: Whether this is the last backward pass of the stage in the step.
        """
        stream = self._reduce_stream
        compute_stream = torch.cuda.current_stream()
        stream.wait_event(compute_stream.record_event())
        with torch.cuda.stream(stream):
            if self.grads is not None:
                self._reduce_scatter()
            if final:
                for process_group in self.plan.replica_groups:
                    dist.all_reduce(cast(Tensor, self.grad_shard), op=dist.ReduceOp.SUM, group=process_group)
                for handle in self.plan.handles:
                    run_post_accumulate_hooks(handle)
                # The hooks can replace the gradient, as PyTorch allows, so its value goes back into the shard.
                self._restore_grads()
        if self.grads is not None:
            self.grads.release([compute_stream, stream])

    @torch.no_grad()
    def _restore_grads(self):
        for handle, expected in self._sharded_grads:
            current = handle.sharded.grad
            if current is expected:
                continue
            if current is None:
                raise ValueError(
                    f"A hook set the gradient of ({_name(handle)}) to None. Zero it in place, e.g. with grad.zero_()."
                )
            expected.copy_(current)
            handle.sharded.grad = expected

    def check_grads(self):
        """Checks that the gradient of every sharded parameter still views the gradient shard.

        Raises:
            ValueError: If a gradient was replaced after the final reduction.
        """
        for handle, expected in self._sharded_grads:
            if handle.sharded.grad is not expected:
                raise ValueError(
                    f"The gradient of ({_name(handle)}) was replaced, so the runtime can neither scale nor clear it. "
                    "Change the gradient in place."
                )

    def _reduce_scatter(self):
        grad_shard = cast(Tensor, self.grad_shard)
        ledger = self._ledger
        size = cast(ManagedBuffer, self.grads).tensor.nbytes + (grad_shard.nbytes if self.reduced else 0)
        ledger.reserve(size)
        # The rows of each rank next to each other, the layout of the collective. One collective per bucket, as in
        # the gather.
        source = torch.cat(self._grad_rows, dim=1).view(-1)
        # The first reduction of a step lands on zeros, so it needs neither a temporary nor an addition.
        output = torch.empty_like(grad_shard) if self.reduced else grad_shard
        dist.reduce_scatter_single(output, source, op=dist.ReduceOp.SUM, group=self.plan.gather_group)
        if self.reduced:
            grad_shard.add_(output)
        self.reduced = True
        ledger.retire([torch.cuda.current_stream().record_event()], size)

    def zero_grad(self, compute_stream: torch.cuda.Stream):
        """Clears the gradients and drops the weights that the optimizer changed.

        Args:
            compute_stream: The stream that last read the weights.
        """
        if self.grad_shard is not None:
            self.grad_shard.zero_()
        self.reduced = False
        if self.weights_ready and self.plan.trainable:
            self.release_weights(compute_stream)

    def invalidate_weights(self, compute_stream: torch.cuda.Stream):
        """Drops the unsharded weights, including frozen and inference weights.

        Args:
            compute_stream: The stream that last read the weights.
        """
        self.finish_unshard()
        if self.weights is not None and self.weights.allocated:
            self.release_weights(compute_stream)
        self.weights_ready = False

    def unbind(self):
        """Releases the buffers after their final reads and detaches the gradients of the sharded parameters."""
        # Pinned buffers can still be read on the compute stream when the runtime is torn down.
        self.finish_unshard()
        for buffer in (self.weights, self.grads):
            if buffer is not None and buffer.allocated:
                buffer.release([torch.cuda.current_stream()])
        self.weights_ready = False
        for handle in self.plan.handles:
            handle.sharded.grad = None
        self._sharded_grads = []
        self.grad_shard = None
