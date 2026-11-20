"""The unsharded weights of a gathered bucket and the all-gather that fills them."""

import dataclasses
from typing import cast

import torch
import torch.distributed as dist
from torch import Tensor

from ..plan import BucketPlan, GatherPlan, chunk_places, local_tensor
from .buffer import ManagedBuffer
from .parameter import ParameterHandle, new_unsharded


@dataclasses.dataclass(slots=True)
class _PendingGather:
    """A started all-gather.

    Attributes:
        work: The running collective.
        gathered: The output of the collective: the chunks of all parameters of one rank, rank after rank.
        source: The input of the collective. It must stay alive until the collective ends.
    """

    work: dist.Work
    gathered: Tensor
    source: Tensor


class WeightGather:
    """The unsharded weights of a gathered bucket, which one all-gather fills from the shards."""

    def __init__(self, plan: BucketPlan, gather: GatherPlan, gather_stream: torch.cuda.Stream):
        """Constructs the ``WeightGather`` object with unsharded parameters that have no memory yet.

        Args:
            plan: The plan of the bucket.
            gather: The gather of the bucket.
            gather_stream: The stream that gathers weights and owns their memory.
        """
        self._gather = gather
        self._parts = gather.process_group.size()
        self._stream = gather_stream
        self._ready = False
        self._pending: _PendingGather | None = None

        self._buffer = ManagedBuffer(plan.numel, plan.dtype, gather_stream)
        # The views need memory to be created. The parameters view one Storage, so one resize frees all of them.
        self._buffer.acquire()
        self._handles: list[ParameterHandle] = []
        self._chunks: list[Tensor] = []
        for parameter, place in zip(plan.parameters, chunk_places(gather, self._buffer.tensor), strict=True):
            unsharded = new_unsharded(parameter, place[: parameter.sharded.numel()].view(parameter.sharded.shape))
            self._handles.append(ParameterHandle(plan=parameter, unsharded=unsharded))
            # Views of .data, because a write to the weights bumps the version that saved graphs check.
            self._chunks.append(place.data.view(self._parts, -1))
        self._buffer.release()

    @property
    def handles(self) -> list[ParameterHandle]:
        """The parameters, whose unsharded weights view the weight buffer."""
        return self._handles

    @torch.no_grad()
    def start(self, weights_written: torch.cuda.Event):
        """Starts the all-gather of the unsharded weights, unless they are ready or already gathering.

        Args:
            weights_written: The event after the latest writes to the sharded weights.
        """
        if self._pending is not None or self._ready:
            return

        self._buffer.acquire()

        # The optimizer writes the sharded weights on the compute stream, and the gather reads them.
        self._stream.wait_event(weights_written)
        with torch.cuda.stream(self._stream):
            pieces = []
            for handle, chunks in zip(self._handles, self._chunks, strict=True):
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
        self._pending = _PendingGather(cast(dist.Work, work), gathered, source)

    @torch.no_grad()
    def finish(self):
        """Makes the current stream wait for the started all-gather.

        Without a started all-gather, it does nothing.
        """
        if self._pending is None:
            return
        pending = self._pending
        self._pending = None

        with torch.cuda.stream(self._stream):
            # Inside the stream: a collective orders only the stream that waits for it.
            pending.work.wait()
            widths = [chunks.shape[1] for chunks in self._chunks]
            torch.split_with_sizes_copy(pending.gathered.view(self._parts, -1), widths, dim=1, out=self._chunks)
            ready = self._stream.record_event()

        torch.cuda.current_stream().wait_event(ready)
        # The weights stay valid until they are freed, so a later start() does nothing.
        self._ready = True

    def free(self):
        """Frees the unsharded weights after the reads of the current stream.

        The next all-gather of any bucket starts only after these reads.
        """
        self._ready = False
        self._buffer.release(after=torch.cuda.current_stream().record_event())

    def invalidate(self):
        """Drops the unsharded weights, also while they are gathering, so the next use gathers the shards anew."""
        self.finish()
        if self._buffer.allocated:
            self.free()
