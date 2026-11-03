"""Stage and step lifecycle of explicit data parallelism."""

import dataclasses
from collections.abc import Collection, Generator, Sequence
from contextlib import contextmanager
from typing import cast

import torch
from torch import nn

from d9d.parallel.memory import MemoryLedger

from .bucket import Bucket
from .parameter import (
    ParameterHandle,
    bind_parameters,
    convert_buffers,
    forward_tensor_hooks,
    materialize_own_copies,
    refresh_own_copies,
)
from .plan import StagePlan, minimum_stage_budget, plan_stages
from .types import BucketId, ReductionPlan


@dataclasses.dataclass(slots=True)
class StageState:
    """The parameter state of one pipeline stage.

    Attributes:
        handles: The parameters that each segment uses, by segment index.
        buckets: The buckets in reduction order.
        gathered: The gathered buckets of each segment, by segment index.
    """

    handles: list[list[ParameterHandle]]
    buckets: list[Bucket]
    gathered: list[list[Bucket]]


class ParameterRuntime:
    """Runtime of data parallelism that the pipeline program drives.

    Modules keep sharded parameters. Around each segment, the program puts the unsharded parameters into the
    modules, and after it, the program reduces the gradients that are complete. Buckets own their buffers and
    collectives.
    """

    def __init__(
        self,
        gather_dims: Collection[str],
        bucket_size_mb: int,
        compute_dtype: torch.dtype | None = None,
        memory_budget_bytes: int | None = None,
    ):
        """Constructs the ``ParameterRuntime`` object.

        Args:
            gather_dims: The mesh dimensions along which data parallelism shards parameters. The runtime gathers
                parameters along them, and leaves every other ``Shard`` dimension to the layer.
            bucket_size_mb: The size limit of buckets without a gather, in MiB.
            compute_dtype: The dtype of the unsharded weights, or ``None`` to keep the dtype of the storage.
            memory_budget_bytes: The budget of gathered weights and gradients. Room above the minimum keeps the
                weights gathered for the whole step. ``None`` selects a quarter of the GPU memory.
        """
        self._gather_dims = frozenset(gather_dims)
        self._bucket_size_mb = bucket_size_mb
        self._compute_dtype = compute_dtype
        self._memory_budget_bytes = memory_budget_bytes
        # A checkpoint can change the sharded weights before the first step.
        self._stale = True
        self._modules: dict[int, nn.Module] = {}
        self._stages: dict[int, StageState] = {}
        self._gather_stream: torch.cuda.Stream | None = None
        self._reduce_stream: torch.cuda.Stream | None = None
        self._ledger: MemoryLedger | None = None
        self._storage_written: torch.cuda.Event | None = None
        self._reduced: set[BucketId] = set()

    def register_stage(self, stage_index: int, module: nn.Module):
        """Registers a pipeline stage. A module without segments runs as one segment.

        Args:
            stage_index: The index of the stage.
            module: The stage module.
        """
        self._modules[stage_index] = module

    def bind(self, training: bool = True):
        """Builds the unsharded parameters and the gradient buffers.

        Args:
            training: Whether the job computes gradients. Without them, kept weights stay valid between steps.

        Raises:
            ValueError: If the runtime is already bound or a parameter is shared by several stages.
        """
        if self._reduce_stream is not None:
            raise ValueError("The runtime is already bound. Call unbind() first.")

        plans = plan_stages(self._modules, self._compute_dtype, training, self._bucket_size_mb, self._gather_dims)

        self._gather_stream = torch.cuda.Stream()
        self._reduce_stream = torch.cuda.Stream()
        self._stale = True
        self._storage_written = None
        self._reduced.clear()
        try:
            if self._compute_dtype is not None:
                for module in self._modules.values():
                    convert_buffers(module, self._compute_dtype)
            spare = self._open_ledger(list(plans.values()))
            for stage_index, plan in plans.items():
                spare = self._bind_stage(stage_index, plan, spare)
        except Exception:
            self.unbind()
            raise

    def _bind_stage(self, stage_index: int, plan: StagePlan, spare: int) -> int:
        buckets = [
            Bucket(
                bucket_plan,
                cast(torch.cuda.Stream, self._gather_stream),
                cast(torch.cuda.Stream, self._reduce_stream),
                cast(MemoryLedger, self._ledger),
            )
            for bucket_plan in plan.buckets
        ]
        gathered: list[list[Bucket]] = [[] for _ in plan.handles]
        self._stages[stage_index] = StageState(plan.handles, buckets, gathered)
        handles = {id(x): x for segment_handles in plan.handles for x in segment_handles}.values()
        # Gathered buckets replace the unsharded parameters with views of their buffers, so the copies of the other
        # parameters are made after them and before the plain buckets point gradients at them.
        for bucket in buckets:
            if bucket.plan.gathers:
                bucket.bind()
                for segment_index in bucket.plan.segments:
                    gathered[segment_index].append(bucket)
        materialize_own_copies(handles)
        for bucket in buckets:
            if not bucket.plan.gathers:
                bucket.bind()
        for handle in handles:
            forward_tensor_hooks(handle)
        # Frozen weights stay valid between steps. Then keep the weights that the backward pass reads soonest.
        for bucket in sorted(buckets, key=lambda x: x.plan.trainable):
            size = bucket.plan.numel * bucket.plan.handles[0].unsharded.dtype.itemsize
            if bucket.plan.gathers and size <= spare:
                bucket.pinned = True
                spare -= size
        return spare

    def _open_ledger(self, stages: Sequence[StagePlan]) -> int:
        minimum = max((minimum_stage_budget(stage.buckets) for stage in stages), default=0)
        budget = self._memory_budget_bytes
        if budget is None:
            budget = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory // 4
        self._ledger = MemoryLedger(max(minimum, budget))
        return max(0, budget - minimum)

    def unbind(self):
        """Releases the unsharded parameters and the gradient buffers."""
        if self._reduce_stream is not None:
            torch.cuda.current_stream().wait_stream(self._reduce_stream)
        if self._gather_stream is not None:
            torch.cuda.current_stream().wait_stream(self._gather_stream)
        for stage in self._stages.values():
            for bucket in stage.buckets:
                bucket.unbind()
        self._stages = {}
        self._gather_stream = None
        self._reduce_stream = None
        self._ledger = None
        self._storage_written = None
        self._stale = True
        self._reduced.clear()

    @contextmanager
    def segment_weights(
        self, stage_index: int, segment_index: int, writes_grads: bool, next_segment: int | None = None
    ) -> Generator[None, None, None]:
        """Puts the unsharded weights of a segment into its modules for the duration of the block.

        The weights are gathered on entry and freed on exit, unless the budget keeps them. The block that writes
        the first gradient of a bucket in a backward pass also gets cleared gradient buffers.

        Args:
            stage_index: The index of the stage.
            segment_index: The index of the segment in the stage.
            writes_grads: Whether this pass writes weight gradients.
            next_segment: The segment that runs next in this pass. Its gather starts during this segment.

        Yields:
            Nothing.
        """
        if self._stale:
            # The first segment of a step comes after the optimizer and after checkpoint loads.
            self._stale = False
            for stage in self._stages.values():
                refresh_own_copies(stage.handles)
            self._storage_written = torch.cuda.current_stream().record_event()
        storage_written = cast(torch.cuda.Event, self._storage_written)
        stage = self._stages[stage_index]
        gathered = stage.gathered[segment_index]
        for bucket in gathered:
            bucket.start_unshard(storage_written)
            bucket.finish_unshard()
            # The backward pass of a tied bucket writes its gradient in several segments, from the last one on.
            if writes_grads and bucket.grads is not None and segment_index == bucket.plan.segments[-1]:
                bucket.prepare_grads()
        # The own gathers start first, so the prefetch cannot take the budget that they need.
        following = [] if next_segment is None else stage.gathered[next_segment]
        for bucket in following:
            bucket.start_unshard(storage_written)
        handles = stage.handles[segment_index]
        bind_parameters(handles, unsharded=True)
        try:
            yield
        finally:
            bind_parameters(handles, unsharded=False)
        compute_stream = torch.cuda.current_stream()
        for bucket in gathered:
            # A bucket that the next segment uses too stays gathered, otherwise that segment gathers it again.
            if not bucket.pinned and all(bucket is not x for x in following):
                bucket.release_weights(compute_stream)

    def reduction_plan(self, stage_index: int, final: bool) -> ReductionPlan:
        """Returns which buckets of a stage reduce after which segment of one backward pass.

        A bucket reduces after its first segment, which is the last one of the backward pass. A gathered bucket
        reduces after every pass. Other buckets reduce only after the final pass.

        Args:
            stage_index: The index of the stage.
            final: Whether this is the last backward pass of the stage in the step.

        Returns:
            The plan of the pass.
        """
        after_segment: dict[int, list[BucketId]] = {}
        for index, bucket in enumerate(self._stages[stage_index].buckets):
            if bucket.plan.trainable and (bucket.plan.gathers or final):
                after_segment.setdefault(bucket.plan.segments[0], []).append(BucketId(stage_index, index))
        return ReductionPlan(final, {segment: tuple(ids) for segment, ids in after_segment.items()})

    @torch.no_grad()
    def reduce(self, bucket_id: BucketId, final: bool):
        """Starts the reduction of a bucket on the reduce stream.

        Args:
            bucket_id: A bucket from ``reduction_plan``.
            final: The ``final`` flag of the plan.

        Raises:
            ValueError: If the runtime is not bound.
        """
        if self._reduce_stream is None:
            raise ValueError("The runtime is not bound. Call bind() first.")

        self._stages[bucket_id.stage].buckets[bucket_id.index].reduce(final)
        if final:
            self._reduced.add(bucket_id)

    def wait(self):
        """Makes the current stream wait for all reductions.

        Raises:
            ValueError: If the runtime is not bound, or if the program did not reduce a bucket.
        """
        if self._reduce_stream is None:
            raise ValueError("The runtime is not bound. Call bind() first.")

        plans = [self.reduction_plan(stage_index, final=True) for stage_index in self._stages]
        expected = {bucket_id for plan in plans for ids in plan.after_segment.values() for bucket_id in ids}
        if expected != self._reduced:
            missing = sorted((x.stage, x.index) for x in expected - self._reduced)
            raise ValueError(f"The program did not reduce the buckets ({missing}).")
        for stage in self._stages.values():
            for bucket in stage.buckets:
                bucket.check_grads()

        torch.cuda.current_stream().wait_stream(self._reduce_stream)

    @torch.no_grad()
    def scale_grads(self, factor: torch.Tensor):
        """Multiplies all gradients in place.

        Args:
            factor: The scalar to multiply the gradients by.
        """
        buckets = (bucket for stage in self._stages.values() for bucket in stage.buckets)
        shards = [bucket.grad_shard for bucket in buckets if bucket.grad_shard is not None]
        if shards:
            torch._foreach_mul_(shards, factor)

    def zero_grad(self):
        """Resets all gradients for the next step."""
        compute_stream = torch.cuda.current_stream()
        for stage in self._stages.values():
            for bucket in stage.buckets:
                bucket.zero_grad(compute_stream)
        self._stale = True
        self._reduced.clear()

    def invalidate_weights(self):
        """Drops the unsharded weights before a load of the sharded ones, including frozen and inference weights.

        This can be called while the runtime is unbound. The next segment reads the new sharded weights.
        """
        self._stale = True
        if not self._stages:
            return
        compute_stream = torch.cuda.current_stream()
        for stage in self._stages.values():
            for bucket in stage.buckets:
                bucket.invalidate_weights(compute_stream)

    @property
    def compute_dtype(self) -> torch.dtype | None:
        """The dtype of the unsharded weights, or ``None`` if it is the dtype of the storage."""
        return self._compute_dtype
