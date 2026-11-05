"""Stage and step lifecycle of explicit data parallelism."""

import dataclasses
from collections.abc import Collection, Generator, Sequence
from contextlib import contextmanager

import torch
from torch import nn

from d9d.pipelining.api import StageSegment

from .bucket import AnyBucket, GatheredBucket, GradientShard, ReplicatedBucket
from .memory import MemoryLedger
from .parameter import (
    ParameterHandle,
    bind_parameters,
    convert_buffers,
    install_tensor_hook_bridge,
    install_usage_hook,
    make_local_handle,
    refresh_own_copies,
)
from .plan import StagePlan, minimum_stage_budget, plan_stages
from .types import BucketId, ReductionPlan


@dataclasses.dataclass(slots=True)
class _StageState:
    """The parameter state of one pipeline stage.

    Attributes:
        segments: The segments of the stage, in forward order.
        handles: The parameters of the stage, each once.
        handles_by_segment: The parameters that each segment uses, by segment index.
        buckets: The buckets in reduction order.
        gathered_by_segment: The gathered buckets of each segment, by segment index.
    """

    segments: Sequence[StageSegment]
    handles: list[ParameterHandle] = dataclasses.field(default_factory=list)
    handles_by_segment: list[list[ParameterHandle]] = dataclasses.field(default_factory=list)
    buckets: list[AnyBucket] = dataclasses.field(default_factory=list)
    gathered_by_segment: list[list[GatheredBucket]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class _BoundState:
    """The resources that exist between ``bind()`` and ``unbind()``.

    Attributes:
        gather_stream: The stream that gathers weights.
        reduce_stream: The stream that reduces gradients.
        ledger: The memory budget of gathered weights and gradients.
        stages: The state of each stage, by stage index.
        storage_written: The event after the latest writes to the sharded weights, or ``None`` before the first
            segment of a step.
        reduced: The buckets that a final pass reduced in this step.
        finished: Whether ``wait()`` has started to finish this step, also if a hook raised.
    """

    gather_stream: torch.cuda.Stream
    reduce_stream: torch.cuda.Stream
    ledger: MemoryLedger
    stages: dict[int, _StageState]
    storage_written: torch.cuda.Event | None = None
    reduced: set[BucketId] = dataclasses.field(default_factory=set)
    finished: bool = False


class ParameterRuntime:
    """Runtime of data parallelism that the pipeline program drives.

    Modules keep sharded parameters. Around each segment, the program puts the unsharded parameters into the
    modules, and after it, the program reduces the gradients that are complete. Buckets own their buffers and
    collectives.

    The lifecycle is ``register_stage()`` for every stage, then ``bind()``, the steps, and ``unbind()``. A bound
    runtime can be unbound and bound again, e.g. around an offload.
    """

    def __init__(
        self,
        gather_dims: Collection[str],
        bucket_size_mib: int,
        compute_dtype: torch.dtype | None = None,
        memory_budget_bytes: int = 0,
    ):
        """Constructs the ``ParameterRuntime`` object.

        Args:
            gather_dims: The mesh dimensions along which data parallelism shards parameters. The runtime gathers
                parameters along them, and leaves every other ``Shard`` dimension to the layer.
            bucket_size_mib: The size limit of buckets without a gather, in MiB.
            compute_dtype: The dtype of the unsharded weights, or ``None`` to keep the dtype of the storage. ``bind()``
                also converts the non-persistent floating-point buffers of the stages to it.
            memory_budget_bytes: The target budget of gathered weights, gradient buffers and collective temporaries,
                without the gradient shards. A smaller value is raised to the floor that the largest segment needs.
                Room above the floor keeps weights gathered for the step. The default keeps no weights between
                segments, as FSDP does.
        """
        self._gather_dims = frozenset(gather_dims)
        self._bucket_size_mib = bucket_size_mib
        self._compute_dtype = compute_dtype
        self._memory_budget_bytes = memory_budget_bytes
        self._modules: dict[int, nn.Module] = {}
        self._bound: _BoundState | None = None
        # A checkpoint can change the sharded weights before the first step.
        self._stale = True

    def register_stage(self, stage_index: int, module: nn.Module):
        """Registers a pipeline stage. A module without segments runs as one segment.

        Args:
            stage_index: The index of the stage.
            module: The stage module.

        Raises:
            ValueError: If the runtime is bound, or if the stage is already registered.
        """
        if self._bound is not None:
            raise ValueError(f"The runtime is bound, so it cannot register stage ({stage_index}). Call unbind() first.")
        if stage_index in self._modules:
            raise ValueError(f"The stage ({stage_index}) is already registered.")
        self._modules[stage_index] = module

    def _require_bound(self) -> _BoundState:
        if self._bound is None:
            raise ValueError("The runtime is not bound. Call bind() first.")
        return self._bound

    def bind(self, training: bool = True):
        """Builds the unsharded parameters and the gradient buffers.

        Args:
            training: Whether the job computes gradients. Without them, kept weights stay valid between steps.

        Raises:
            ValueError: If the runtime is already bound, if a stage has no valid segments, or if a parameter is
                shared by several stages.
        """
        if self._bound is not None:
            raise ValueError("The runtime is already bound. Call unbind() first.")

        plans = plan_stages(self._modules, self._compute_dtype, training, self._bucket_size_mib, self._gather_dims)
        minimum = max((minimum_stage_budget(plan.buckets) for plan in plans.values()), default=0)
        budget = self._memory_budget_bytes
        bound = _BoundState(
            gather_stream=torch.cuda.Stream(),
            reduce_stream=torch.cuda.Stream(),
            ledger=MemoryLedger(max(minimum, budget)),
            stages={},
        )
        self._bound = bound
        self._stale = True

        try:
            if self._compute_dtype is not None:
                convert_buffers(self._modules.values(), self._compute_dtype)

            spare = max(0, budget - minimum)
            for stage_index, plan in plans.items():
                # Teardown must see the stage even if one of its buckets or hooks fails to bind.
                state = _StageState(segments=plan.segments)
                bound.stages[stage_index] = state
                spare = self._bind_stage(bound, state, plan, spare)
        except Exception:
            self.unbind()
            raise

    @staticmethod
    def _bind_stage(bound: _BoundState, state: _StageState, plan: StagePlan, spare: int) -> int:
        # A parameter without a gather views its local data, or holds a copy of it in the compute dtype. The
        # unsharded parameters of a gathered bucket view its weight buffer, so the bucket creates them.
        gathered_ids = {id(x) for bucket in plan.buckets if bucket.gathers for x in bucket.parameters}
        handles = {id(x): make_local_handle(x) for x in plan.parameters if id(x) not in gathered_ids}

        # The buckets keep the order of the plan, because bucket ids index them. Each bucket joins the stage at once,
        # so teardown releases it if a later one fails.
        state.gathered_by_segment = [[] for _ in plan.parameters_by_segment]
        for bucket_plan in plan.buckets:
            if bucket_plan.gather_group is None:
                local_handles = [handles[id(x)] for x in bucket_plan.parameters]
                state.buckets.append(ReplicatedBucket(bucket_plan, local_handles, bound.reduce_stream))
                continue

            bucket = GatheredBucket(
                bucket_plan, bucket_plan.gather_group, bound.gather_stream, bound.reduce_stream, bound.ledger
            )
            state.buckets.append(bucket)
            handles.update((id(handle.plan), handle) for handle in bucket.handles)
            for segment_index in bucket_plan.segments:
                state.gathered_by_segment[segment_index].append(bucket)

        state.handles = list(handles.values())
        state.handles_by_segment = [[handles[id(x)] for x in segment] for segment in plan.parameters_by_segment]
        for handle in state.handles:
            install_tensor_hook_bridge(handle)
            install_usage_hook(handle)

        # Frozen weights stay valid between steps. Then keep the weights that the backward pass reads soonest.
        gathered = [x for x in state.buckets if isinstance(x, GatheredBucket)]
        for bucket in sorted(gathered, key=lambda x: x.plan.trainable):
            size = bucket.plan.numel * bucket.plan.compute_dtype.itemsize
            if size <= spare:
                bucket.keep_weights = True
                spare -= size

        return spare

    def unbind(self):
        """Releases the unsharded parameters and the gradient buffers. It does nothing on an unbound runtime."""
        bound = self._bound
        if bound is None:
            return
        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        torch.cuda.current_stream().wait_stream(bound.gather_stream)
        for stage in bound.stages.values():
            for bucket in stage.buckets:
                bucket.unbind()
        self._bound = None
        self._stale = True

    def stage_segments(self, stage_index: int) -> Sequence[StageSegment]:
        """Returns the segments of a stage, which ``bind()`` read and checked once.

        Args:
            stage_index: The index of the stage.

        Returns:
            The segments, in forward order.

        Raises:
            ValueError: If the runtime is not bound.
        """
        return self._require_bound().stages[stage_index].segments

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

        Raises:
            ValueError: If the runtime is not bound.
        """
        bound = self._require_bound()
        if self._stale or bound.storage_written is None:
            # The first segment of a step comes after the optimizer and after checkpoint loads.
            self._stale = False
            for state in bound.stages.values():
                refresh_own_copies(state.handles)
            bound.storage_written = torch.cuda.current_stream().record_event()
        storage_written = bound.storage_written

        stage = bound.stages[stage_index]
        gathered = stage.gathered_by_segment[segment_index]
        for bucket in gathered:
            bucket.start_unshard(storage_written)
            bucket.finish_unshard()
            # The backward pass of a tied bucket writes its gradient in several segments, from the last one on.
            if writes_grads and bucket.plan.trainable and segment_index == bucket.plan.segments[-1]:
                bucket.prepare_grads()
        # The own gathers start first, so the prefetch cannot take the budget that they need.
        following = [] if next_segment is None else stage.gathered_by_segment[next_segment]
        for bucket in following:
            bucket.start_unshard(storage_written)

        handles = stage.handles_by_segment[segment_index]
        bind_parameters(handles, unsharded=True)
        try:
            yield
        finally:
            bind_parameters(handles, unsharded=False)
            compute_stream = torch.cuda.current_stream()
            for bucket in gathered:
                # A bucket that the next segment uses too stays gathered, otherwise that segment gathers it again.
                if not bucket.keep_weights and all(bucket is not x for x in following):
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

        Raises:
            ValueError: If the runtime is not bound.
        """
        after_segment: dict[int, list[BucketId]] = {}
        for index, bucket in enumerate(self._require_bound().stages[stage_index].buckets):
            if bucket.plan.trainable and (bucket.plan.gathers or final):
                after_segment.setdefault(bucket.plan.segments[0], []).append(BucketId(stage_index, index))
        return ReductionPlan(final, {segment: tuple(ids) for segment, ids in after_segment.items()})

    @torch.no_grad()
    def reduce(self, plan: ReductionPlan, bucket_id: BucketId):
        """Starts the reduction of a bucket on the reduce stream.

        Args:
            plan: The plan of the backward pass, from ``reduction_plan``.
            bucket_id: A bucket of ``plan``.

        Raises:
            ValueError: If the runtime is not bound.
        """
        bound = self._require_bound()
        bound.stages[bucket_id.stage].buckets[bucket_id.index].reduce(plan.final)
        if plan.final:
            bound.reduced.add(bucket_id)

    def wait(self):
        """Finishes the gradients of the step for the current stream.

        A parameter that no rank used gets no gradient. The others run their post-accumulate-grad hooks. Call it
        once per step, after the final reductions and before the optimizer step.

        Raises:
            ValueError: If the runtime is not bound, if this step is already finished, if the program did not reduce a
                bucket, or if a gradient was replaced during the step.
        """
        bound = self._require_bound()
        if bound.finished:
            raise ValueError("The gradients of this step are already finished. Call zero_grad() before the next step.")

        plans = [self.reduction_plan(stage_index, final=True) for stage_index in bound.stages]
        expected = {bucket_id for plan in plans for ids in plan.after_segment.values() for bucket_id in ids}
        if expected != bound.reduced:
            missing = sorted((x.stage, x.index) for x in expected - bound.reduced)
            raise ValueError(f"The program did not reduce the buckets ({missing}).")

        shards = self._gradient_shards(bound)
        for shard in shards:
            shard.check()

        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        # From here on, a retry would run the hooks that already ran.
        bound.finished = True
        if not shards:
            return

        # The host must know which parameters got no gradient, because the optimizer skips only those with None.
        used = torch.cat([shard.usage for shard in shards]).tolist()
        offset = 0
        for shard in shards:
            count = len(shard.usage)
            shard.finish([bool(x) for x in used[offset : offset + count]])
            offset += count

    @staticmethod
    def _gradient_shards(bound: _BoundState) -> list[GradientShard]:
        buckets = (bucket for stage in bound.stages.values() for bucket in stage.buckets)
        return [bucket.gradients for bucket in buckets if bucket.gradients is not None]

    @torch.no_grad()
    def scale_grads(self, factor: torch.Tensor):
        """Multiplies all gradients in place.

        Args:
            factor: The scalar to multiply the gradients by.

        Raises:
            ValueError: If the runtime is not bound.
        """
        shards = [shard.tensor for shard in self._gradient_shards(self._require_bound())]
        if shards:
            torch._foreach_mul_(shards, factor)

    def zero_grad(self):
        """Resets all gradients for the next step. Without a bind, there are no gradients to reset."""
        self._stale = True
        bound = self._bound
        if bound is None:
            return
        # A failed step can leave reductions queued on the side stream.
        torch.cuda.current_stream().wait_stream(bound.reduce_stream)
        for stage in bound.stages.values():
            for bucket in stage.buckets:
                bucket.zero_grad()
        bound.reduced.clear()
        bound.finished = False

    def invalidate_weights(self):
        """Drops the unsharded weights before a load of the sharded ones, including frozen and inference weights.

        This can be called while the runtime is unbound. The next segment reads the new sharded weights.
        """
        self._stale = True
        bound = self._bound
        if bound is None:
            return
        compute_stream = torch.cuda.current_stream()
        for stage in bound.stages.values():
            for bucket in stage.buckets:
                if isinstance(bucket, GatheredBucket):
                    bucket.invalidate_weights(compute_stream)

    @property
    def compute_dtype(self) -> torch.dtype | None:
        """The dtype of the unsharded weights, or ``None`` if it is the dtype of the storage."""
        return self._compute_dtype
