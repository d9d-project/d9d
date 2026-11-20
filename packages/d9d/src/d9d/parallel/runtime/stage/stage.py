"""The parameters of one stage: the weights of its groups, the order of its regions and its backward passes."""

from collections.abc import Generator, Sequence
from contextlib import contextmanager

import torch
from torch import nn

from ..bucket import (
    AnyReduction,
    BucketGradients,
    ParameterHandle,
    PendingGradientBuffers,
    ReplicatedReduction,
    ShardedReduction,
    WeightGather,
    make_local_handle,
)
from ..plan import BucketPlan, StagePlan
from .groups import ParameterGroups
from .regions import BackwardPass, RegionPlan, plan_regions


class StageParameters:
    """The parameters of one bound stage: the weights of its groups, the order of its regions and its backward passes.

    The program runs the regions with it. The runtime clears, finishes and releases the gradients through it.
    """

    def __init__(
        self,
        stage_index: int,
        groups: ParameterGroups,
        gathers: Sequence[WeightGather],
        reductions: Sequence[AnyReduction],
    ):
        """Constructs the ``StageParameters`` object.

        ``ParameterRuntime.stage()`` returns it.

        Args:
            stage_index: The index of the stage.
            groups: The parameter groups of the stage.
            gathers: The weight gathers of the stage.
            reductions: The gradient reductions of the stage, in reduction order.
        """
        self._stage_index = stage_index
        self._groups = groups
        self._gathers = gathers
        self._reductions = reductions
        self._regions: RegionPlan | None = None

    def group_index(self, module: nn.Module) -> int:
        """Returns the index of the parameter group of a module.

        Args:
            module: The root of the group.

        Returns:
            The index of the group.

        Raises:
            ValueError: If the runtime was unbound since, or if the module is not the root of a group of this stage.
        """
        self._groups.require_open()
        return self._groups.index(module)

    @property
    def order(self) -> tuple[int | None, ...] | None:
        """The group of each region of the recorded forward pass, or ``None`` before ``record_order()``."""
        return None if self._regions is None else self._regions.order

    def record_order(self, order: Sequence[int | None]):
        """Records the group order of the first forward pass, or checks a later one against it.

        Args:
            order: The group of each region of the forward pass, including repeated groups, or ``None`` for a region
                without a group.

        Raises:
            ValueError: If the runtime was unbound since, if the order names a group that does not exist, if it
                differs from the recorded one, or if it omits parameters.
        """
        self._groups.require_open()
        order = tuple(order)
        if self._regions is not None:
            if self._regions.order != order:
                raise ValueError(
                    f"The parameter group order ({order}) of stage ({self._stage_index}) differs from the recorded "
                    f"order ({self._regions.order}). Use the same order for every microbatch and rank."
                )
            return
        for group in order:
            if group is not None:
                self._groups.check(group)
        omitted = self._groups.omitted(order)
        if omitted:
            raise ValueError(
                f"The forward pass of stage ({self._stage_index}) omitted the parameters ({omitted}). Enter the group "
                "of every parameter on every forward pass."
            )
        self._regions = plan_regions(order, self._reductions)

    @contextmanager
    def group_weights(self, group: int, next_group: int | None = None) -> Generator[None, None, None]:
        """Puts the unsharded weights of a group into its modules for a forward pass.

        It gathers the weights that a previous region did not prefetch or keep. On exit, it frees the weights that
        ``next_group`` does not use. Run backward passes in ``backward()`` instead.

        Args:
            group: The index of the group.
            next_group: The group of the next region, or ``None``. Its weights start gathering before this region.

        Yields:
            Nothing.

        Raises:
            ValueError: If the runtime was unbound since, or if a group does not exist.
        """
        self._groups.require_open()
        self._groups.check(group)
        if next_group is not None:
            self._groups.check(next_group)
        with self._groups.weights(group, next_group):
            yield

    def backward(self, final: bool | None) -> BackwardPass:
        """Starts a backward pass of the stage over the recorded order.

        Args:
            final: Whether this is the last backward pass of the stage in the step, or ``None`` for a pass that
                writes no weight gradients, such as the input pass of a split backward pass.

        Returns:
            The backward pass. Run its regions in reverse order of the forward pass.

        Raises:
            ValueError: If the runtime was unbound since, or if no order is recorded.
        """
        self._groups.require_open()
        if self._regions is None:
            raise ValueError(
                f"The stage ({self._stage_index}) has no recorded order. Call record_order() after the first forward "
                "pass."
            )
        return BackwardPass(self._stage_index, self._groups, self._regions, final)

    @property
    def gradients(self) -> list[BucketGradients]:
        """The gradients of the trainable buckets."""
        return [reduction.gradients for reduction in self._reductions]

    def unreduced_groups(self) -> list[int]:
        """Returns the groups whose gradients no final backward pass of the step reduced.

        Returns:
            The indices of the groups, in ascending order.
        """
        return sorted(
            {group for reduction in self._reductions if not reduction.gradients.reduced for group in reduction.groups}
        )

    def zero_grad(self):
        """Zeroes the gradients for a new step."""
        for reduction in self._reductions:
            reduction.zero_grad()

    def invalidate_weights(self):
        """Drops the unsharded weights, so the next region gathers the sharded ones anew."""
        for gather in self._gathers:
            gather.invalidate()

    def close(self):
        """Frees the buffers, sets the gradients of the sharded parameters to ``None`` and empties the unsharded ones.

        Afterwards, the methods for the program raise.
        """
        self.invalidate_weights()
        for reduction in self._reductions:
            reduction.close()
        self._groups.close()
        self._gathers = []
        self._reductions = []
        self._regions = None


def bind_stage(
    stage_index: int,
    plan: StagePlan,
    gather_stream: torch.cuda.Stream,
    reduce_stream: torch.cuda.Stream,
    pending: PendingGradientBuffers,
) -> StageParameters:
    """Builds the unsharded parameters, the weight gathers and the gradient reductions of a stage.

    Args:
        stage_index: The index of the stage.
        plan: The plan of the stage.
        gather_stream: The stream that gathers weights.
        reduce_stream: The stream that reduces gradients.
        pending: The gradient buffers of all buckets that the reduce stream may still read.

    Returns:
        The parameters of the stage.
    """
    # A parameter without a gather views its local data. The unsharded parameters of a gathered bucket view its
    # weight buffer, so its gather creates them.
    gathered = {x for bucket in plan.buckets if bucket.gather is not None for x in bucket.parameters}
    handles = {x: make_local_handle(x) for x in plan.parameters if x not in gathered}
    gathers: list[WeightGather] = []
    gathers_by_group: list[list[WeightGather]] = [[] for _ in plan.parameters_by_group]
    for bucket in plan.buckets:
        if bucket.gather is None:
            continue
        gather = WeightGather(bucket, bucket.gather, gather_stream)
        gathers.append(gather)
        handles.update((handle.plan, handle) for handle in gather.handles)
        for group in bucket.groups:
            gathers_by_group[group].append(gather)

    # The reductions keep the order of the plan, which is the reduction order.
    reductions: list[AnyReduction] = []
    try:
        for bucket in plan.buckets:
            reduction = _new_reduction(bucket, [handles[x] for x in bucket.parameters], reduce_stream, pending)
            if reduction is not None:
                reductions.append(reduction)
    except Exception:
        # The gradients of the sharded parameters would keep the gradient memory of the built reductions.
        for reduction in reductions:
            reduction.close()
        raise

    groups = ParameterGroups(
        stage_index=stage_index,
        roots=plan.groups,
        handles_by_group=[[handles[x] for x in group] for group in plan.parameters_by_group],
        gathers_by_group=gathers_by_group,
    )
    return StageParameters(stage_index, groups, gathers, reductions)


def _new_reduction(
    bucket: BucketPlan,
    handles: list[ParameterHandle],
    reduce_stream: torch.cuda.Stream,
    pending: PendingGradientBuffers,
) -> AnyReduction | None:
    match bucket.gather:
        case None:
            # The plan has a bucket without a gather only for trainable parameters.
            return ReplicatedReduction(bucket, handles, reduce_stream)
        case gather if bucket.trainable:
            return ShardedReduction(bucket, gather, handles, reduce_stream, pending)
        case _:
            return None
