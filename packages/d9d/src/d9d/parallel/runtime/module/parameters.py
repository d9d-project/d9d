from collections.abc import Generator, Sequence
from contextlib import contextmanager

from torch import nn

from ..bucket import (
    AnyReduction,
    BucketGradients,
    ParameterHandle,
    ReplicatedReduction,
    ShardedReduction,
    SideStreams,
    WeightGather,
    make_local_handle,
)
from ..plan import BucketPlan, ModulePlan
from .groups import BoundGroup, ParameterGroups
from .regions import BackwardPass, RegionPlan, plan_regions


class ModuleParameters:
    """The parameters of one registered module: the weights of its groups, its region order and its backward passes."""

    def __init__(self, name: str, groups: ParameterGroups, reductions: Sequence[AnyReduction]):
        """Constructs the ``ModuleParameters`` object.

        Args:
            name: The name of the module, which errors report.
            groups: The parameter groups of the module.
            reductions: The gradient reductions of the module, in reduction order.
        """
        self._name = name
        self._groups = groups
        self._reductions = reductions
        self._regions: RegionPlan | None = None

    @property
    def order(self) -> tuple[nn.Module | None, ...] | None:
        """The root of the group of each region of the recorded order, or ``None`` before ``record_order()``."""
        if self._regions is None:
            return None
        return tuple(None if index is None else self._groups.root(index) for index in self._regions.order)

    def _names(self, order: Sequence[int | None]) -> list[str | None]:
        return [None if index is None else self._groups.name(index) for index in order]

    def record_order(self, order: Sequence[nn.Module | None]):
        """Records the order of the first forward pass, or checks a later one against it.

        Args:
            order: The root of the group of each region of the forward pass, or ``None`` for a region without a
                group.

        Raises:
            ValueError: If the bound parameters are closed, if a module is not the root of a group, if the order
                differs from the recorded one, or if it omits parameters.
        """
        self._groups.require_open()
        indices = tuple(None if root is None else self._groups.index(root) for root in order)
        if self._regions is not None:
            if self._regions.order != indices:
                raise ValueError(
                    f"The parameter group order ({self._names(indices)}) of the module ({self._name}) differs from "
                    f"the recorded order ({self._names(self._regions.order)}). Use the same order for every "
                    "microbatch and rank."
                )
            return
        omitted = self._groups.omitted(indices)
        if omitted:
            raise ValueError(
                f"The forward pass of the module ({self._name}) omitted the parameters ({omitted}). Enter the group "
                "of every parameter on every forward pass."
            )
        self._regions = plan_regions(indices, self._reductions)

    @contextmanager
    def group_weights(self, root: nn.Module, next_root: nn.Module | None) -> Generator[None, None, None]:
        """Puts the unsharded weights of a group into its modules for a region of the forward pass.

        It gathers the weights unless a previous region prefetched or kept them, and starts the gather of the group of
        ``next_root``. On exit, it frees the weights that this group does not use.

        Args:
            root: The root of the group.
            next_root: The root of the group of the next region, or ``None``.

        Raises:
            ValueError: If the bound parameters are closed, or if a module is not the root of a group.
        """
        self._groups.require_open()
        index = self._groups.index(root)
        next_index = None if next_root is None else self._groups.index(next_root)
        with self._groups.weights(index, next_index):
            yield

    def backward(self, final: bool | None) -> BackwardPass:
        """Starts a backward pass over the recorded order.

        Args:
            final: Whether this is the final backward pass of the step, or ``None`` for a pass that writes no weight
                gradients, such as the input pass of a split backward pass.

        Returns:
            The backward pass.

        Raises:
            ValueError: If the bound parameters are closed, or if no order is recorded.
        """
        self._groups.require_open()
        if self._regions is None:
            raise ValueError(
                f"The module ({self._name}) has no recorded order. Call record_order() after the first forward pass."
            )
        return BackwardPass(self._name, self._groups, self._regions, final)


class BoundModule:
    """The buffers, the hooks and the unsharded parameters of one registered module while it is bound."""

    def __init__(
        self,
        name: str,
        groups: ParameterGroups,
        gathers: Sequence[WeightGather],
        reductions: Sequence[AnyReduction],
    ):
        """Constructs the ``BoundModule`` object.

        Args:
            name: The name of the module, which errors report.
            groups: The parameter groups of the module.
            gathers: The weight gathers of the module.
            reductions: The gradient reductions of the module, in reduction order.
        """
        self._groups = groups
        self._gathers = gathers
        self._reductions = reductions
        self._parameters = ModuleParameters(name, groups, reductions)

    @property
    def parameters(self) -> ModuleParameters:
        """The parameters of the module for its caller."""
        return self._parameters

    @property
    def gradients(self) -> list[BucketGradients]:
        """The gradients of the trainable buckets."""
        return [reduction.gradients for reduction in self._reductions]

    def unreduced_groups(self) -> list[str]:
        """Returns the names of the groups that no final backward pass of the step reduced.

        Returns:
            The names of the groups.
        """
        indices = {
            index for reduction in self._reductions if not reduction.gradients.reduced for index in reduction.groups
        }
        return [self._groups.name(index) for index in sorted(indices)]

    def zero_grad(self):
        """Zeroes the gradients."""
        for reduction in self._reductions:
            reduction.zero_grad()

    def invalidate_weights(self):
        """Drops the unsharded weights, so the next region gathers the sharded weights again."""
        for gather in self._gathers:
            gather.invalidate()

    def close(self):
        """Frees the buffers and the unsharded parameters, and sets the gradients of the sharded parameters to ``None``.

        Afterwards, the methods of its ``ModuleParameters`` raise.
        """
        self.invalidate_weights()
        for reduction in self._reductions:
            reduction.close()
        self._groups.close()


def bind_module(
    module: nn.Module,
    plan: ModulePlan,
    streams: SideStreams,
) -> BoundModule:
    """Builds the unsharded parameters, the weight gathers and the gradient reductions of a registered module.

    Args:
        module: The registered module.
        plan: The plan of the module.
        streams: The side streams of the runtime.

    Returns:
        The bound module.
    """
    # A parameter without a gather views its local data. The unsharded parameters of a gathered bucket view its
    # weight buffer, so its gather creates them.
    gathered = {x for bucket in plan.buckets if bucket.gather is not None for x in bucket.parameters}
    handles = {x: make_local_handle(x) for x in plan.parameters if x not in gathered}
    gathers: list[WeightGather] = []
    gathers_by_group: list[list[WeightGather]] = [[] for _ in plan.groups]
    for bucket in plan.buckets:
        if bucket.gather is None:
            continue
        gather = WeightGather(bucket, bucket.gather, streams.gather)
        gathers.append(gather)
        handles.update((handle.plan, handle) for handle in gather.handles)
        for group in bucket.groups:
            gathers_by_group[group].append(gather)

    # The reductions keep the order of the plan, which is the reduction order.
    reductions: list[AnyReduction] = []
    try:
        for bucket in plan.buckets:
            reduction = _new_reduction(bucket, [handles[x] for x in bucket.parameters], streams)
            if reduction is not None:
                reductions.append(reduction)
    except Exception:
        # The gradients of the sharded parameters would keep the gradient memory of the built reductions.
        for reduction in reductions:
            reduction.close()
        raise

    groups = ParameterGroups(
        module=module,
        groups=[
            BoundGroup(
                root=group.root,
                name=group.name,
                handles=[handles[x] for x in group.parameters],
                gathers=gathers_by_group[index],
            )
            for index, group in enumerate(plan.groups)
        ],
    )
    return BoundModule(type(module).__name__, groups, gathers, reductions)


def _new_reduction(
    bucket: BucketPlan,
    handles: list[ParameterHandle],
    streams: SideStreams,
) -> AnyReduction | None:
    match bucket.gather:
        case None:
            # The plan has a bucket without a gather only for trainable parameters.
            return ReplicatedReduction(bucket, handles, streams.reduce)
        case gather if bucket.trainable:
            return ShardedReduction(bucket, gather, handles, streams.reduce, streams.pending)
        case _:
            return None
