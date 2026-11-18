"""The parameter groups of one stage: their weights around each region and the reductions of a backward pass."""

import dataclasses
from collections.abc import Generator, Sequence
from contextlib import contextmanager, nullcontext
from typing import cast

import torch
from torch import nn

from .bucket import AnyBucket, GatheredBucket, ReplicatedBucket
from .parameter import ParameterHandle, bind_parameters


@dataclasses.dataclass(slots=True)
class StepState:
    """The state of the current step that all stages share.

    Attributes:
        reduced: The buckets that a final backward pass reduced.
        finished: Whether ``wait()`` finished the gradients of the step.
    """

    reduced: set[AnyBucket] = dataclasses.field(default_factory=set)
    finished: bool = False


@dataclasses.dataclass(slots=True)
class StageState:
    """The parameter state of one bound stage.

    Attributes:
        stage_index: The index of the stage.
        group_indices: The index of each group, by its root module.
        handles: The parameters of the stage, each once.
        handles_by_group: The parameters that each group uses, by group index.
        buckets: The buckets in reduction order.
        gathered_by_group: The gathered buckets of each group, by group index.
        step: The state of the current step.
        order: The group of each region of the first forward pass, or ``None`` before it.
        prepare_at: The trainable gathered buckets whose gradient buffers each region of a backward pass allocates,
            by region index.
        reduce_at: The trainable buckets that each region of a backward pass reduces after it, by region index.
        bound: Whether the runtime is still bound.
    """

    stage_index: int
    group_indices: dict[nn.Module, int]
    handles: list[ParameterHandle]
    handles_by_group: list[list[ParameterHandle]]
    buckets: list[AnyBucket]
    gathered_by_group: list[list[GatheredBucket]]
    step: StepState
    order: tuple[int | None, ...] | None = None
    prepare_at: dict[int, list[GatheredBucket]] = dataclasses.field(default_factory=dict)
    reduce_at: dict[int, list[AnyBucket]] = dataclasses.field(default_factory=dict)
    bound: bool = True

    def require_bound(self):
        """Checks that the runtime is still bound.

        Raises:
            ValueError: If the runtime was unbound since.
        """
        if not self.bound:
            raise ValueError(
                f"The runtime was unbound after stage ({self.stage_index}) was taken. Call stage() again after bind()."
            )

    def check_group(self, group: int):
        """Checks that a group index exists.

        Args:
            group: The index of the group.

        Raises:
            ValueError: If the stage has no such group.
        """
        if not 0 <= group < len(self.handles_by_group):
            raise ValueError(
                f"The group ({group}) does not exist in stage ({self.stage_index}), which has "
                f"({len(self.handles_by_group)}) groups. Pass an index from group_index() or from the recorded order."
            )

    def unbind(self):
        """Drops the buckets and the parameters, so that a kept ``StageParameters`` holds no memory."""
        self.bound = False
        self.group_indices = {}
        self.handles = []
        self.handles_by_group = []
        self.buckets = []
        self.gathered_by_group = []
        self.order = None
        self.prepare_at = {}
        self.reduce_at = {}


@contextmanager
def _group_weights(state: StageState, group: int, next_group: int | None) -> Generator[None, None, None]:
    # A fresh event for every region: the sharded weights can change between regions, e.g. by an optimizer step
    # before an evaluation. In steady state, the gather stream already waits for the previous region, so the event
    # adds no wait.
    weights_written = torch.cuda.current_stream().record_event()
    gathered = state.gathered_by_group[group]
    for bucket in gathered:
        bucket.start_gather(weights_written)
        bucket.finish_gather()
    # The gathers of this group start first, so the prefetch does not delay them.
    following = [] if next_group is None else state.gathered_by_group[next_group]
    for bucket in following:
        bucket.start_gather(weights_written)

    handles = state.handles_by_group[group]
    bind_parameters(handles, unsharded=True)
    try:
        yield
    finally:
        bind_parameters(handles, unsharded=False)
        for bucket in gathered:
            # A bucket that the next group uses too stays gathered, otherwise that group gathers it again.
            if bucket not in following:
                bucket.free_weights()


class BackwardPass:
    """The weights and the gradient reductions of one backward pass of a stage, region by region.

    Run the regions in reverse order of the forward pass, each inside ``region()``.
    """

    def __init__(self, state: StageState, final: bool | None):
        """Constructs the ``BackwardPass`` object.

        Args:
            state: The parameter state of the stage, with a recorded order.
            final: Whether this is the last backward pass of the stage in the step, or ``None`` for a pass that
                writes no weight gradients.
        """
        self._state = state
        self._final = final
        # The regions run in reverse order, each once: a skipped region would leave a gradient buffer without memory,
        # and a region out of order would reduce gradients that are not complete yet.
        self._next_index = len(cast(tuple[int | None, ...], state.order)) - 1

    @contextmanager
    def region(self, index: int) -> Generator[None, None, None]:
        """Puts the unsharded weights of a region into its modules and reduces the gradients that are complete after it.

        Args:
            index: The index of the region in the forward pass.

        Yields:
            Nothing.

        Raises:
            ValueError: If the runtime was unbound since, if the region is not the next one in reverse order, if it
                runs on another stream than ``bind()``, or if autograd replaced a gradient instead of accumulating it.
        """
        state = self._state
        state.require_bound()
        if index != self._next_index:
            raise ValueError(
                f"The backward pass of stage ({state.stage_index}) expects the region ({self._next_index}), not "
                f"({index}). Run the regions in reverse order of the forward pass, each once."
            )
        # StageParameters.backward() checked that the order is recorded, and it stays until unbind().
        order = cast(tuple[int | None, ...], state.order)
        group = order[index]
        next_group = order[index - 1] if index > 0 else None
        if self._final is not None:
            for bucket in state.prepare_at.get(index, ()):
                bucket.prepare_grads()
        with nullcontext() if group is None else _group_weights(state, group, next_group):
            # Without try: the reductions run only after a successful region.
            yield
        self._next_index -= 1
        if self._final is None:
            return
        for bucket in state.reduce_at.get(index, ()):
            bucket.reduce(self._final)
            if self._final:
                state.step.reduced.add(bucket)


class StageParameters:
    """The parameter groups of one bound stage."""

    def __init__(self, state: StageState):
        """Constructs the ``StageParameters`` object.

        ``ParameterRuntime.stage()`` returns it.

        Args:
            state: The parameter state of the stage.
        """
        self._state = state

    def group_index(self, module: nn.Module) -> int:
        """Returns the index of the parameter group of a module.

        Args:
            module: The root of the group.

        Returns:
            The index of the group.

        Raises:
            ValueError: If the runtime was unbound since, or if the module is not the root of a group of this stage.
        """
        self._state.require_bound()
        index = self._state.group_indices.get(module)
        if index is None:
            raise ValueError(
                f"The module ({type(module).__name__}) is not the root of a parameter group of stage "
                f"({self._state.stage_index}). Pass it in the groups of register_stage()."
            )
        return index

    @property
    def order(self) -> tuple[int | None, ...] | None:
        """The group of each region of the recorded forward pass, or ``None`` before ``record_order()``."""
        return self._state.order

    def record_order(self, order: Sequence[int | None]):
        """Records the group order of the first forward pass, or checks a later one against it.

        Args:
            order: The group of each region of the forward pass, including repeated groups, or ``None`` for a region
                without a group.

        Raises:
            ValueError: If the runtime was unbound since, if the order names a group that does not exist, if it
                differs from the recorded one, or if it omits parameters.
        """
        state = self._state
        state.require_bound()
        order = tuple(order)
        if state.order is not None:
            if state.order != order:
                raise ValueError(
                    f"The parameter group order ({order}) of stage ({state.stage_index}) differs from the recorded "
                    f"order ({state.order}). Use the same order for every microbatch and rank."
                )
            return
        for group in order:
            if group is not None:
                state.check_group(group)
        used = {id(handle) for group in order if group is not None for handle in state.handles_by_group[group]}
        omitted = [handle.plan.name for handle in state.handles if id(handle) not in used]
        if omitted:
            raise ValueError(
                f"The forward pass of stage ({state.stage_index}) omitted the parameters ({omitted}). Enter the group "
                "of every parameter on every forward pass."
            )

        for bucket in state.buckets:
            if not bucket.plan.trainable:
                continue
            uses = [index for index, group in enumerate(order) if group in bucket.plan.groups]
            # The first region in forward order ends the backward pass of the bucket.
            state.reduce_at.setdefault(uses[0], []).append(bucket)
            match bucket:
                case GatheredBucket():
                    # A tied bucket gets gradients in several regions, from its last one in forward order on.
                    state.prepare_at.setdefault(uses[-1], []).append(bucket)
                case ReplicatedBucket():
                    # Autograd accumulates into the buffer of the bucket over the whole step.
                    pass
        state.order = order

    @contextmanager
    def group_weights(self, group: int, next_group: int | None = None) -> Generator[None, None, None]:
        """Puts the unsharded weights of a group into its modules for a forward pass.

        It gathers the weights that a previous group did not prefetch or keep. On exit, it frees the weights that
        ``next_group`` does not use. Run backward passes in ``backward()`` instead.

        Args:
            group: The index of the group.
            next_group: The group that runs next, or ``None``. Its gather starts during this group.

        Yields:
            Nothing.

        Raises:
            ValueError: If the runtime was unbound since, or if a group does not exist.
        """
        state = self._state
        state.require_bound()
        state.check_group(group)
        if next_group is not None:
            state.check_group(next_group)
        with _group_weights(state, group, next_group):
            yield

    def backward(self, final: bool | None) -> BackwardPass:
        """Plans a backward pass of the stage over the recorded order.

        Args:
            final: Whether this is the last backward pass of the stage in the step, or ``None`` for a pass that
                writes no weight gradients, such as the input pass of a split backward pass.

        Returns:
            The backward pass.

        Raises:
            ValueError: If the runtime was unbound since, or if no order is recorded.
        """
        state = self._state
        state.require_bound()
        if state.order is None:
            raise ValueError(
                f"The stage ({state.stage_index}) has no recorded order. Call record_order() after the first forward "
                "pass."
            )
        return BackwardPass(state, final)
