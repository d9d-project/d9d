import dataclasses
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext

from ..bucket import AnyReduction, ReplicatedReduction, ShardedReduction
from .groups import ParameterGroups


@dataclasses.dataclass(frozen=True, slots=True)
class RegionPlan:
    """The recorded order of the regions of a module and the gradient work of each region in a backward pass.

    Attributes:
        order: The group of each region, or ``None`` for a region without a group.
        prepare_at: The reductions that allocate their gradient buffers before each region, by region index.
        reduce_at: The reductions that run after each region, by region index.
    """

    order: tuple[int | None, ...]
    prepare_at: Mapping[int, Sequence[ShardedReduction]]
    reduce_at: Mapping[int, Sequence[AnyReduction]]


def plan_regions(order: tuple[int | None, ...], reductions: Sequence[AnyReduction]) -> RegionPlan:
    """Plans the gradient work of the backward passes for a recorded order.

    Args:
        order: The group of each region, or ``None`` for a region without a group.
        reductions: The gradient reductions of the module, in reduction order.

    Returns:
        The plan of the regions.
    """
    prepare_at: dict[int, list[ShardedReduction]] = {}
    reduce_at: dict[int, list[AnyReduction]] = {}
    for reduction in reductions:
        uses = [index for index, group in enumerate(order) if group in reduction.groups]
        # The first region in forward order ends the backward pass of the bucket.
        reduce_at.setdefault(uses[0], []).append(reduction)
        match reduction:
            case ShardedReduction():
                # A tied bucket gets gradients in several regions, from its last one in forward order on.
                prepare_at.setdefault(uses[-1], []).append(reduction)
            case ReplicatedReduction():
                # Autograd accumulates into the buffer of the bucket over the whole step.
                pass
    return RegionPlan(order=order, prepare_at=prepare_at, reduce_at=reduce_at)


class BackwardPass:
    """One backward pass of a module, region by region."""

    def __init__(self, name: str, groups: ParameterGroups, regions: RegionPlan, final: bool | None):
        """Constructs the ``BackwardPass`` object.

        Args:
            name: The name of the module, which errors report.
            groups: The parameter groups of the module.
            regions: The plan of the regions.
            final: Whether this is the final backward pass of the step, or ``None`` for a pass that writes no weight
                gradients.
        """
        self._name = name
        self._groups = groups
        self._regions = regions
        self._final = final
        # The regions run in reverse order, each once: a skipped region would leave a gradient buffer without memory,
        # and a region out of order would reduce gradients that are not complete yet.
        self._next_index = len(regions.order) - 1

    @contextmanager
    def region(self, index: int) -> Generator[None, None, None]:
        """Puts the weights of a region into its modules, and reduces the complete gradients after the region.

        Args:
            index: The index of the region in the forward pass.

        Raises:
            ValueError: If the bound parameters are closed, if the region is not the next one in reverse order, if it
                runs on another stream than ``bind()``, or if autograd replaced a gradient instead of accumulating it.
        """
        self._groups.require_open()
        if index != self._next_index:
            raise ValueError(
                f"The backward pass of the module ({self._name}) expects the region ({self._next_index}), not "
                f"({index}). Run the regions in reverse order of the forward pass, each once."
            )
        order = self._regions.order
        group = order[index]
        next_group = order[index - 1] if index > 0 else None
        if self._final is not None:
            for reduction in self._regions.prepare_at.get(index, ()):
                reduction.prepare()
        with nullcontext() if group is None else self._groups.weights(group, next_group):
            # Without try: the reductions run only after a successful region.
            yield
        self._next_index -= 1
        if self._final is None:
            return
        for reduction in self._regions.reduce_at.get(index, ()):
            reduction.reduce(self._final)
