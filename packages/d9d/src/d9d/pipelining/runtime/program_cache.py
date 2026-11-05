from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Iterator, Mapping
from itertools import groupby
from typing import TYPE_CHECKING, Literal, cast

from torch.distributed import ProcessGroup

from .action import (
    ActionBase,
    BackwardFullInputComputeAction,
    BackwardWeightComputeAction,
    BackwardWindowComputeAction,
    ComposeAction,
    ForwardComputeAction,
    ForwardWindowComputeAction,
)

if TYPE_CHECKING:
    from d9d.parallel.runtime import ParameterRuntime
    from d9d.pipelining.program.builder import PipelineProgramBuilder


@dataclasses.dataclass(frozen=True, slots=True)
class CachedPipelineProgram:
    """A composed pipeline program for one microbatch count.

    Attributes:
        program_this_rank: The action sequence this rank executes.
        has_backward: Whether the program contains backward work (derived from the composed actions).
        final_passes: Whether each pass that writes weight gradients is the last one of its stage, by stage and
            microbatch.
    """

    program_this_rank: list[ActionBase]
    has_backward: bool
    final_passes: Mapping[tuple[int, int], bool] = dataclasses.field(default_factory=dict)


def _leaf_actions(program: Iterable[ActionBase]) -> Iterator[ActionBase]:
    # ComposeAction.apply() runs nested actions at any depth, so the planning passes see the same leaves.
    for action in program:
        if isinstance(action, ComposeAction):
            yield from _leaf_actions(action.actions)
        else:
            yield action


def _window_key(action: ActionBase) -> tuple[Literal["forward", "backward"], int] | None:
    match action:
        case ForwardComputeAction():
            return "forward", action.stage_idx
        case BackwardFullInputComputeAction() if action.full_backward:
            return "backward", action.stage_idx
        case _:
            return None


def merge_windows(program: list[ActionBase]) -> list[ActionBase]:
    """Joins consecutive forward or full backward computes of one stage into windows.

    Args:
        program: The program of a rank with one stage.

    Returns:
        The program with windows. A program with several stages comes back unchanged.
    """
    # Consecutive microbatches of one stage share each gather, so the weights of a region arrive once per pass.
    flat = list(_leaf_actions(program))
    if len({x.stage_idx for x in flat if isinstance(x, ForwardComputeAction)}) > 1:
        return program

    result: list[ActionBase] = []
    for key, actions in groupby(flat, key=_window_key):
        run = list(actions)
        # One compute has nothing to share; unrelated actions keep their original order.
        if key is None or len(run) < 2:
            result.extend(run)
            continue
        computes = cast(list[ForwardComputeAction | BackwardFullInputComputeAction], run)
        direction, stage_index = key
        microbatches = tuple(action.microbatch_idx for action in computes)
        if direction == "forward":
            result.append(ForwardWindowComputeAction(stage_index, microbatches))
        else:
            result.append(BackwardWindowComputeAction(stage_index, microbatches))
    return result


def plan_final_passes(program: list[ActionBase]) -> Mapping[tuple[int, int], bool]:
    """Returns whether each action that writes weight gradients is the last one of its stage.

    Args:
        program: The program of a rank.

    Returns:
        The flag by stage and microbatch. Only the last such action of a stage is final.
    """
    # A gathered bucket reduces after each action that writes its weight gradients, the others after the last one.
    writes: list[tuple[int, int]] = []
    for action in _leaf_actions(program):
        match action:
            case BackwardWeightComputeAction():
                writes.append((action.stage_idx, action.microbatch_idx))
            case BackwardFullInputComputeAction() if action.full_backward:
                writes.append((action.stage_idx, action.microbatch_idx))
            case BackwardWindowComputeAction():
                writes.append((action.stage_idx, action.microbatches[-1]))
    last_write = {stage: microbatch for stage, microbatch in writes}
    return {(s, mb): last_write[s] == mb for s, mb in writes}


class PipelineProgramCache:
    """Cache of composed per-rank pipeline programs, keyed on the microbatch count.

    A program depends only on the microbatch count and the (fixed) pipeline topology, so each count is
    composed once and reused across steps.
    """

    def __init__(
        self, pp_group: ProcessGroup, builder: PipelineProgramBuilder, runtime: ParameterRuntime | None = None
    ):
        """Constructs the ``PipelineProgramCache`` object.

        Args:
            pp_group: The pipeline-parallel process group, used to find the position of this rank in it.
            builder: Builder that composes the all-rank action program for a microbatch count.
            runtime: The ``ParameterRuntime`` that owns the gradient reductions, or ``None``.
        """
        self._runtime = runtime
        self._pp_size = pp_group.size()
        self._pp_rank = pp_group.rank()
        self._builder = builder
        self._cache: dict[int, CachedPipelineProgram] = {}

    def program_for(self, num_microbatches: int) -> CachedPipelineProgram:
        """Returns the cached program for a microbatch count, composing it on first request.

        Args:
            num_microbatches: The number of microbatches in the step.

        Returns:
            The cached program for this rank and whether it contains backward work.
        """
        cache_value = self._cache.get(num_microbatches)
        if cache_value is None:
            program_all_ranks = self._builder.compose(num_microbatches=num_microbatches, pp_size=self._pp_size)
            has_backward = any(
                any(action.has_backward_work for action in sub_program) for sub_program in program_all_ranks.values()
            )
            program_this_rank = program_all_ranks[self._pp_rank]
            # With peers, a window can deadlock. Rank A holds stages 0 and 2, rank B holds 1 and 3. A delays the output
            # of stage 2 for microbatch 0 until it has the input of stage 2 for microbatch 1. B makes this input only
            # after stage 3 ran on microbatch 0, and that needs the delayed output. A window also stalls the next stage.
            if self._runtime is not None and self._pp_size == 1:
                program_this_rank = merge_windows(program_this_rank)
            cache_value = CachedPipelineProgram(
                program_this_rank=program_this_rank,
                has_backward=has_backward,
                final_passes={} if self._runtime is None else plan_final_passes(program_this_rank),
            )
            self._cache[num_microbatches] = cache_value

        return cache_value
