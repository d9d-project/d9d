import dataclasses
from typing import TYPE_CHECKING

from d9d.core.dist_context import REGULAR_DOMAIN, DistributedContext

from .action import ActionBase

if TYPE_CHECKING:
    from ..program import PipelineProgramBuilder


@dataclasses.dataclass(frozen=True, slots=True)
class CachedPipelineProgram:
    """A composed pipeline program for one microbatch count.

    Attributes:
        program_this_rank: The action sequence this rank executes.
        has_backward: Whether the program contains backward work (derived from the composed actions).
    """

    program_this_rank: list[ActionBase]
    has_backward: bool


class PipelineProgramCache:
    """Composes and caches per-rank pipeline programs, keyed on the microbatch count.

    A program depends only on the microbatch count and the (fixed) pipeline topology, so each count is
    composed once and reused across steps.
    """

    def __init__(self, dist_context: DistributedContext, builder: "PipelineProgramBuilder"):
        """Constructs a PipelineProgramCache.

        Args:
            dist_context: The distributed context, used to resolve this rank's position in the pp group.
            builder: Builder that composes the all-rank action program for a microbatch count.
        """
        pp_group = dist_context.mesh_for(REGULAR_DOMAIN).get_group("pp")
        self._pp_size = pp_group.size()
        self._pp_rank = pp_group.rank()
        self._builder = builder
        self._cache: dict[int, CachedPipelineProgram] = {}

    def program_for(self, num_microbatches: int) -> CachedPipelineProgram:
        """Returns the cached program for a microbatch count, composing it on first request.

        Args:
            num_microbatches: The number of microbatches in the step.

        Returns:
            The cached program for this rank plus whether it contains backward work.
        """
        cache_value = self._cache.get(num_microbatches)
        if cache_value is None:
            program_all_ranks = self._builder.compose(num_microbatches=num_microbatches, pp_size=self._pp_size)
            has_backward = any(
                any(action.has_backward_work for action in sub_program) for sub_program in program_all_ranks.values()
            )
            cache_value = CachedPipelineProgram(
                program_this_rank=program_all_ranks[self._pp_rank], has_backward=has_backward
            )
            self._cache[num_microbatches] = cache_value

        return cache_value
