import abc

from ..program.topology import ScheduleStyle
from ..runtime import ActionBase


class PipelineProgramBuilder(abc.ABC):
    """Abstract interface for building pipeline execution schedules."""

    @abc.abstractmethod
    def compose(self, num_microbatches: int, pp_size: int) -> dict[int, list[ActionBase]]:
        """
        Generates the execution program for all ranks in the pipeline.

        Args:
            num_microbatches: Number of microbatches per step.
            pp_size: Number of pipeline parallel ranks.

        Returns:
            A dictionary mapping rank indices to their list of sequential actions.
        """
        ...

    @property
    @abc.abstractmethod
    def num_stages_per_rank(self) -> int:
        """Returns the number of model stages designated for each rank."""

        ...

    @property
    @abc.abstractmethod
    def topology_style(self) -> ScheduleStyle:
        """Returns the topology style strategy used to assign stages to ranks."""
        ...
