import abc

from ..runtime.action import ActionBase


class PipelineProgramBuilder(abc.ABC):
    """Abstract interface for building pipeline execution schedules."""

    @abc.abstractmethod
    def compose(self, num_stages: int, num_microbatches: int, pp_size: int) -> dict[int, list[ActionBase]]:
        """
        Generates the execution program for all ranks in the pipeline.

        Args:
            num_stages: Total number of model stages (chunks).
            num_microbatches: Number of microbatches per step.
            pp_size: Number of pipeline parallel ranks.

        Returns:
            A dictionary mapping rank indices to their list of sequential actions.
        """
        ...
