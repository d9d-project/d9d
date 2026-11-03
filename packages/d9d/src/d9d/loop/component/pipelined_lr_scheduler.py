from typing import Any

from d9d.optim import LRSchedulerProtocol


class PipelinedLRScheduler(LRSchedulerProtocol):
    """Learning rate scheduler that combines the schedulers of all stages hosted on a pipeline-parallel rank.

    Like ``PipelinedOptimizer``, it exposes the schedulers of several model stages as one scheduler.
    """

    def __init__(self, schedulers: dict[int, LRSchedulerProtocol]):
        """Constructs the ``PipelinedLRScheduler`` object.

        Args:
            schedulers: One LR scheduler per stage hosted on this rank, keyed by the global index of the stage.
        """
        self._schedulers = schedulers

    def state_dict(self) -> dict[str, Any]:
        return {f"stage_{stage_index}": scheduler.state_dict() for stage_index, scheduler in self._schedulers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for stage_index, scheduler in self._schedulers.items():
            scheduler.load_state_dict(state_dict[f"stage_{stage_index}"])

    def step(self) -> None:
        for scheduler in self._schedulers.values():
            scheduler.step()
