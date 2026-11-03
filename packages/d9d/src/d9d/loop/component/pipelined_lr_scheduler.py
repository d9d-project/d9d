from typing import Any

from torch.distributed import DeviceMesh

from d9d.optim import LRSchedulerProtocol


class PipelinedLRScheduler(LRSchedulerProtocol):
    """Learning rate scheduler that combines the schedulers of all stages hosted on a pipeline-parallel rank.

    Like ``PipelinedOptimizer``, it exposes the schedulers of several model stages as one scheduler.
    """

    def __init__(self, mesh_pp: DeviceMesh | None, schedulers: dict[int, LRSchedulerProtocol]):
        """Constructs the ``PipelinedLRScheduler`` object.

        Args:
            mesh_pp: The pipeline-parallel mesh, or ``None`` without pipeline parallelism.
            schedulers: One LR scheduler per stage hosted on this rank, keyed by the global index of the stage.
        """
        self._pp_rank = mesh_pp.get_local_rank() if mesh_pp is not None else 0
        self._schedulers = schedulers

    def state_dict(self) -> dict[str, Any]:
        return {
            f"pp_{self._pp_rank}_stage_{i}": scheduler.state_dict()
            for i, scheduler in enumerate(self._schedulers.values())
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for i, scheduler in enumerate(self._schedulers.values()):
            scheduler.load_state_dict(state_dict[f"pp_{self._pp_rank}_stage_{i}"])

    def step(self) -> None:
        for scheduler in self._schedulers.values():
            scheduler.step()
