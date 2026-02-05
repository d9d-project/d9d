from typing import Any

from torch.distributed import DeviceMesh

from d9d.core.protocol import LRSchedulerProtocol


class PipelinedLRScheduler(LRSchedulerProtocol):
    """
    Wrapper that manages multiple LR schedulers for a pipeline parallel rank.

    Similar to `PipelinedOptimizer`, this aggregates schedulers corresponding to
    multiple model stages hosted on the current rank.
    """

    def __init__(self, mesh_pp: DeviceMesh | None, schedulers: list[LRSchedulerProtocol]):
        self._pp_rank = mesh_pp.get_local_rank() if mesh_pp is not None else 0
        self._schedulers = schedulers

    def state_dict(self) -> dict[str, Any]:
        return {
            f"pp_{self._pp_rank}_stage_{i}": scheduler.state_dict()
            for i, scheduler in enumerate(self._schedulers)
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for i, scheduler in enumerate(self._schedulers):
            scheduler.load_state_dict(state_dict[f"pp_{self._pp_rank}_stage_{i}"])

    def step(self) -> None:
        for scheduler in self._schedulers:
            scheduler.step()
