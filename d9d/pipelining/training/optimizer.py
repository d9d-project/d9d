from typing import Any

from torch.distributed import DeviceMesh

from d9d.core.protocol import OptimizerProtocol


class PipelinedOptimizer(OptimizerProtocol):
    """
    Wrapper that manages multiple optimizers for a pipeline parallel rank.

    In a pipeline parallel setup, a single rank might host multiple stages, each having its own parameters
    and optimizer.
    This class aggregates them into a single interface.
    """

    def __init__(self, mesh_pp: DeviceMesh | None, optimizers: list[OptimizerProtocol]):
        super().__init__()

        self._pp_rank = mesh_pp.get_local_rank() if mesh_pp is not None else 0
        self._optimizers = optimizers

    def state_dict(self) -> dict[str, Any]:
        pp_rank = self._pp_rank
        return {f"pp_{pp_rank}_stage_{i}": optimizer.state_dict() for i, optimizer in enumerate(self._optimizers)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pp_rank = self._pp_rank
        for i, optimizer in enumerate(self._optimizers):
            optimizer.load_state_dict(state_dict[f"pp_{pp_rank}_stage_{i}"])

    def step(self) -> None:
        for optimizer in self._optimizers:
            optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self._optimizers:
            optimizer.zero_grad()
