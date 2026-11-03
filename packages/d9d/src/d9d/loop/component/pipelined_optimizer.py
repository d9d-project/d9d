from typing import Any, cast

import torch
from torch.profiler import record_function

from d9d.offload import Offloadable, OffloadContext, OffloadedTensor, OnloadContext, offload_tensor, onload_tensor
from d9d.optim import OptimizerProtocol


class PipelinedOptimizer(OptimizerProtocol, Offloadable):
    """Optimizer that combines the optimizers of all stages hosted on a pipeline-parallel rank.

    With pipeline parallelism, one rank can host several stages, each with its own parameters and
    optimizer. This class exposes them as one optimizer.
    """

    def __init__(self, optimizers: dict[int, OptimizerProtocol]):
        """Constructs the ``PipelinedOptimizer`` object.

        Args:
            optimizers: One optimizer per stage hosted on this rank, keyed by the global index of the stage.
        """
        super().__init__()

        self._optimizers = optimizers
        self._offload_mirror: dict[tuple[int, int, str], OffloadedTensor] | None = None

    def state_dict(self) -> dict[str, Any]:
        return {f"stage_{stage_index}": optimizer.state_dict() for stage_index, optimizer in self._optimizers.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for stage_index, optimizer in self._optimizers.items():
            optimizer.load_state_dict(state_dict[f"stage_{stage_index}"])

    def step(self) -> None:
        with record_function("Optimizer Step"):
            for optimizer in self._optimizers.values():
                optimizer.step()

    def zero_grad(self) -> None:
        for optimizer in self._optimizers.values():
            optimizer.zero_grad()

    def offload(self, ctx: OffloadContext) -> None:
        """Releases the GPU memory of the optimizer state, moving its tensors to host memory.

        Each tensor in ``optimizer.state`` keeps its identity, including ``DTensor`` wrappers. Only its
        local storage moves to host memory. The state dict keys and the tensors held by the optimizer
        stay valid.

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the optimizer state is already offloaded.
        """
        if self._offload_mirror is not None:
            raise RuntimeError("PipelinedOptimizer is already offloaded.")

        mirror: dict[tuple[int, int, str], OffloadedTensor] = {}
        for optimizer_index, optimizer in enumerate(self._optimizers.values()):
            for param, param_state in cast(torch.optim.Optimizer, optimizer).state.items():
                for key, value in param_state.items():
                    if not torch.is_tensor(value):
                        continue
                    mirror[optimizer_index, id(param), key] = offload_tensor(value, pin_memory=ctx.pin_memory)

        # Drain the pending non-blocking device-to-host copies before the device storage is released.
        torch.cuda.synchronize(ctx.device)
        self._offload_mirror = mirror

    def onload(self, ctx: OnloadContext) -> None:
        """Moves the optimizer state released by ``offload`` back to GPU memory.

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the optimizer state is not offloaded.
        """
        if self._offload_mirror is None:
            raise RuntimeError("PipelinedOptimizer is not offloaded.")

        device = ctx.device
        for optimizer_index, optimizer in enumerate(self._optimizers.values()):
            for param, param_state in cast(torch.optim.Optimizer, optimizer).state.items():
                for key, value in param_state.items():
                    if not torch.is_tensor(value):
                        continue
                    offloaded = self._offload_mirror[optimizer_index, id(param), key]
                    onload_tensor(value, offloaded, device=device)

        # Drain pending host-to-device copies before the host mirror buffers are released.
        torch.cuda.synchronize(device)
        self._offload_mirror = None

    def is_offloaded(self) -> bool:
        """Reports whether the optimizer state is currently in host memory.

        Returns:
            ``True`` if the state is offloaded, ``False`` otherwise.
        """
        return self._offload_mirror is not None
