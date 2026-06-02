from typing import Any, cast

import torch
from torch.distributed import DeviceMesh

from d9d.core.offload import Offloadable, OffloadContext, OffloadedTensor, OnloadContext, offload_tensor, onload_tensor
from d9d.core.protocol import OptimizerProtocol


class PipelinedOptimizer(OptimizerProtocol, Offloadable):
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
        self._offload_mirror: dict[tuple[int, int, str], OffloadedTensor] | None = None

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

    def offload(self, ctx: OffloadContext) -> None:
        """
        Releases the GPU memory of the optimizer state, moving its tensors to host memory.

        Each tensor entry of "optimizer.state" has its local storage swapped in place to a host
        copy; the tensor objects (including DTensor wrappers) themselves are preserved, so the
        state dict keys and the wrappers held by the optimizer step continue to be valid.

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the optimizer state is already offloaded.
        """

        if self._offload_mirror is not None:
            raise RuntimeError("PipelinedOptimizer is already offloaded.")

        mirror: dict[tuple[int, int, str], OffloadedTensor] = {}
        for optimizer_index, optimizer in enumerate(self._optimizers):
            for param, param_state in cast(torch.optim.Optimizer, optimizer).state.items():
                for key, value in param_state.items():
                    if not torch.is_tensor(value):
                        continue
                    mirror[optimizer_index, id(param), key] = offload_tensor(value, pin_memory=ctx.pin_memory)

        # Drain the pending non-blocking device-to-host copies before the device storage is released.
        torch.cuda.synchronize(ctx.dist_context.current_device)
        self._offload_mirror = mirror

    def onload(self, ctx: OnloadContext) -> None:
        """
        Restores GPU residency of the optimizer state released by "offload".

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the optimizer state is not offloaded.
        """

        if self._offload_mirror is None:
            raise RuntimeError("PipelinedOptimizer is not offloaded.")

        device = ctx.dist_context.current_device
        for optimizer_index, optimizer in enumerate(self._optimizers):
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
        """Reports whether the optimizer state is currently on host memory."""
        return self._offload_mirror is not None
