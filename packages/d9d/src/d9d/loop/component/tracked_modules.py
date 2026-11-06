import itertools
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.stateful import Stateful

from d9d.dist_context import DistributedContext
from d9d.offload import Offloadable, OffloadContext, OffloadedTensor, OnloadContext, offload_tensor, onload_tensor
from d9d.parallel.runtime import ParameterRuntime

StatefulPredicate = Callable[[str, torch.Tensor], bool]
"""Selects the parameters and buffers included in a checkpoint."""


class TrackedModules(Stateful, Offloadable):
    """Wrapper of the model stages of a rank that manages their state for distributed checkpointing.

    It implements the PyTorch Distributed ``Stateful`` protocol over all pipeline stages of the current
    rank. State keys hold the global index of each stage, so they are unique across ranks.
    """

    def __init__(
        self,
        dist_context: DistributedContext,
        modules: dict[int, nn.Module],
        stateful_predicate: StatefulPredicate,
        parameter_runtime: ParameterRuntime,
    ):
        """Constructs the ``TrackedModules`` object.

        Args:
            dist_context: The distributed context.
            modules: The model stages held by the current rank, keyed by the global index of their pipeline stage.
            stateful_predicate: Selects the parameters and buffers stored in the state dict.
            parameter_runtime: The runtime whose unsharded weights must be dropped before checkpoint loads.
        """
        self._dist_context = dist_context
        self._modules = modules
        self._stateful_predicate = stateful_predicate
        self._parameter_runtime = parameter_runtime
        self._offload_mirror: dict[int, OffloadedTensor] | None = None

    @property
    def modules(self) -> dict[int, nn.Module]:
        """The model stages held by the current rank, keyed by the global index of their pipeline stage."""
        return self._modules

    def _tensors(self) -> Iterator[torch.Tensor]:
        """Yields every parameter and buffer (persistent and non-persistent) of all tracked modules.

        Returns:
            An iterator over all tensors in the tracked modules.
        """
        return itertools.chain.from_iterable(
            itertools.chain(module.parameters(), module.buffers()) for module in self._modules.values()
        )

    def offload(self, ctx: OffloadContext) -> None:
        """Releases the GPU memory of all model parameters and buffers, moving them to host memory.

        Each tensor's local storage is swapped in place for a host copy. The parameter and buffer
        objects, including ``DTensor`` wrappers, are kept. So optimizer state keys, gradient hooks and
        external references (e.g. a frozen reference model held by a task) stay valid.

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the modules are already offloaded.
        """
        if self._offload_mirror is not None:
            raise RuntimeError("TrackedModules is already offloaded.")

        mirror: dict[int, OffloadedTensor] = {}
        for tensor in self._tensors():
            mirror[id(tensor)] = offload_tensor(tensor, pin_memory=ctx.pin_memory)

        # Drain the pending non-blocking device-to-host copies before the device storage is released.
        torch.cuda.synchronize(ctx.device)
        self._offload_mirror = mirror

    def onload(self, ctx: OnloadContext) -> None:
        """Restores all model parameters and buffers released by ``offload`` to the GPU.

        Args:
            ctx: Context for this operation.

        Raises:
            RuntimeError: If the modules are not offloaded.
        """
        if self._offload_mirror is None:
            raise RuntimeError("TrackedModules is not offloaded.")

        device = ctx.device
        for tensor in self._tensors():
            onload_tensor(tensor, self._offload_mirror[id(tensor)], device=device)

        # Drain pending host-to-device copies before the host mirror buffers are released.
        torch.cuda.synchronize(device)
        self._offload_mirror = None

    def is_offloaded(self) -> bool:
        """Reports whether the model parameters and buffers are offloaded to host memory.

        Returns:
            ``True`` if the model is offloaded, otherwise ``False``.
        """
        return self._offload_mirror is not None

    def _checkpoint_keys(self, module: nn.Module) -> set[str]:
        return {
            name
            for name, tensor in itertools.chain(module.named_parameters(), module.named_buffers())
            if self._stateful_predicate(name, tensor)
        }

    def state_dict(self) -> dict[str, Any]:
        """Generates the state dictionary for all tracked modules.

        The keys hold the global index of the pipeline stage, e.g. ``stage_0``, not the pipeline rank that holds it.
        Only parameters and buffers accepted by ``stateful_predicate`` are included.

        Returns:
            A dictionary containing the states of all managed modules.
        """
        state = {}
        for stage_index, module in self._modules.items():
            keys = self._checkpoint_keys(module)
            state[f"stage_{stage_index}"] = {
                name: tensor for name, tensor in module.state_dict().items() if name in keys
            }
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state dictionary into the tracked modules.

        Args:
            state_dict: The state dictionary to load. Must contain the keys of the stages held by this instance.

        Raises:
            ValueError: If keys accepted by ``stateful_predicate`` are missing or unexpected keys are present.
        """
        self._parameter_runtime.invalidate_weights()
        for stage_index, module in self._modules.items():
            stage_state = state_dict[f"stage_{stage_index}"]
            keys = self._checkpoint_keys(module)
            result = module.load_state_dict(stage_state, strict=False)
            missing = keys.intersection(result.missing_keys)
            unexpected = set(result.unexpected_keys)
            if missing:
                raise ValueError(f"The state dict is missing keys ({missing}).")
            if unexpected:
                raise ValueError(f"The state dict has unexpected keys ({unexpected}).")
