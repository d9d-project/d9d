from typing import Any

import torch
from torch.profiler import record_function

from d9d.offload import Offloadable, OffloadContext, OffloadedTensor, OnloadContext, offload_tensor, onload_tensor
from d9d.optim import OptimizerProtocol
from d9d.rng import fork_rng_child


def _create_missing_state(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    params = [param for group in optimizer.param_groups for param in group["params"]]
    missing = [param for param in params if param.requires_grad and param not in optimizer.state]
    if not missing:
        return missing

    # Distributed Checkpoint saves and loads only tensors that exist, and an optimizer creates the state of a parameter
    # on its first step with a gradient. A step on zero gradients creates it, as _init_optim_state() in
    # torch.distributed.checkpoint.state_dict does. A zero learning rate keeps the parameters, a zero weight decay keeps
    # the L2 penalty out of the state, and the parameters without a gradient keep their state.
    grads = [param.grad for param in params]
    settings = [(group["lr"], group.get("weight_decay")) for group in optimizer.param_groups]
    for param in params:
        param.grad = None
    for param in missing:
        param.grad = torch.zeros_like(param)
    for group in optimizer.param_groups:
        group["lr"] = torch.zeros_like(group["lr"]) if torch.is_tensor(group["lr"]) else 0.0
        if "weight_decay" in group:
            group["weight_decay"] = 0.0

    # The step must not shift the numbers of the training, e.g. through stochastic rounding.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
        optimizer.step()

    for group, (learning_rate, weight_decay) in zip(optimizer.param_groups, settings, strict=True):
        group["lr"] = learning_rate
        if weight_decay is not None:
            group["weight_decay"] = weight_decay
    for param, grad in zip(params, grads, strict=True):
        param.grad = grad
    return missing


class PipelinedOptimizer(OptimizerProtocol, Offloadable):
    """Optimizer that combines the optimizers of all stages hosted on a pipeline-parallel rank.

    With pipeline parallelism, one rank can host several stages, each with its own parameters and
    optimizer. This class exposes them as one optimizer.
    """

    def __init__(self, optimizers: dict[int, torch.optim.Optimizer]):
        """Constructs the ``PipelinedOptimizer`` object.

        Args:
            optimizers: One optimizer per stage hosted on this rank, keyed by the global index of the stage.
        """
        super().__init__()

        self._optimizers = optimizers
        self._offload_mirror: dict[tuple[int, int, str], OffloadedTensor] | None = None

    def state_dict(self) -> dict[str, Any]:
        """Returns the state of every optimizer, with state for every trainable parameter.

        A parameter that has no state yet gets placeholder state, which ``placeholders`` lists by stage. The optimizers
        keep no placeholder state, so their steps stay the same as without the call.

        Returns:
            The state dict.
        """
        state_dict: dict[str, Any] = {}
        placeholders: dict[str, list[int]] = {}
        for stage_index, optimizer in self._optimizers.items():
            missing = _create_missing_state(optimizer)
            stage_state = optimizer.state_dict()
            indices = {
                id(param): index
                for group, packed in zip(optimizer.param_groups, stage_state["param_groups"], strict=True)
                for param, index in zip(group["params"], packed["params"], strict=True)
            }
            # The state dict keeps the new state, and the optimizer creates it again on the first real step.
            for param in missing:
                del optimizer.state[param]
            state_dict[f"stage_{stage_index}"] = stage_state
            placeholders[f"stage_{stage_index}"] = [indices[id(param)] for param in missing]
        state_dict["placeholders"] = placeholders
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state of every optimizer and drops the placeholder state.

        Args:
            state_dict: The state dict from ``state_dict()``.
        """
        for stage_index, optimizer in self._optimizers.items():
            stage_state = state_dict[f"stage_{stage_index}"]
            placeholders = set(state_dict["placeholders"][f"stage_{stage_index}"])
            state = {index: value for index, value in stage_state["state"].items() if index not in placeholders}
            optimizer.load_state_dict({**stage_state, "state": state})

    def step(self) -> None:
        with record_function("Optimizer Step"):
            for stage_index, optimizer in self._optimizers.items():
                with fork_rng_child("optimizer_step", stage_index):
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
            for param, param_state in optimizer.state.items():
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
            for param, param_state in optimizer.state.items():
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
