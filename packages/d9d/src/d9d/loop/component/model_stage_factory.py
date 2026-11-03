import dataclasses
import itertools
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.stateful import Stateful

from d9d.dist_context import REGULAR_DOMAIN, DistributedContext
from d9d.model_state.io import load_model_state
from d9d.nn import ModuleLateInit
from d9d.offload import Offloadable, OffloadContext, OffloadedTensor, OnloadContext, offload_tensor, onload_tensor
from d9d.pipelining.api import PipelineStageInfo
from d9d.pipelining.factory import PipelineScheduleInfo, build_program_builder, build_schedule
from d9d.rng import fork_rng_child

from ..config import ModelStageFactoryConfig, PipeliningConfig
from ..control import InitializeModelStageContext, ModelProvider, ParallelizeModelStageContext

StatefulPredicate = Callable[[str, torch.Tensor], bool]
"""Determines if a specific parameter or buffer should be included in the state dictionary."""


def _stateful_predicate_requires_grad(key: str, value: torch.Tensor) -> bool:
    """Allows saving only tensors that require gradients.

    Returns:
        ``True`` if the tensor requires gradients, otherwise ``False``.
    """
    return value.requires_grad


def _stateful_predicate_always(key: str, value: torch.Tensor) -> bool:
    """Always allows saving.

    Returns:
        Always ``True``.
    """
    return True


class TrackedModules(Stateful, Offloadable):
    """Wrapper of the model stages of a rank that manages their state for distributed checkpointing.

    It implements the PyTorch Distributed ``Stateful`` protocol over all pipeline stages of the current
    rank. State keys hold the global index of each stage, so they are unique across ranks.
    """

    def __init__(
        self, dist_context: DistributedContext, modules: dict[int, nn.Module], stateful_predicate: StatefulPredicate
    ):
        """Constructs the ``TrackedModules`` object.

        Args:
            dist_context: The distributed context.
            modules: The model stages held by the current rank, keyed by the global index of their pipeline stage.
            stateful_predicate: Selects the parameters and buffers stored in the state dict.
        """
        self._dist_context = dist_context
        self._modules = modules
        self._stateful_predicate = stateful_predicate
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

    def _whitelisted_params(self, module: nn.Module) -> set[str]:
        allow_saving = set()
        for param_name, param in itertools.chain(module.named_parameters(), module.named_buffers()):
            if self._stateful_predicate(param_name, param):
                allow_saving.add(param_name)
        return allow_saving

    def _state_dict_stage(self, module: nn.Module) -> dict[str, Any]:
        whitelist = self._whitelisted_params(module)
        result = {k: v for k, v in module.state_dict().items() if k in whitelist}
        return result

    def state_dict(self) -> dict[str, Any]:
        """Generates the state dictionary for all tracked modules.

        The keys hold the global index of the pipeline stage, e.g. ``stage_0``, not the pipeline rank that holds it.
        Only parameters and buffers accepted by ``stateful_predicate`` are included.

        Returns:
            A dictionary containing the states of all managed modules.
        """
        return {f"stage_{stage_index}": self._state_dict_stage(module) for stage_index, module in self._modules.items()}

    def _load_state_dict_stage(self, module: nn.Module, state_dict: dict[str, Any]):
        whitelist = self._whitelisted_params(module)

        loading_result = module.load_state_dict(state_dict, strict=False)
        missing_keys = set(loading_result.missing_keys)
        extra_keys = set(loading_result.unexpected_keys)

        if len(whitelist.intersection(missing_keys)) > 0:
            raise ValueError(f"The state dict is missing keys ({whitelist.intersection(missing_keys)}).")
        if len(extra_keys) > 0:
            raise ValueError(f"The state dict has unexpected keys ({extra_keys}).")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Loads the state dictionary into the tracked modules.

        Args:
            state_dict: The state dictionary to load. Must contain the keys of the stages held by this instance.

        Raises:
            ValueError: If keys accepted by ``stateful_predicate`` are missing or unexpected keys are present.
        """
        for stage_index, module in self._modules.items():
            self._load_state_dict_stage(module, state_dict[f"stage_{stage_index}"])


@dataclasses.dataclass(frozen=True, slots=True)
class _TensorSlot:
    """A place that holds a parameter or a buffer: a module and the name of the tensor in it.

    Attributes:
        module: The module that holds the tensor.
        name: The name of the tensor in the module.
        is_buffer: Whether the tensor is a buffer rather than a parameter.
    """

    module: nn.Module
    name: str
    is_buffer: bool

    def get(self) -> torch.Tensor:
        """Returns the tensor that the slot holds now."""
        return self.module.get_buffer(self.name) if self.is_buffer else self.module.get_parameter(self.name)


def _tied_slots(model: nn.Module) -> list[list[_TensorSlot]]:
    """Returns the slots of each parameter or buffer held in several places, one list per tensor."""
    slots: dict[int, list[_TensorSlot]] = {}
    for module in model.modules():
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            slots.setdefault(id(parameter), []).append(_TensorSlot(module, name, is_buffer=False))
        for name, buffer in module.named_buffers(recurse=False, remove_duplicate=False):
            slots.setdefault(id(buffer), []).append(_TensorSlot(module, name, is_buffer=True))
    return [tensor_slots for tensor_slots in slots.values() if len(tensor_slots) > 1]


def _to_empty_keeping_ties(model: nn.Module, device: torch.device):
    """Moves the model to a device with uninitialized storage, like ``nn.Module.to_empty()``, and keeps its ties.

    Args:
        model: The model to move.
        device: The device to allocate the parameters and buffers on.
    """
    # to_empty() replaces the tensors of each module on its own, so it unties a tensor that several modules hold.
    ties = _tied_slots(model)
    model.to_empty(device=device)
    for first, *others in ties:
        shared = first.get()
        for slot in others:
            # nn.Module.__setattr__() replaces a registered parameter or buffer and keeps the persistence of a buffer.
            setattr(slot.module, slot.name, shared)


class ModelStageFactory:
    """Factory that creates, initializes and parallelizes model stages.

    It drives the ``ModelProvider`` to:

    1.  Initialize models on the meta device.
    2.  Apply horizontal parallelism (TP, DP, FSDP, etc.).
    3.  Materialize weights on the target device.
    4.  Load the initial model state from a checkpoint.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        dist_context: DistributedContext,
        config_model: ModelStageFactoryConfig,
        config_pipelining: PipeliningConfig,
    ):
        """Constructs the ``ModelStageFactory`` object.

        Args:
            model_provider: The provider that builds and parallelizes each model stage.
            dist_context: The distributed context.
            config_model: The model initialization configuration.
            config_pipelining: The pipeline parallelism configuration.
        """
        self._model_provider = model_provider
        self._dist_context = dist_context
        self._config_model = config_model
        self._config_pipelining = config_pipelining

    def _build_model_stage(self, stage: PipelineStageInfo) -> nn.Module:
        stage_index = stage.current_stage

        # Build on the meta device, so no memory is allocated yet.
        with fork_rng_child("initialize_model_stage", stage_index), torch.device("meta"):
            factored = self._model_provider.initialize_model_stage(
                InitializeModelStageContext(
                    dist_context=self._dist_context,
                    stage=stage,
                )
            )

        model = factored.model

        if not isinstance(model, ModuleLateInit) or not isinstance(model, nn.Module):
            raise ValueError(
                f"Model stage ({type(model).__name__}) must be an nn.Module that implements ModuleLateInit."
            )

        if self._dist_context.mesh_params.is_distributed:
            with fork_rng_child("parallelize_model_stage", stage_index):
                self._model_provider.parallelize_model_stage(
                    ParallelizeModelStageContext(model=model, stage=stage, dist_context=self._dist_context)
                )

        _to_empty_keeping_ties(model, self._dist_context.current_device)

        # to_empty allocates uninitialized storage, so initialize the parameters.
        with torch.no_grad(), fork_rng_child("reset_parameters", stage_index):
            model.reset_parameters()

        if self._config_model.source_checkpoint:
            load_model_state(
                src_dir=self._config_model.source_checkpoint,
                model=model,
                mapper=factored.state_mapper,
                device=f"cuda:{torch.cuda.current_device()}",
                position=self._dist_context.local_rank,  # Progress bar is displayed once per rank inside this node
            )

        model.train()

        return model

    def build_pipeline_and_modules(self) -> tuple[PipelineScheduleInfo, TrackedModules]:
        """Builds the pipeline schedule and the model stages.

        Returns:
            A tuple of the pipeline schedule info and the ``TrackedModules`` that wrap the model stages.
        """
        if self._config_model.checkpoint_only_trainable_parameters:
            stateful_predicate = _stateful_predicate_requires_grad
        else:
            stateful_predicate = _stateful_predicate_always

        pp_mesh = (
            self._dist_context.mesh_for(REGULAR_DOMAIN)["pp"] if self._dist_context.mesh_params.is_distributed else None
        )
        schedule, modules = build_schedule(
            pp_mesh=pp_mesh,
            program_builder=build_program_builder(self._config_pipelining.schedule),
            model_provider=self._build_model_stage,
        )

        return schedule, TrackedModules(self._dist_context, modules, stateful_predicate)
